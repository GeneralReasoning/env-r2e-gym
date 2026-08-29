import json
import os
import tempfile
from pathlib import Path
from shlex import quote
from typing import Any, Iterable, List, cast

from datasets import Dataset, load_dataset
from openreward import AsyncOpenReward, SandboxSettings
from openreward.environments import (Environment, JSONObject, TextBlock,
                                     ToolOutput, tool, Split)
from pydantic import BaseModel

from utils import decode_patch_bytes, decolor_dict_keys, parse_log

FULL_DATASET = cast(Dataset, load_dataset("R2E-Gym/R2E-Gym-V1", split="train"))
SUBSET_DATASET = cast(Dataset, load_dataset("R2E-Gym/R2E-Gym-Subset", split="train"))


# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1


class SandboxUnavailableError(RuntimeError):
    """The task's sandbox is gone, so no command could be executed in it."""


_DEAD_SANDBOX_MARKERS = ("no such container", "is not running")

_PYTEST_CONFIG_FILES = (
    "pytest.ini",
    "setup.cfg",
    "tox.ini",
    "pyproject.toml",
    "conftest.py",
)


def _patched_config_files(patch: str) -> list[str]:
    """Root-level pytest config files the submitted diff touches, if any.

    Anything untracked at setup() is in .git/info/exclude, so a config path can
    only reach here by being tracked or newly added by the diff itself.
    """
    names = set()
    for line in patch.splitlines():
        for prefix in ("--- a/", "+++ b/"):
            if line.startswith(prefix):
                path = line[len(prefix):].split("\t")[0].strip()
                if path in _PYTEST_CONFIG_FILES:
                    names.add(path)
    return sorted(names)


# The task id IS the upstream fix commit and the image leaves it reachable, so
# `git show <task id>` prints the gold patch and the withheld test.
_STRIP_HISTORY = r"""
set -eu
cd /testbed
# Keeps pre-existing untracked files out of the index, and so out of the patch.
git status --porcelain | sed -n 's/^?? //p' >> .git/info/exclude
git add -A >/dev/null
COMMIT=$(git -c user.email=env@r2e-gym.invalid -c user.name=R2E-Gym \
         commit-tree "$(git write-tree)" -m "Task base state")
git symbolic-ref HEAD refs/heads/__r2e_base
git update-ref refs/heads/__r2e_base "$COMMIT"
git for-each-ref --format='%(refname)' \
  | grep -vx refs/heads/__r2e_base \
  | while read -r ref; do git update-ref -d "$ref"; done
for remote in $(git remote); do git remote remove "$remote"; done
rm -rf .git/logs
git reflog expire --expire=now --expire-unreachable=now --all
git gc --prune=now --quiet
git prune --expire=now
"""


def _dead_sandbox_error(output: str) -> str | None:
    """Return the daemon's message if `output` is *only* a container-liveness error.

    Requiring it to be the whole output keeps a command that merely prints the
    string from being mistaken for a dead sandbox.
    """
    text = output.strip()
    if "\n" in text or not text.startswith("Error response from daemon:"):
        return None
    lowered = text.lower()
    return text if any(marker in lowered for marker in _DEAD_SANDBOX_MARKERS) else None


class BashParams(BaseModel, extra="forbid"):
    command: str

class ValidatedSpec(BaseModel, extra="forbid"):
    # Stable per-task identifier. Harnesses key their trial logs off
    # id/index/scenario_id/task_id and fall back to a timestamp when none is
    # present, which made every trial log indistinguishable ("trial_unknown_*")
    # and per-task reward attribution impossible. Note this model is
    # extra="forbid", so the field has to exist here for list_tasks() to be
    # able to emit it.
    id: str
    repo_name: str
    docker_image: str
    commit_hash: str
    prompt: str
    problem_statement: str
    expected_output_json: str
    bash_timeout: int | None = 600
    test_timeout: int | None = 1800
    download_model_patch: bool = True


class R2EGym(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = ValidatedSpec.model_validate(task_spec)

        # Scored submissions this session. answer() runs the withheld tests and
        # reports the outcome, and the agent holds the editor tools, so an
        # uncapped answer() is a free CI loop against the held-out suite. The
        # docstring already said this can only be called once; this enforces it.
        self.submitted = 0

        self.or_client = AsyncOpenReward(api_key=secrets.get("api_key"))
        # Every task is built from a public upstream fix commit, so the answer
        # lives at a stable URL for as long as the repo does -- see the leak note
        # on _STRIP_HISTORY, which closes only the copy inside /testbed.
        self.compute_settings = SandboxSettings(
            environment="Naman/R2E-Gym",
            image=self.validated.docker_image,
            machine_size="4:8",
            block_network=True,
        ) # changed to Naman
        self.sandbox = self.or_client.sandbox(self.compute_settings)

        self._grading_sandbox = self.or_client.sandbox(self.compute_settings)
        self._baseline_tree: str | None = None

    async def setup(self) -> None:
        await self.sandbox.start()
        await self.sandbox.check_run("git config --global --add safe.directory /testbed")

        # delete .pyc, __pycache__
        await self.sandbox.check_run("find /testbed -name '*.pyc' -delete")
        await self.sandbox.check_run("find /testbed -name '__pycache__' -delete")
        await self.sandbox.check_run("find /r2e_tests -name '*.pyc' -delete")
        await self.sandbox.check_run("find /r2e_tests -name '__pycache__' -delete")

        await self.sandbox.check_run("ln -s /testbed/.venv/bin/python /root/.local/bin/python")
        await self.sandbox.check_run("ln -s /testbed/.venv/bin/python /root/.local/bin/python3")
        await self.sandbox.check_run("ln -s /testbed/.venv/ /root/.venv")

        # Subsumes the .git/info/exclude line that used to be here.
        await self.sandbox.check_run(_STRIP_HISTORY)

        # A strip that quietly no-ops looks exactly like one that worked, until
        # it shows up as an inflated pass rate -- so fail setup instead.
        leak = await self.sandbox.run(
            f"cd /testbed && git cat-file -e {quote(self.validated.commit_hash)}^{{commit}}"
        )
        if leak.return_code == 0:
            raise RuntimeError(
                f"gold commit {self.validated.commit_hash} is still readable in "
                f"/testbed after stripping history for task {self.validated.id}; "
                "refusing to run a task whose answer is in the box."
            )

        self._baseline_tree = (
            await self.sandbox.check_run("cd /testbed && git add -A >/dev/null && git write-tree")
        ).strip()

        await self.sandbox.check_run("rm -rf /r2e_tests /testbed/run_tests.sh")

    async def teardown(self) -> None:
        await self.sandbox.stop()
        try:
            await self._grading_sandbox.stop()
        except Exception:
            pass

    async def get_prompt(self) -> List[TextBlock]:
        return [TextBlock(text=self.validated.problem_statement)]

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """
        Execute a bash command as an unprivileged user in the /testbed/.venv environment
        """
        output, code = await self.sandbox.run(
            f"source /root/.venv/bin/activate && {params.command.strip()}",
            timeout=self.validated.bash_timeout,
        )
        # A dead sandbox means we could not run the command, not that it failed;
        # raise so the platform can retry or mark the trial ungraded.
        dead = _dead_sandbox_error(output)
        if dead is not None:
            raise SandboxUnavailableError(
                f"sandbox for task {self.validated.id} is unavailable: {dead}"
            )
        return ToolOutput(
            metadata={"output": output, "exit_code": code},
            blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
            reward=0.0,
            finished=False,
        )

    @tool
    async def answer(self) -> ToolOutput:
        """
        Computes the final score. Executes the relevant unit and system tests, including withheld tests.
        This can only be called once, after all steps have been taken; only call this tool after you have finished all your steps and solved the coding issue.
        """
        if self.submitted > 0:
            return ToolOutput(
                blocks=[TextBlock(text="A solution has already been submitted for this task. "
                                       "This episode is over: it is not re-scored, and repeat "
                                       "submissions are penalised (reward -0.1).")],
                metadata={"already_submitted": True, "submission_count": self.submitted},
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        expected_json = json.loads(self.validated.expected_output_json)
        # Grading failures are deliberately NOT caught. Everything below is a
        # network round-trip to the sandbox (check_run / download / run), and a
        # failed round-trip means we could not grade -- not that the patch was
        # wrong. Letting it propagate lets the platform retry and, if it keeps
        # failing, mark the trial ungraded. This used to be wrapped in a bare
        # `except Exception` that returned reward=0.0, finished=True, so a
        # transient sandbox error was banked as a genuine patch failure.
        baseline = quote(cast(str, self._baseline_tree))
        await self.sandbox.check_run(
            f"cd /testbed && git add -A >/dev/null && git diff --cached --binary {baseline} > /tmp/model.patch"
        )
        patch_bytes = await self.sandbox.download("/tmp/model.patch")
        patch = decode_patch_bytes(patch_bytes)

        await self._grading_sandbox.start()
        await self._grading_sandbox.check_run("git config --global --add safe.directory /testbed")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "model.patch"
            temp_file.write_bytes(patch_bytes)
            await self._grading_sandbox.upload(temp_file, "/tmp/model.patch")

        applied = await self._grading_sandbox.run("cd /testbed && git apply /tmp/model.patch")
        if applied.return_code != 0:
            return ToolOutput(
                metadata={"error": "patch_did_not_apply", "detail": applied.output, "patch": patch},
                blocks=[TextBlock(text=f"Submitted changes could not be applied for grading:\n{applied.output}\n\nReward: 0.0")],
                reward=0.0,
                finished=True,
            )

        # run_tests.sh must run from the repo root with the withheld tests
        # reachable at the relative path it names, or pytest resolves a
        # different rootdir and silently drops the repo's own config.
        await self._grading_sandbox.check_run(
            "rm -rf /testbed/r2e_tests && ln -s /r2e_tests /testbed/r2e_tests"
        )
        # Grade under the config the expected output was recorded with: a diff
        # that edits it would rewrite the withheld suite's own selection.
        config = _patched_config_files(patch)
        if config:
            await self._grading_sandbox.check_run(
                "cd /testbed && for f in " + " ".join(quote(c) for c in config) + "; do "
                'git ls-files --error-unmatch "$f" >/dev/null 2>&1 '
                '&& git checkout -- "$f" || rm -f "$f"; done'
            )
        graded = await self._grading_sandbox.run(
            "cd /testbed && bash run_tests.sh",
            timeout=self.validated.test_timeout,
            max_bytes=None,
        )
        test_output = graded.output

        if graded.timed_out:
            raise RuntimeError(
                f"Withheld test suite for task {self.validated.id} timed out after "
                f"{self.validated.test_timeout}s; cannot grade this rollout."
            )
        if graded.truncated:
            raise RuntimeError(
                f"Withheld test suite output for task {self.validated.id} was truncated; "
                "the reported results are incomplete and cannot be graded."
            )

        # No output at all means the suite never ran (sandbox died, script
        # missing, timeout before first write) rather than "every test failed".
        # Raise so the platform can retry instead of banking a 0.0.
        if not test_output or not test_output.strip():
            raise RuntimeError(
                "Withheld test suite produced no output; cannot grade this "
                "rollout. Treating as ungraded rather than scoring it 0.0."
            )

        parse_res = parse_log(test_output)
        parse_res = decolor_dict_keys(parse_res)
        parse_res = {k.split(" - ")[0]: parse_res[k] for k in sorted(parse_res.keys())}

        expected = decolor_dict_keys(expected_json)
        expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}

        # A run that collected zero tests is not the same thing as a run where
        # every test failed, but both used to land on an indistinguishable 0.0.
        # We still score it 0.0 -- a patch that breaks collection outright is a
        # real failure, and making it ungraded would let an agent dodge the
        # penalty by breaking the suite -- but flag it so it is separable in
        # analysis.
        no_tests_collected = not parse_res and bool(expected)

        # Score over the tests the expected mapping names. Every one of them has
        # to be present and agree: a patch that suppresses a test leaves its key
        # missing and still fails. A test the mapping does not name has no status
        # to compare against, so it cannot be graded either way -- dev-mode-only
        # tests, for instance, whose selection depends on how pytest is invoked.
        missing = [k for k in expected if k and k not in parse_res]
        mismatched = [k for k in expected if k and k in parse_res and parse_res[k] != expected[k]]
        ungraded_tests = [k for k in parse_res if k and k not in expected]
        reward = 0.0 if missing or mismatched else 1.0

        note = (
            "\n\nNOTE: the withheld suite collected 0 tests. Scored 0.0, but "
            "this is a collection failure rather than a per-test comparison."
            if no_tests_collected
            else ""
        )
        if ungraded_tests:
            note += (
                "\n\nNOTE: not scored, absent from the expected results: "
                + ", ".join(ungraded_tests)
            )
        # A patch that would not apply returns above without running the tests, so
        # it does not consume the attempt.
        self.submitted += 1

        return ToolOutput(
            metadata={
                "parse_res": parse_res,
                "expected": expected,
                "patch": patch,
                "no_tests_collected": no_tests_collected,
                "grading_return_code": graded.return_code,
                "missing_tests": missing,
                "mismatched_tests": mismatched,
                "ungraded_tests": ungraded_tests,
            },
            blocks=[TextBlock(text=f"Test Results:\n{json.dumps(parse_res, indent=2)}\n\nExpected:\n{json.dumps(expected, indent=2)}\n\nReward: {reward}{note}")],
            reward=reward,
            finished=True,
        )

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "all":
            df = FULL_DATASET
        elif split == "subset":
            df = SUBSET_DATASET
        validated_spec = [
            ValidatedSpec(
                id=f"{r['repo_name']}-{r['commit_hash'][:12]}",
                repo_name=r["repo_name"],
                docker_image=r["docker_image"],
                commit_hash=r["commit_hash"],
                prompt=r["prompt"],
                problem_statement=r["problem_statement"],
                expected_output_json=r["expected_output_json"],
            ) for r in cast(Iterable[dict[str, Any]], df)
        ]
        return [v.model_dump() for v in validated_spec]

    @classmethod
    def list_splits(cls) -> list[Split]:
        return [
            Split(name="all", type="train"),
            Split(name="subset", type="train"),
        ]
