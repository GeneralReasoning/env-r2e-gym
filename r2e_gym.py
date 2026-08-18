import json
import os
import re
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


class SandboxUnavailableError(RuntimeError):
    """The task's sandbox is gone, so no command could be executed in it."""


_DEAD_SANDBOX_MARKERS = ("no such container", "is not running")


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

        self.or_client = AsyncOpenReward(api_key=secrets.get("api_key"))
        self.compute_settings = SandboxSettings(
            environment="Naman/R2E-Gym",
            image=self.validated.docker_image,
            machine_size="4:8"
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

        await self.sandbox.check_run(
            "cd /testbed && git status --porcelain | sed -n 's/^?? //p' >> .git/info/exclude"
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

        script = await self._grading_sandbox.check_run("cat /testbed/run_tests.sh")
        hardened = re.sub(r"(?<![\w/.])\.venv/", "/testbed/.venv/", script.strip())
        hardened = re.sub(r"(?<![\w/])r2e_tests(?![\w/])", "/r2e_tests", hardened)
        test_output, _ = await self._grading_sandbox.run(
            f"cd / && PYTHONPATH=/testbed bash -c {quote(hardened)}",
            timeout=self.validated.test_timeout,
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

        # Compare
        if len(parse_res) != len(expected):
            reward = 0.0
        else:
            # If ANY mismatch, reward = 0.0, else = 1.0
            match = True
            for k in parse_res.keys():
                if not k:
                    continue
                if k not in expected:
                    match = False
                    break
                if parse_res[k] != expected[k]:
                    match = False
                    break
            reward = 1.0 if match else 0.0

        note = (
            "\n\nNOTE: the withheld suite collected 0 tests. Scored 0.0, but "
            "this is a collection failure rather than a per-test comparison."
            if no_tests_collected
            else ""
        )
        return ToolOutput(
            metadata={
                "parse_res": parse_res,
                "expected": expected,
                "patch": patch,
                "no_tests_collected": no_tests_collected,
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
