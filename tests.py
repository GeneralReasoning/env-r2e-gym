import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from openreward.environments import JSONObject, ToolOutput

from r2e_gym import BashParams, R2EGym, _dead_sandbox_error

OPENREWARD_API_KEY = os.getenv("OPENREWARD_API_KEY", "")

tasks = R2EGym.list_tasks("all")
EXAMPLE_R2E_TASK = tasks[0]

# Needs no sandbox and no API key: the detector is pure text handling.
@pytest.mark.parametrize("output", [
    "Error response from daemon: No such container: orshim-fb61de1a67ca",
    "  Error response from daemon: No such container: orshim-fb61de1a67ca\n",
    "Error response from daemon: Container orshim-fb61de1a67ca is not running",
])
def test_dead_sandbox_error_detected(output: str):
    assert _dead_sandbox_error(output) is not None

@pytest.mark.parametrize("output", [
    "",
    "bash: line 1: nosuchcmd: command not found",
    # A command whose output merely quotes the daemon error.
    "Error response from daemon: No such container: orshim-fb61de1a67ca\nnext line",
    "$ cat err.log\nError response from daemon: No such container: abc",
    # A daemon error unrelated to container liveness.
    "Error response from daemon: conflict: unable to remove image",
])
def test_live_sandbox_output_not_flagged(output: str):
    assert _dead_sandbox_error(output) is None

class _FakeSandbox:
    """Answers answer()'s sandbox calls without a sandbox: `git apply` exits
    with the next code in `apply_codes`, and the test run prints `test_log`."""

    def __init__(self, apply_codes: list[int] = [0], test_log: str = ""):
        self.apply_codes, self.test_log = list(apply_codes), test_log
        self.starts = self.stops = 0

    async def start(self):
        self.starts += 1

    async def stop(self):
        self.stops += 1

    async def check_run(self, cmd: str, **kwargs):
        return ""

    async def download(self, path: str):
        return b"diff --git a/f.py b/f.py\n"

    async def upload(self, local_path, container_path: str):
        pass

    async def run(self, cmd: str, **kwargs):
        if "git apply" in cmd:
            output, code = "error: No valid patches in input", self.apply_codes.pop(0)
        else:
            output, code = self.test_log, 0
        return SimpleNamespace(output=output, return_code=code, timed_out=False, truncated=False)


def _fake_env(expected: dict[str, str], grading: _FakeSandbox) -> R2EGym:
    env = R2EGym(
        task_spec={**EXAMPLE_R2E_TASK, "expected_output_json": json.dumps(expected)},
        secrets={"api_key": "unused"},
    )
    env.sandbox, env._grading_sandbox = _FakeSandbox(), grading
    env._baseline_tree = "0" * 40
    return env

EXPECTED = {"test_withheld_alpha": "PASSED", "test_withheld_beta": "FAILED"}
MATCHING_LOG = (
    "=== short test summary info ===\n"
    "PASSED r2e_tests/test_x.py::test_withheld_alpha\n"
    "FAILED r2e_tests/test_x.py::test_withheld_beta - AssertionError\n"
)

# Goes through _call_tool, the server path, so the SDK's end-of-episode latch applies.
@pytest.mark.asyncio
async def test_unappliable_patch_leaves_episode_open():
    grading = _FakeSandbox(apply_codes=[1, 0], test_log=MATCHING_LOG)
    env = _fake_env(EXPECTED, grading)

    first = (await env._call_tool("answer", {})).root
    assert first.ok
    assert first.output.reward == 0.0 and not first.output.finished
    assert first.output.metadata["error"] == "patch_did_not_apply"
    assert grading.stops == 1

    second = (await env._call_tool("answer", {})).root
    assert second.ok
    assert second.output.reward == 1.0 and second.output.finished
    assert grading.starts == 2

@pytest.mark.asyncio
@pytest.mark.parametrize("log, reward", [
    (MATCHING_LOG, 1.0),
    (MATCHING_LOG.replace("FAILED", "PASSED"), 0.0),
], ids=["right", "wrong"])
async def test_graded_result_hides_withheld_tests(log: str, reward: float):
    env = _fake_env(EXPECTED, _FakeSandbox(test_log=log))
    out = (await env._call_tool("answer", {})).root.output
    assert out.reward == reward and out.finished
    payload = json.dumps({
        "blocks": [b.model_dump() for b in out.blocks],
        "metadata": out.metadata, "reward": out.reward, "finished": out.finished,
    })
    for name in EXPECTED:
        assert name not in payload
    assert out.metadata["graded_tests"] == 2
    assert out.metadata["matched_tests"] == (2 if reward == 1.0 else 1)

@pytest.mark.asyncio
@pytest.mark.skipif(not OPENREWARD_API_KEY, reason="OPENREWARD_API_KEY is not set")
async def test_gold_commit_unreachable_after_setup():
    """setup() raises on this itself; the test pins the property so a rewrite
    of the strip cannot quietly weaken it."""
    env = R2EGym(task_spec=EXAMPLE_R2E_TASK, secrets={"OPENREWARD_API_KEY": OPENREWARD_API_KEY})
    try:
        await env.setup()
        commit = env.validated.commit_hash
        probe = await env.bash(BashParams(
            command=f"cd /testbed && git cat-file -e {commit}^{{commit}} "
                    "&& echo READABLE || echo GONE"
        ))
        assert "GONE" in cast(str, probe.metadata["output"])
        # One parentless commit and nothing else: no second route to the fix
        # through a ref, a reflog, or an unpruned object.
        history = await env.bash(BashParams(
            command="cd /testbed && git rev-list --all | wc -l"
        ))
        assert cast(str, history.metadata["output"]).strip() == "1"
    finally:
        await env.teardown()

@pytest.mark.asyncio
@pytest.mark.skipif(not OPENREWARD_API_KEY, reason="OPENREWARD_API_KEY is not set")
async def test_r2e_bash():
    env = R2EGym(task_spec=EXAMPLE_R2E_TASK, secrets={"OPENREWARD_API_KEY": OPENREWARD_API_KEY})
    try:
        await env.setup()
        output: ToolOutput = await env.bash(BashParams(command="whoami"))
        output_value = output.metadata["output"]
        assert isinstance(output_value, str)
        output_str = cast(str, output_value)
        assert "root" in output_str, f"Expected 'root' in output, got {output_str}"
    finally:
        await env.teardown()

GOLD_PATCHES = pd.read_csv(Path(__file__).parent / "gold_patches.csv")

@pytest.mark.asyncio
@pytest.mark.parametrize("task", tasks)
@pytest.mark.skipif(not OPENREWARD_API_KEY, reason="OPENREWARD_API_KEY is not set")
async def test_r2e_gold(task: JSONObject):
    env = R2EGym(task_spec=task, secrets={"OPENREWARD_API_KEY": OPENREWARD_API_KEY})
    try:
        await env.setup()

        # get gold patch
        gold_patch = GOLD_PATCHES[GOLD_PATCHES["commit_hash"] == task["commit_hash"]]["patch"]
        if len(gold_patch) == 0:
            pytest.skip(f"No gold patch found for task {task['commit_hash']}")
        gold_patch = str(list(gold_patch)[0])

        # apply gold patch
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "model.patch"
            temp_file.write_text(gold_patch)
            await env.computer.upload(temp_file, "/tmp/model.patch")
        await env.computer.check_run(f"git apply /tmp/model.patch")

        res: ToolOutput = await env.answer()
        assert res.reward == 1, f"Expected reward of 1, got {res.reward}, full output: {res}"
        assert res.finished
    finally:
        await env.teardown()

@pytest.mark.asyncio
@pytest.mark.parametrize("task", tasks)
@pytest.mark.skipif(not OPENREWARD_API_KEY, reason="OPENREWARD_API_KEY is not set")
async def test_r2e_xfail_state(task: JSONObject):
    env = R2EGym(task_spec=task, secrets={"OPENREWARD_API_KEY": OPENREWARD_API_KEY})
    try:
        await env.setup()

        # An untouched repo is an empty diff: not graded, and the episode stays open.
        res: ToolOutput = await env.answer()
        assert res.reward == 0, f"Expected reward of 0, got {res.reward}, full output: {res}"
        assert not res.finished
        assert res.metadata["error"] == "patch_did_not_apply"

        # A change that leaves the bug in place is graded and fails.
        await env.bash(BashParams(command="echo unrelated > /testbed/r2e_unrelated_change.txt"))
        res = await env.answer()
        assert res.reward == 0, f"Expected reward of 0, got {res.reward}, full output: {res}"
        assert res.finished
    finally:
        await env.teardown()
