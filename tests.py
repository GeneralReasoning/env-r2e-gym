import os
import tempfile
from pathlib import Path
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

        res: ToolOutput = await env.answer()
        assert res.reward == 0, f"Expected reward of 0, got {res.reward}, full output: {res}"
        assert res.finished
    finally:
        await env.teardown()
