from pathlib import Path
import tempfile
from typing import cast
import pytest
import pandas as pd

from environments.r2e_gym.server import R2EGym, BashParams
from matrix.server import JSONObject, ToolOutput

tasks = R2EGym.get_tasks("all")
EXAMPLE_R2E_TASK = tasks[0]

@pytest.mark.asyncio
async def test_r2e_bash():
    env = R2EGym(task_spec=EXAMPLE_R2E_TASK)
    try:
        await env.setup()
        output: ToolOutput = await env.bash(BashParams(command="whoami"))
        output_value = output.data["output"]
        assert isinstance(output_value, str)
        output_str = cast(str, output_value)
        assert "root" in output_str, f"Expected 'root' in output, got {output_str}"
    finally:
        await env.teardown()

GOLD_PATCHES = pd.read_csv(Path(__file__).parent / "gold_patches.csv")

@pytest.mark.asyncio
@pytest.mark.parametrize("task", tasks)
async def test_r2e_gold(task: JSONObject):
    env = R2EGym(task_spec=task)
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
async def test_r2e_xfail_state(task: JSONObject):
    env = R2EGym(task_spec=task)
    try:
        await env.setup()

        res: ToolOutput = await env.answer()
        assert res.reward == 0, f"Expected reward of 0, got {res.reward}, full output: {res}"
        assert res.finished
    finally:
        await env.teardown()
