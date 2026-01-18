import json
import os
import traceback
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

class BashParams(BaseModel, extra="forbid"):
    command: str

class ValidatedSpec(BaseModel, extra="forbid"):
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

        api_key = secrets.get("OPENREWARD_API_KEY") or secrets.get("API_KEY")
        if not api_key:
            raise ValueError("OPENREWARD_API_KEY or API_KEY is not set")

        self.or_client = AsyncOpenReward(api_key=api_key)
        self.compute_settings = SandboxSettings(
            environment="GeneralReasoning/R2E-Gym",
            image=self.validated.docker_image,
            machine_size="4:8"
        )
        self.computer = self.or_client.sandbox(self.compute_settings)

        self._hidden_dir = Path("/var/lib/.r2e_gym")
        self._hidden_tests_tar = self._hidden_dir / f".r2e_tests.{self.validated.commit_hash}.tar"
        self._hidden_run_tests = self._hidden_dir / f".run_tests.{self.validated.commit_hash}.sh"

    async def setup(self) -> None:
        await self.computer.start()
        await self.computer.check_run("git config --global --add safe.directory /testbed")

        # delete .pyc, __pycache__
        await self.computer.check_run("find /testbed -name '*.pyc' -delete")
        await self.computer.check_run("find /testbed -name '__pycache__' -delete")
        await self.computer.check_run("find /r2e_tests -name '*.pyc' -delete")
        await self.computer.check_run("find /r2e_tests -name '__pycache__' -delete")

        # symlinks, so we can run tests from /root
        await self.computer.check_run("ln -s /testbed/.venv/bin/python /root/.local/bin/python")
        await self.computer.check_run("ln -s /testbed/.venv/bin/python /root/.local/bin/python3")
        await self.computer.check_run("ln -s /testbed/.venv/ /root/.venv")
        await self.computer.check_run("ln -s /root/r2e_tests /testbed/r2e_tests")

        hidden_dir_q = quote(str(self._hidden_dir))
        hidden_tests_tar_q = quote(str(self._hidden_tests_tar))
        hidden_run_tests_q = quote(str(self._hidden_run_tests))
        await self.computer.check_run(
            " && ".join(
                [
                    f"mkdir -p {hidden_dir_q}",
                    f"chmod 700 {hidden_dir_q}",
                    f"tar -cf {hidden_tests_tar_q} -C / r2e_tests",
                    f"chmod 600 {hidden_tests_tar_q}",
                    f"mv /testbed/run_tests.sh {hidden_run_tests_q}",
                    f"chmod 700 {hidden_run_tests_q}",
                    "rm -rf /r2e_tests",
                ]
            )
        )

    async def teardown(self) -> None:
        await self.computer.stop()

    async def get_prompt(self) -> List[TextBlock]:
        return [TextBlock(text=self.validated.problem_statement)]

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """
        Execute a bash command in the in the /root/.venv environment
        """
        output, code = await self.computer.run(f"source /root/.venv/bin/activate && {params.command.strip()}", timeout=self.validated.bash_timeout)
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
        try:
            # extract patch for logging
            if self.validated.download_model_patch:
                await self.computer.check_run(f"git add -A && git diff --cached {self.validated.commit_hash} > /testbed/model.patch")
                patch_bytes = await self.computer.download("/testbed/model.patch")
                patch = decode_patch_bytes(patch_bytes)
            else:
                patch = ""

            # Restore withheld tests for grading only
            hidden_tests_tar_q = quote(str(self._hidden_tests_tar))
            hidden_run_tests_q = quote(str(self._hidden_run_tests))
            await self.computer.check_run(f"cp {hidden_run_tests_q} /root/run_tests.sh && chmod 700 /root/run_tests.sh")
            await self.computer.check_run(f"tar -xf {hidden_tests_tar_q} -C /root")

            test_output, _ = await self.computer.run("bash /root/run_tests.sh", timeout=self.validated.test_timeout)

            parse_res = parse_log(test_output)
            parse_res = decolor_dict_keys(parse_res)
            parse_res = {k.split(" - ")[0]: parse_res[k] for k in sorted(parse_res.keys())}

            expected = decolor_dict_keys(expected_json)
            expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}

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
            return ToolOutput(
                metadata={"parse_res": parse_res, "expected": expected, "patch": patch},
                blocks=[TextBlock(text=f"Test Results:\n{json.dumps(parse_res, indent=2)}\n\nExpected:\n{json.dumps(expected, indent=2)}\n\nReward: {reward}")],
                reward=reward,
                finished=True,
            )
        except Exception:
            return ToolOutput(
                metadata={"error": traceback.format_exc()},
                blocks=[TextBlock(text=f"Error during evaluation:\n{traceback.format_exc()}")],
                reward=0.0,
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
