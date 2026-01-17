import json
import os
from shlex import quote
import traceback
from pathlib import Path
from typing import Any, Iterable, cast

from datasets import Dataset, load_dataset
from pydantic import BaseModel

from construct.client import Computer, ComputeSettings
from matrix.server import Environment, JSONObject, Server, TextOutput, ToolOutput, tool
from environments.r2e_gym.utils import parse_log, decolor_dict_keys, decode_patch_bytes


GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "indigo-idea-457514-b5")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
GCP_REPOSITORY = os.getenv("GCP_REPOSITORY", "r2e-proxy")

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
    def __init__(self, task_spec: JSONObject) -> None:
        super().__init__(task_spec)        
        self.validated = ValidatedSpec.model_validate(task_spec)
        
        image = f"{GCP_REGION}-docker.pkg.dev/{GCP_PROJECT_ID}/{GCP_REPOSITORY}/{self.validated.docker_image}"

        self.compute_settings = ComputeSettings(
            image=image,
            cpu_request="4",
            cpu_limit="4",
            memory_request="8G",
            memory_limit="8G",
        )
        self.computer = Computer(self.compute_settings)
        self._hidden_dir = Path("/var/lib/.r2e_gym")
        self._hidden_tests_tar = self._hidden_dir / f".r2e_tests.{self.validated.commit_hash}.tar"
        self._hidden_run_tests = self._hidden_dir / f".run_tests.{self.validated.commit_hash}.sh"

    async def setup(self) -> None:
        await self.computer.__aenter__()
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
        await self.computer.__aexit__(None, None, None)

    def get_prompt(self) -> str:
        return self.validated.problem_statement

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """
        Execute a bash command in the in the /root/.venv environment
        """
        output, code = await self.computer.run(f"source /root/.venv/bin/activate && {params.command.strip()}", timeout=self.validated.bash_timeout)
        return ToolOutput(
            data={"output": output, "exit_code": code},
            raw=TextOutput(text=f"{output}\n\n(exit {code})"),
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
                data={"parse_res": parse_res, "expected": expected, "patch": patch},
                reward=reward,
                finished=True,
            )
        except Exception:
            return ToolOutput(
                data={"error": traceback.format_exc()},
                reward=0.0,
                finished=True,
            )

    @classmethod
    def get_tasks(cls, split: str) -> list[JSONObject]:
        if split not in cls.get_splits():
            raise ValueError(f"Unknown split: {split}")

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
    def get_splits(cls) -> list[str]:
        return ["all", "subset"]

if __name__ == "__main__":
    Server([R2EGym]).run()