# R2E-Gym

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/R2E-Gym)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-V1)

## Description

R2E-Gym is a real-world software engineering agent training environment built on 8,100+ executable coding problem environments with procedurally generated and hybrid-verified test suites. Each task presents the agent with a real GitHub repository containing a bug or missing feature, and the agent must explore the codebase, diagnose the issue, and produce a working patch. Tasks are drawn from real open-source repositories and cover bug fixes, feature implementations, and other software maintenance activities.

## Capabilities

- Real-world software engineering across diverse open-source repositories
- Bug diagnosis and fixing in production codebases
- Feature implementation from natural language specifications
- Codebase exploration and navigation
- Test-based verification of solutions

## Compute Requirements

Each task runs in an isolated Docker sandbox provisioned with 4 CPUs and 8GB of RAM. Per-task Docker images are pulled at runtime, each pre-configured with the target repository and its dependencies.

## License

[ORLv1](https://openreward.ai/orlv1.md).

## Tasks

There are two splits in this environment:

- **all**: ~8,100 tasks sourced from the [R2E-Gym-V1](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-V1) dataset. Each task corresponds to a real commit in an open-source repository, with a problem statement describing the issue and withheld tests for grading.
- **subset**: ~4,578 tasks sourced from the [R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) dataset. A curated subset of the full dataset.

Both splits are typed as `train` splits. Each task specifies a `repo_name`, `docker_image`, `commit_hash`, `problem_statement`, and `expected_output_json` for grading.

## Reward Structure

Rewards are binary and deterministic. The `answer` tool runs withheld pytest tests inside the sandbox and parses the test output into a mapping of test names to statuses (PASSED, FAILED, ERROR). This mapping is compared against the expected output via exact match:

- **1.0** if all test results match the expected output exactly.
- **0.0** if there is any mismatch in test count or individual test status.

No LLM graders are used. Grading is fully deterministic based on pytest output parsing.

## Data

Task data is loaded at runtime from HuggingFace datasets:
- [R2E-Gym/R2E-Gym-V1](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-V1) for the "all" split
- [R2E-Gym/R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) for the "subset" split

A `gold_patches.csv` file (42MB) is included in the repository for validation testing. It contains reference patches keyed by `commit_hash` that, when applied, should produce a reward of 1.0.

## Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `bash` | `command: str` | Execute a bash command in the sandbox's `/root/.venv` environment. Commands have a 600-second timeout by default. Returns stdout/stderr and exit code. |
| `answer` | *(none)* | Restores withheld tests, runs them via pytest, and computes the final score by comparing test results against expected output. Ends the episode. Can only be called once. |

## Time Horizon

R2E-Gym is a multi-turn environment. The agent receives a problem statement describing a bug or feature request, then iteratively uses the `bash` tool to explore the repository, understand the codebase, identify the relevant files, implement a fix, and verify the solution. When the agent is confident in its patch, it calls `answer` to run the withheld tests and receive a final score.

## Environment Difficulty

R2E-Gym tasks involve complex real-world repository modifications. The original R2E-Gym paper reports that agents trained on R2E-Gym achieve 51% on SWE-Bench Verified, representing the state of the art for open-weight software engineering agents.

## Other Environment Requirements

There are no external API keys required beyond OpenReward platform access. The environment uses the OpenReward sandbox API to provision per-task Docker containers.

## Safety

Agents operate in isolated Docker sandboxes with no access to the host system or external network beyond what is required for the task. Each sandbox is provisioned per-task and destroyed after the episode completes. The agent's actions are confined to the `/testbed` directory containing the target repository.

## Citations

```bibtex
@article{jain2025r2egym,
  title={R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents},
  author={Jain, Naman and Singh, Jaskirat and Shetty, Manish and Zheng, Liang and Sen, Koushik and Stoica, Ion},
  journal={arXiv preprint arXiv:2504.07164},
  year={2025}
}
```
