# R2E-Gym

Quick setup for running the R2E-Gym environment server.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the server

```bash
python server.py
```

## Running tests

You'll need to run the tests with with OPENREWARD_API_KEY set

```bash
OPENREWARD_API_KEY=... pytest tests.py
```