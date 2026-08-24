# Repository guidance

## Project overview

- This repository contains LangGraph Academy notebooks and LangGraph Studio examples.
- `module-0` through `module-6` contain course material organized by topic.
- `module-*/studio` contains runnable LangGraph Studio applications.
- Keep examples approachable for learners and preserve the instructional intent.

## Environment and setup

- Use Python 3.11 when possible; it matches the development container.
- Install the root dependencies with `pip install -r requirements.txt`.
- Some Studio directories have an additional `requirements.txt`; install it when working in that directory.
- Copy `.env.example` to `.env` and fill in the required credentials locally.
- Never commit API keys, access tokens, passwords, or other secrets.

## Editing guidelines

- Make focused changes and avoid unrelated cleanup.
- Preserve notebook outputs unless the task explicitly requires executing or clearing them.
- Do not edit files inside `lc-academy-env/`; it is a local virtual environment.
- Keep Python examples and comments concise and suitable for an educational repository.
- Do not add a new production dependency unless it is necessary for the requested change.

## Verification

- Verify changed Python files with an appropriate targeted command, such as `python -m compileall <changed-files>`.
- When changing a Studio graph, run `langgraph dev` from the corresponding `module-*/studio` directory when practical.
- Do not execute API-backed notebooks unless the required credentials and network access are available.
- Report checks that could not be run and the reason.
