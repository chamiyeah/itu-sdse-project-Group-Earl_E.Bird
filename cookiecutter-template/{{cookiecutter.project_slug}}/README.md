# {{ cookiecutter.project_name }}

{{ cookiecutter.description }}

Maintained by {{ cookiecutter.author_name }}.

## Setup

This project uses Python {{ cookiecutter.python_version }} and a `src` layout.

```bash
python{{ cookiecutter.python_version }} -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Test

```bash
pytest
```

## Project layout

- `src/{{ cookiecutter.package_name }}/`: application and training code
- `tests/`: automated tests
- `data/`: DVC-managed input data
- `artifacts/`: generated model outputs (not committed)
- `mlflow-state/`: local MLflow tracking state (not committed)
- `notebooks/`: exploratory notebooks

## Config

Training settings :
[`src/{{ cookiecutter.package_name }}/config.py`](src/{{ cookiecutter.package_name }}/config.py).
Keep credentials and object-store configuration in environment variables or a
local `.env` file; do not commit secrets.
