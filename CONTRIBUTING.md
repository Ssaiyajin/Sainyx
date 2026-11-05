# Contributing

Thanks for helping improve Sainyx. Keep changes focused and avoid committing credentials, checkpoints, raw datasets, or generated personal data.

## Before opening a pull request

1. Create a branch from `Dev` and describe the behavior change and how it was tested.
2. Install the pinned test dependencies with `python -m pip install -r requirements-dev.txt`, then run `python -m pytest -q tests`.
3. Keep model/checkpoint claims and dataset-rights statements evidence-based. Do not include private tokens in code, notebooks, screenshots, logs, or example commands.
4. Update API documentation or model cards when public behavior or model details change.

GitHub Actions runs tests and security checks. A successful CI run does not certify model quality, dataset licensing, or checkpoint provenance.
