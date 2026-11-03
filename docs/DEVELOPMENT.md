# Sainyx development guide

## Requirements

Use Python 3.11, matching the pre-deployment validation workflow. Install
the pinned PyTorch and torchvision pair for the local platform before
installing the remaining requirements. The following Windows example uses
CPU wheels:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

For Linux/macOS, activate the virtual environment with its platform-specific
command. For CUDA, install the matching official PyTorch wheel while keeping
the repository's pinned versions.

## Run and configure

```powershell
python app.py
```

The app listens on `http://localhost:7860`. Model checkpoints are not stored
in Git; they are fetched from the Hugging Face Hub when needed. Set `HF_TOKEN`
only when the configured model repository requires authentication. Configure
`SAINYX_API_KEY` for protected generation routes and never commit real tokens.
Use `SAINYX_VOICE_BACKEND=tone` to select the deterministic voice fallback in
tests.

## Test and security checks

Run the same CPU-capable test suite used by CI:

```powershell
python -m pytest -q tests
```

CI also invokes the security workflow, which audits Python dependencies,
scans repository history for secrets, and runs a Trivy filesystem scan. The
manual API smoke workflow loads the real text model and requires any needed
Hugging Face access; it is distinct from the unit tests and is not required
for every CPU-only test run.

## Contribution checklist

Add focused tests for behavior changes, update the API/model/deployment docs
when public behavior changes, and do not commit credentials, checkpoints,
raw datasets, generated personal data, or model artifacts. Keep quality and
availability claims grounded in tested checkpoints and current deployment
evidence.
