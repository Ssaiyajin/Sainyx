# Sainyx Space deployment

## Configured targets

- Staging: `ssaiyajin/Sainyx-staging`
- Production: `ssaiyajin/Sainyx`

These are configured Hugging Face Space targets. The repository does not
verify that either Space is currently healthy or that every configured model
checkpoint is present.

## Release paths

On a push to `Dev`, `.github/workflows/staging.yml` runs reusable CPU tests
and security checks, pushes the commit to the staging Space, then calls the
public API smoke script.

On a push to `main`, `.github/workflows/deploy.yml` runs the same CI,
pushes the commit to staging and waits for the staging API smoke test, then
promotes that commit to production through the `production` GitHub
Environment. The workflow runs a production API smoke test after promotion.
Configure required reviewers in the Environment settings; merely referencing
the Environment does not create an effective manual approval gate.

The deployment steps currently force-push the checked-out commit to each
Space's `main` branch. Review this behavior and confirm the intended target
branch before changing the workflow.

## GitHub Actions secrets

Configure the following secrets in repository settings, using the same API
keys in the corresponding Space settings:

| Secret | Used for |
| --- | --- |
| `HF_TOKEN` | Push the app source to the configured Spaces; grant only required access |
| `SAINYX_STAGING_API_KEY` | Authenticate the post-deploy staging smoke test |
| `SAINYX_API_KEY` | Authenticate the post-deploy production smoke test |

Never put these values in source, notebooks, workflow output, or client-side
JavaScript. Treat the API keys as shared credentials.

## What this deployment does not provide

There is no configured Prometheus/Grafana monitoring stack or Kubernetes
deployment in this repository. CI tests application code and checks the
hosted API; it does not deploy to Kubernetes. Hugging Face Space deployment
does not imply that model generation is available: the configured checkpoint
must also be accessible and compatible with the code.
