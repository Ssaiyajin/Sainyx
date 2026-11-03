# Sainyx Space deployment

## Configured targets

- Staging: `ssaiyajin/Sainyx-staging`
- Production: `ssaiyajin/Sainyx`

These are configured Hugging Face Space targets. The repository does not
verify that either Space is currently healthy or that every configured model
checkpoint is present.

## Release paths

On a push to `Dev`, `.github/workflows/staging.yml` runs CPU tests and
security checks, pushes the commit to the staging Space, then checks the
API discovery endpoint. This readiness check makes no generation request and
does not need the app API key. It uses `HF_TOKEN` to access the private Space.

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

Configure this secret in GitHub repository settings:

| Secret | Used for |
| --- | --- |
| `HF_TOKEN` | Push app source and authenticate readiness checks against the private staging Space; grant only required access |

Never put this token in source, notebooks, workflow output, or client-side
JavaScript. Treat API keys as shared credentials.

`SAINYX_API_KEY` remains a Space runtime setting for protected generation
requests; the deployment readiness checks do not need it.

## What this deployment does not provide

There is no configured Prometheus/Grafana monitoring stack or Kubernetes
deployment in this repository. CI tests application code and checks the
hosted API; it does not deploy to Kubernetes. Hugging Face Space deployment
does not imply that model generation is available: the configured checkpoint
must also be accessible and compatible with the code.
