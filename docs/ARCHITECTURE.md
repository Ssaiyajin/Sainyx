# Sainyx architecture and release flow

The diagrams distinguish the earlier web/API and deployment flow from the
current typed API and staging-to-production workflow. They describe the
repository configuration and do not guarantee that a Space or checkpoint is
currently available.

## Earlier application flow

The earlier interface and API were served by one Flask application. The
browser used the chat interface, while API clients called the generic
generation endpoints. Generation code loaded configured model checkpoints
from the Hugging Face Hub.

```mermaid
flowchart LR
    Browser["Browser"] --> UI["Flask-rendered chat UI"]
    Client["API client"] --> GenericAPI["Flask REST API<br/>status + generic generation"]
    UI --> Flask["Flask application"]
    GenericAPI --> Flask
    Flask --> Generation["Text, image, video<br/>and data generation"]
    Generation --> Models["Configured model checkpoints"]
    Models --> Hub["Hugging Face Hub"]
    Flask --> Space["Hugging Face Space runtime"]
```

## Current application architecture

The Flask application serves the browser interface and versioned REST API.
Typed chat, image, and video routes and the generic generation route share
the application's registered generation code and `ModelFactory` checkpoint
cache. Checkpoints may be stored in different Hugging Face model repositories;
a configured repository ID does not prove that a checkpoint is published or
accessible.

```mermaid
flowchart LR
    Browser["Browser UI"] --> Flask["Flask application<br/>Hugging Face Space"]
    Client["API client"] --> API["Versioned REST API<br/>/api/v1/chat<br/>/api/v1/image<br/>/api/v1/video"]
    API --> Flask
    Flask --> Text["Text generation"]
    Flask --> Image["Image generation"]
    Flask --> Video["Video generation"]
    Flask --> Voice["Voice demo path"]
    Text --> Factory["ModelFactory<br/>shared checkpoint cache"]
    Image --> Factory
    Video --> Factory
    Factory --> Hub["Hugging Face model repositories"]
    Training["External or local training"] -. "checkpoint upload" .-> Hub
```

Training scripts and dataset builders are present in the repository, but
training provenance, dataset rights, and published artifact contents must be
verified separately.

## Earlier release flow

Before the current promotion flow, a push to `Dev` ran the API smoke check
before deploying to the staging Space. A push to `main` likewise ran the
smoke check before deploying directly to the production Space. That flow
did not establish staging verification followed by a protected production
approval.

```mermaid
flowchart LR
    PushDev["Push to Dev"] --> DevSmoke["API smoke check"]
    DevSmoke --> Stage["Deploy to staging Space"]

    PushMain["Push to main"] --> MainSmoke["API smoke check"]
    MainSmoke --> DirectProd["Direct deploy to production Space"]
```

This is a historical workflow shape, not an assertion about past Space
availability.

## Current release flow

The `Dev` workflow runs CPU tests and security scans, deploys to the
staging Space, and checks its public API. The `main` workflow runs the checks,
deploys the commit to staging, and waits for the staging API smoke test before
requesting approval through the protected GitHub `production` Environment.
After approval it promotes the commit to production and runs a production API
smoke test. Required reviewers must be configured on that Environment for
the approval gate to be effective. Each workflow defines its checks as direct
jobs so the Actions graph shows concise job names; shared steps live in
composite actions. Pull requests run the separate validation workflow.

```mermaid
flowchart LR
    PushDev["Push to Dev"] --> ChecksDev["CPU tests + security scans"]
    ChecksDev --> StageDev["Deploy to staging Space"]
    StageDev --> SmokeDev["Staging API smoke test"]

    PushMain["Push to main"] --> ChecksMain["CPU tests + security scans"]
    ChecksMain --> StageMain["Deploy commit to staging Space"]
    StageMain --> SmokeMain["Staging API smoke test"]
    SmokeMain --> Gate["Protected production Environment"]
    Gate --> Prod["Promote approved commit"]
    Prod --> SmokeProd["Production API smoke test"]
```

The workflows are [`staging.yml`](../.github/workflows/staging.yml),
[`deploy.yml`](../.github/workflows/deploy.yml), and the pull-request
validation workflow [`ci.yml`](../.github/workflows/ci.yml). Shared job steps
live in [composite actions](../.github/actions).

Sainyx does not currently configure Prometheus/Grafana monitoring or
Kubernetes manifests and deployment validation.
