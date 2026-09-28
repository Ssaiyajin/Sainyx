# Sainyx Generation API

The versioned REST API is mounted at `/api/v1`. Call it like other hosted AI APIs: send an HTTPS request with your API key. No Hugging Face account or SDK is needed for API clients.

## Quick Start

Ask the Sainyx API owner for the base URL and API key. Send the key in the `X-API-Key` header:

```bash
curl -X POST https://YOUR_HOST/api/v1/generate \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"type":"text","prompt":"Who is Goku?","max_tokens":80}'
```

Python example:

```python
import os
import requests

response = requests.post(
    "https://YOUR_HOST/api/v1/generate",
    headers={"X-API-Key": os.environ["SAINYX_API_KEY"]},
    json={"type": "text", "prompt": "Who is Goku?", "max_tokens": 80},
    timeout=120,
)
response.raise_for_status()
print(response.json()["result"]["text"])
```

## API Key Setup (Owner Only)

The Sainyx Space owner creates a random key with `python -c "import secrets; print(secrets.token_urlsafe(32))"` and saves it as `SAINYX_API_KEY` in the Hugging Face Space settings. The owner shares that API key with authorized clients; clients do not create Space secrets and do not need the Hugging Face account token. For the post-deployment check, the owner also saves the same key as a GitHub Actions secret named `SAINYX_API_KEY`.

Clients should store the key in a server-side environment variable, not commit it or put it in public browser code. The current API uses one shared key for all clients.

`GET /status` and `GET /openapi.yaml` are public. Generation and job endpoints require `X-API-Key`.

The live OpenAPI 3.1 contract is available at `/api/v1/openapi.yaml`.

Generation requests are limited to 60 per minute per client IP and 64 KiB per JSON body. A `429` response includes `Retry-After`. The current rate counter and async job queue are in process memory; run a single application worker or enforce shared rate limits and durable jobs at the deployment gateway before scaling to multiple workers.

## Check Capabilities

```bash
curl https://YOUR_HOST/api/v1/status
```

The response reports availability, supported options, and limits. It currently advertises that image prompts are not conditioned into images and voice output is not speech synthesis.

## Generate

All successful JSON responses have the form `{"type":"...","status":"completed","result":{...}}`. Errors use `{"status":"error","error":{"code":"...","message":"..."}}`. Unknown fields and invalid options are rejected instead of silently ignored.

Text generation accepts `prompt` (1-4000 characters), `max_tokens` (1-500, default 80), and an optional non-negative `seed`:

```bash
curl -X POST https://YOUR_HOST/api/v1/generate \
  -H "X-API-Key: $SAINYX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"type":"text","prompt":"Who is Goku?","max_tokens":80,"seed":1234}'
```

Image generation accepts `prompt`, `num_images` (1-4), and `seed`. The current diffusion checkpoint is unconditional, so `prompt` is returned as metadata and does not influence pixels. JSON output contains `result.image_base64` for one image or `result.images_base64` for a batch.

```bash
curl -X POST https://YOUR_HOST/api/v1/generate \
  -H "X-API-Key: $SAINYX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"type":"image","prompt":"blue dragon","num_images":2,"seed":42}'
```

For a direct PNG download, set `response_format` to `binary`. Binary image output requires `num_images: 1`:

```bash
curl -X POST https://YOUR_HOST/api/v1/generate \
  -H "X-API-Key: $SAINYX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"type":"image","prompt":"blue dragon","seed":42,"response_format":"binary"}' \
  --output sainyx-image.png
```

Video currently returns one generated PNG frame, not an animation. It supports an optional `seed` and JSON or binary response format. The endpoint returns `503` while its checkpoint is unavailable.

Voice accepts `text` (or the legacy `prompt` alias) and `voice` (`neutral` or `energetic`). Its current output is a small WAV tone demo with the text included as metadata; it does not yet speak the supplied text.

## Async Jobs

Pass `"async": true` to queue any generation request. The response is `202` with a `job_id`, `status_url`, and `result_url`. Poll the status URL using the same API key, then fetch the result once `status` is `completed`:

```bash
curl -X POST https://YOUR_HOST/api/v1/generate \
  -H "X-API-Key: $SAINYX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"type":"image","prompt":"blue dragon","seed":42,"async":true}'

curl https://YOUR_HOST/api/v1/jobs/JOB_ID \
  -H "X-API-Key: $SAINYX_API_KEY"

curl https://YOUR_HOST/api/v1/jobs/JOB_ID/result \
  -H "X-API-Key: $SAINYX_API_KEY"
```

Job states are `queued`, `running`, `completed`, or `failed`. The in-memory queue runs one model job at a time, accepts at most 100 retained jobs, and expires finished results after one hour. Restarting the server clears jobs. An early result fetch returns `409`; missing or expired jobs return `404`.

## Error Codes

Common codes include `api_not_configured`, `unauthorized`, `invalid_json`, `unknown_type`, `unknown_fields`, `invalid_seed`, `invalid_num_images`, `generator_unavailable`, `rate_limit_exceeded`, `job_queue_full`, and `job_not_complete`. Check the HTTP status as well as the `error.code` value.