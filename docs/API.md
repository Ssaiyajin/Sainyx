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

The Sainyx Space owner creates a random key with
`python -c "import secrets; print(secrets.token_urlsafe(32))"` and saves it as
`SAINYX_API_KEY` in the Hugging Face Space settings. The owner shares that API
key with authorized clients; clients do not create Space secrets and do not
need the Hugging Face account token. Deployment readiness checks use the
Hugging Face token when needed and do not call generation endpoints, so the
app API key does not need to be copied into GitHub Actions secrets.

Clients should store the key in a server-side environment variable, not commit it or put it in public browser code. The current API uses one shared key for all clients.

`GET /status` and `GET /openapi.yaml` are public. Generation and job endpoints require `X-API-Key`.

The live OpenAPI 3.1 contract is available at `/api/v1/openapi.yaml`.

Generation requests are limited to 60 per minute per client IP and 64 KiB per JSON body. A `429` response includes `Retry-After`. The current rate counter and async job queue are in process memory; run a single application worker or enforce shared rate limits and durable jobs at the deployment gateway before scaling to multiple workers.

## Typed endpoints

The generic `/generate` endpoint remains available. These aliases accept a focused request shape and share its authentication, validation, rate limit, async queue, and response envelope:

### Chat

`POST /chat` accepts `message` in place of `prompt`:

```bash
curl -X POST https://YOUR_HOST/api/v1/chat \
  -H "X-API-Key: $SAINYX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message":"Introduce Sainyx in one sentence.","max_tokens":40,"seed":42}'
```

### Image

`POST /image` accepts `prompt`, `num_images` (1-4), `seed`, `response_format`, and `async`. Prompt behavior depends on the loaded checkpoint. A tag-conditioned checkpoint resolves words against its tag vocabulary and can reject prompts that do not match; the generation code also supports older unconditional checkpoints.

```bash
curl -X POST https://YOUR_HOST/api/v1/image \
  -H "X-API-Key: $SAINYX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"gogeta blue","num_images":1,"seed":42}'
```

### Video

`POST /video` accepts optional `prompt` metadata, `seconds` (1-30), `resolution` (256-2160), `format` (`gif` or `mp4`), `seed`, `response_format`, and `async`. The model's native output is low-resolution; export upscaling does not add model detail. Prompt conditioning depends on the loaded model and may only be used to create an initial frame for a compatible image-to-video model.

```bash
curl -X POST https://YOUR_HOST/api/v1/video \
  -H "X-API-Key: $SAINYX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"seconds":2,"format":"gif","seed":42}'
```

Flask does not provide an interactive OpenAPI UI by default. The API contract is available at `/api/v1/openapi.yaml`; this guide provides the request examples and operational details.

## Check Capabilities

```bash
curl https://YOUR_HOST/api/v1/status
```

The response reports availability, supported options, and limits. Prompt conditioning and checkpoint availability vary by loaded model. Voice may use a speech model when available, otherwise it falls back to a synthetic tone.

## Generate

All successful JSON responses have the form `{"type":"...","status":"completed","result":{...}}`. Errors use `{"status":"error","error":{"code":"...","message":"..."}}`. Unknown fields and invalid options are rejected instead of silently ignored.

Text generation accepts `prompt` (1-4000 characters), `max_tokens` (1-500, default 80), and an optional non-negative `seed`:

```bash
curl -X POST https://YOUR_HOST/api/v1/generate \
  -H "X-API-Key: $SAINYX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"type":"text","prompt":"Who is Goku?","max_tokens":80,"seed":1234}'
```

Image generation accepts `prompt`, `num_images` (1-4), and `seed`. Tag-conditioned checkpoints use the prompt to select learned tags; older unconditional checkpoints may treat it only as metadata. JSON output contains `result.image_base64` for one image or `result.images_base64` for a batch.

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

Video returns a looping GIF (or MP4 with `"format": "mp4"`), base64-encoded as `video_base64` in JSON mode or as `image/gif` / `video/mp4` in binary mode. The model generates low-resolution frames that are enlarged for export. The API accepts `seconds` from 1 to 30, `resolution` from 256 to 2160, an optional `seed`, and JSON or binary response format. The endpoint returns `503` while its checkpoint is unavailable.

Voice accepts `text` (or the legacy `prompt` alias) and `voice` (`neutral` or `energetic`). Depending on the installed/configured backend, it may synthesize speech or return a WAV tone demo with the text as metadata; do not assume the fallback speaks the supplied text.

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