# Sainyx Generation API

The versioned REST API is mounted at `/api/v1`. The live OpenAPI 3.1 contract is available at `/api/v1/openapi.yaml`.

## Authentication

`GET /status` and `GET /openapi.yaml` are public. All generation and job endpoints require the server key in `X-API-Key`:

```powershell
$env:SAINYX_API_KEY = "replace-with-a-long-random-secret"
```

Create a key with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Configure the key as a deployment secret; never put it in browser code or commit it.

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