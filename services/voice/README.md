# Blabin Voice Service

FastAPI microservice for:
- VAD (Silero VAD via torch.hub)
- Voice embeddings (resemblyzer)

## Endpoints
- GET /health
- POST /vad
  - Body: { "samples": [float], "sample_rate": 16000 }
  - Returns: { "prob": float }
- POST /embed
  - Body: { "samples": [float], "sample_rate": 16000 }
  - Returns: { "embedding": [float] }

## Docker
Build and run:
```bash
docker build -t blabin-voice:latest services/voice
docker run -d --name blabin-voice -p 8000:8000 blabin-voice:latest
$BROWSER http://localhost:8000/docs
```

## Build/Tag/Push script
Requires a GCP service account JSON and Artifact Registry repo.

```bash
./build.sh
```

Env vars:
- GOOGLE_CLOUD_PROJECT (required)
- REGION (default: us-central1)
- REGISTRY_REPO (default: fastapi-containers)
- CREDS_FILE (default: ../../.creds/gcp-sa-key.json)

## Terraform (minimal)
Provisions:
- Artifact Registry repo
- IAM bindings (writer/reader)
- Cloud Run service + public invoker

Use:
```bash
cd services/voice/terraform
terraform init
terraform apply -var-file=terraform.tfvars.example
```

## Integrating with main app
Point to Cloud Run:
- VOICE_SERVICE_ENDPOINT=https://<cloud-run-url>
Local fallback is used when endpoint is unset/unhealthy.
