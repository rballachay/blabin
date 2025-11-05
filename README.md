# blabin
Adaptive agent for helping me learn French faster

## Prerequisites
- Use the dev container for this workspace (gcloud CLI is preinstalled).
- A Google Cloud project with billing enabled
- BigQuery API enabled in the project

## Authenticate to Google Cloud
```sh
gcloud init             # set up CLI and choose your project
gcloud auth application-default login
gcloud config set project <YOUR_GCP_PROJECT_ID>
```

## Create a Gemini API key
```sh
"$BROWSER" https://aistudio.google.com/app/apikey
```
Copy the key; you will add it to `.env` below.

## Infrastructure (Terraform)
Set up cloud resources using the READMEs in the terraform folders:
- Environment resources (BigQuery, etc.): see terraform/README.md
- MLflow tracking server (Cloud Run + Cloud SQL + GCS): see terraform/mlflow/README.md

Those guides cover creating tfvars from templates, enabling services, building/pushing the MLflow Docker image, and applying Terraform.

## Create your .env file
Create `.env` at the repo root with:
```sh
# .env
# owner of GCP account
OWNER_EMAIL=<YOUR_EMAIL>

# api key from gemini
GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>

# Environment Configuration
ENVIRONMENT=dev

# GCP Configuration
BIGQUERY_DATASET=dev_blabin
BIGQUERY_LOCATION=US
GOOGLE_CLOUD_PROJECT=<GOOGLE_CLOUD_PROJECT>
GOOGLE_CLOUD_QUOTA_PROJECT=<GOOGLE_CLOUD_PROJECT>

# settings for mlflow
MLFLOW_URI_LOCAL=http://127.0.0.1:8081
MLFLOW_EXPERIMENT=blabin-development
```

## Using MLflow
- Provision the remote MLflow server by following terraform/mlflow/README.md.
- To browse the UI locally without manually handling tokens, use the proxy:
```sh
cd terraform/mlflow
chmod +x open.sh
./open.sh
"$BROWSER" http://localhost:8081
```

## Run the application (chat mode)
```sh
python -m src.main --chat
```

## Notes
- If you see BigQuery permission errors, ensure ADC is set and the selected project matches your `.env` (GOOGLE_CLOUD_PROJECT).
