# MLflow (Cloud Run + Cloud SQL + GCS)

This MLflow server is built from the Dockerfile in this folder, pushed to Google Artifact Registry, and deployed to Cloud Run using Terraform. It uses:
- Cloud SQL (Postgres) as the MLflow backend store
- GCS as the default artifact root

The stack is designed to be created once (global), separate from environment-scoped resources.

## Prerequisites
- A GCP project with billing enabled
- You are an IAM project editor/admin (able to enable services and run Cloud Build)
- gcloud CLI authenticated
- Terraform installed (provided in the dev container)

## 1) Create or update your .env (repo root)
Ensure the following are set (add others as needed):
```dotenv
# .env (repo root)
GCP_PROJECT_ID=<YOUR_GCP_PROJECT_ID>
GCP_REGION=us-central1
OWNER_EMAIL=you@yourdomain.com
```

## 2) Create mlflow.tfvars from the template
```sh
cd terraform/mlflow
cp mlflow.template.tfvars mlflow.tfvars
# Edit mlflow.tfvars:
#   project_id, region, sql_region, bucket_location
#   allow_unauthenticated, invoker_members (usually your OWNER_EMAIL)
# Temporarily set mlflow_image to:
#   us-central1-docker.pkg.dev/<PROJECT>/mlflow/mlflow:v3.4.0
```

## 3) Authenticate with gcloud
```sh
gcloud auth login
gcloud auth application-default login
gcloud config set project "$GCP_PROJECT_ID"
```

## 4) Build and push the MLflow image (Artifact Registry)
The provided script reads .env and creates the Artifact Registry repo (if needed), grants permissions, then builds and pushes the image.

```sh
cd terraform/mlflow
chmod +x build.sh
./build.sh
```

This pushes:
```
us-central1-docker.pkg.dev/$GCP_PROJECT_ID/mlflow/mlflow:v3.4.0
```

Make sure mlflow_image in mlflow.tfvars matches that value.

## 5) Deploy with Terraform
```sh
cd terraform/mlflow
terraform init
terraform apply -var-file=mlflow.tfvars
```

Outputs include the MLflow URL:
```sh
terraform output -raw mlflow_url
```

## 6) Access the server
Once build, you can use the following to access the mlflow server.
- If allow_unauthenticated = false (recommended), use IAM:
```sh
URL=$(terraform output -raw mlflow_url)
export MLFLOW_TRACKING_URI="$URL"
export MLFLOW_TRACKING_TOKEN=$(gcloud auth print-identity-token --audiences="$URL")
```

Browse via a local proxy (no manual token needed):
```sh
cd terraform/mlflow
chmod +x open.sh
./open.sh
# then open:
"$BROWSER" http://localhost:8081
```

## Notes
- The Dockerfile installs Postgres and MySQL clients plus google-cloud-storage so MLflow can write artifacts to GCS and use Cloud SQL.
- The Artifact Registry repo is created by build.sh (not Terraform) to keep the flow simple.
- The service account used by Cloud Run is granted the minimum rights to connect to Cloud SQL and write to the artifact bucket.
- For IP allowlisting, put Cloud Run behind an HTTPS Load Balancer with Cloud Armor (optional).
