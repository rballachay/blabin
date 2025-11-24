# Terraform (env resources)

This folder contains the environment-scoped Terraform (e.g., BigQuery datasets and related IAM).

## Prerequisites
- Google Cloud project with billing enabled
- You have permissions to enable services and create resources
- gcloud CLI authenticated
- Terraform installed (available in the dev container)

## 1) Prepare .env (optional, recommended)
At repo root, set at least:
```dotenv
# .env
OWNER_EMAIL=<you@yourdomain.com>
GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
GCP_REGION=us-central1
BIGQUERY_LOCATION=US
```

## 2) Create dev.tfvars from the template
```sh
cd platform
cp dev.template.tfvars dev.tfvars
# Edit dev.tfvars:
#   project_id           = "<your-gcp-project-id>"
#   region               = "us-central1"
#   location             = "US"
#   dataset_owner_email  = "<you@yourdomain.com>"
#   environment          = "dev"
#   service_account_email (optional)
```

Tip: keep real tfvars out of git; only commit *.template.tfvars.

## 3) Authenticate gcloud
```sh
gcloud auth login
gcloud auth application-default login
gcloud config set project "$GOOGLE_CLOUD_PROJECT"
```

## 4) Apply the Terraform
```sh
cd platform
terraform init
terraform apply -var-file=dev.tfvars
# Review and type 'yes'
```

## 5) Outputs and app config
After apply:
```sh
terraform output
```
Update your repo .env as needed (e.g., BIGQUERY_DATASET, locations).

## 6) Clean up
```sh
terraform destroy -var-file=dev.tfvars
```
