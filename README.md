# blabin
Adaptive agent for helping me learn french faster

## Prerequisites
- A Google Cloud project with billing enabled
- BigQuery API enabled in the project
- This repo opened in the dev container (gcloud CLI is preinstalled)

## 1) Authenticate to Google Cloud
```sh
gcloud init             # set up CLI and choose your project
gcloud auth application-default login
gcloud config set project <YOUR_GCP_PROJECT_ID>
```

## 2) Create a Gemini API key
```sh
"$BROWSER" https://aistudio.google.com/app/apikey
```
Copy the key; you will add it to `.env` below.

## 3) Provision infrastructure with Terraform
```sh
cd terraform
cp dev.template.tfvars dev.tfvars
# edit dev.tfvars values (at minimum):
#   project_id           = "<YOUR_GCP_PROJECT_ID>"
#   region               = "us-central1"         # or your preference
#   location             = "US"                  # BigQuery location
#   dataset_owner_email  = "<you@yourdomain.com>"
#   environment          = "dev"
terraform init
terraform apply -var-file=dev.tfvars
# Review the plan and type 'yes' to apply
terraform output summary
```

## 4) Create your .env file
Create `.env` at the repo root with:
```sh
# .env
GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>

# Environment Configuration
ENVIRONMENT=dev

# GCP Configuration
GCP_PROJECT_ID=<YOUR_GCP_PROJECT_ID>
BIGQUERY_DATASET=dev_blabin
BIGQUERY_LOCATION=US
```

Tip: you can populate the GCP-related values from Terraform outputs:
```sh
cd terraform
terraform output -json python_config | jq -r 'to_entries[] | "\(.key)=\(.value)"'
```
Copy those lines into `.env` (alongside your GEMINI_API_KEY).

## 5) Run the application (chat mode)
```sh
python -m src.main --chat
```

## Notes
- If you see BigQuery permission errors, ensure ADC is set (`gcloud auth application-default login`) and the selected project matches your `.env` (`GCP_PROJECT_ID`).
