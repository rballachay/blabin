# Required
project_id  = "<YOUR_GCP_PROJECT_ID>"
region      = "us-central1"
sql_region  = "us-central1"
bucket_location = "US"

# Optional
artifact_bucket_name   = "<ARTIFACT_BUCKET>"
mlflow_image           = "us-central-docker.pkg.dev/<YOUR_GCP_PROJECT_ID>/mlflow/mlflow:v3.4.0"
db_tier                = "db-f1-micro"
db_version             = "POSTGRES_15"
allow_unauthenticated  = false
invoker_members        = ["user:you@yourdomain.com"]  # used when not public
deletion_protection    = true
