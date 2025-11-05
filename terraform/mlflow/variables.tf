variable "project_id" {
  description = "GCP project ID where the one MLflow server will live"
  type        = string
}

variable "region" {
  description = "Region for Cloud Run"
  type        = string
  default     = "us-central1"
}

variable "sql_region" {
  description = "Region for Cloud SQL (can match region)"
  type        = string
  default     = "us-central1"
}

variable "bucket_location" {
  description = "Location for the artifact bucket"
  type        = string
  default     = "US"
}

variable "artifact_bucket_name" {
  description = "Optional custom bucket name; default derives from project"
  type        = string
  default     = null
}

variable "mlflow_image" {
  description = "Container image for MLflow server"
  type        = string
  default     = "ghcr.io/mlflow/mlflow:2.14.1"
}

variable "db_tier" {
  description = "Cloud SQL machine tier"
  type        = string
  default     = "db-f1-micro"
}

variable "db_version" {
  description = "Cloud SQL Postgres version"
  type        = string
  default     = "POSTGRES_15"
}

variable "allow_unauthenticated" {
  description = "Allow public access to MLflow (otherwise auth required)"
  type        = bool
  default     = false
}

variable "invoker_members" {
  description = "Additional principals allowed to invoke (when not public). Example: [\"user:you@domain.com\"]"
  type        = list(string)
  default     = []
}

variable "deletion_protection" {
  description = "Protect Cloud SQL from accidental deletion"
  type        = bool
  default     = true
}

variable "enable_iap" {
  description = "Enable HTTPS LB + IAP in front of Cloud Run"
  type        = bool
  default     = false
}

variable "domain_name" {
  description = "DNS name for the HTTPS LB (e.g., mlflow.example.com)"
  type        = string
  default     = null
}

variable "iap_oauth_client_id" {
  description = "OAuth 2.0 Client ID for IAP"
  type        = string
  default     = null
}

variable "iap_oauth_client_secret" {
  description = "OAuth 2.0 Client Secret for IAP"
  type        = string
  default     = null
  sensitive   = true
}

variable "iap_members" {
  description = "Principals allowed through IAP (e.g., [\"user:you@domain.com\"])"
  type        = list(string)
  default     = []
}
