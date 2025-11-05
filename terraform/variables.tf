variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "gen-lang-client-0714613402"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "location" {
  description = "BigQuery dataset location"
  type        = string
  default     = "US"
}

variable "environment" {
  description = "Environment (dev, testing, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "testing", "prod"], var.environment)
    error_message = "Environment must be dev, testing, or prod."
  }
}

variable "dataset_owner_email" {
  description = "Email of the dataset owner"
  type        = string
}

variable "service_account_email" {
  description = "Service account email for accessing secrets"
  type        = string
}

variable "gemini_api_key" {
  description = "Gemini API key to store in Secret Manager"
  type        = string
  s
