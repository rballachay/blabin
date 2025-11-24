variable "project_id" {
  description = "The ID of the Google Cloud project"
  type        = string
}

variable "region" {
  description = "The region where the resources will be deployed"
  type        = string
}

variable "service_account_email" {
  description = "The email of the service account for accessing Google Cloud resources"
  type        = string
}

variable "environment" {
  description = "The environment for deployment (dev, testing, prod)"
  type        = string
}
