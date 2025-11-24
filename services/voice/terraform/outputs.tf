output "service_url" {
  description = "URL of the deployed FastAPI application"
  value       = google_cloud_run_service.fastapi_service.status[0].url
}

output "service_name" {
  description = "Name of the Cloud Run service"
  value       = google_cloud_run_service.fastapi_service.name
}

output "project_id" {
  description = "Google Cloud project ID"
  value       = var.project_id
}

output "region" {
  description = "Google Cloud region"
  value       = var.region
}

output "service_account_email" {
  description = "Email of the service account"
  value       = google_service_account.fastapi_service_account.email
}
