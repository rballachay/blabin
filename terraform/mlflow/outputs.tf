output "mlflow_url" {
  description = "Cloud Run URL for the MLflow server"
  value       = google_cloud_run_service.mlflow.status[0].url
}

output "artifact_bucket" {
  description = "GCS bucket used as the MLflow default artifact root"
  value       = google_storage_bucket.artifacts.name
}

output "sql_instance_connection_name" {
  description = "Cloud SQL instance connection name for MLflow backend"
  value       = google_sql_database_instance.pg.connection_name
}

output "database" {
  description = "MLflow database connection details (password is sensitive)"
  value = {
    instance = google_sql_database_instance.pg.name
    name     = google_sql_database.mlflowdb.name
    user     = google_sql_user.mlflow.name
    password = random_password.db_password.result
  }
  sensitive = true
}

output "service_account_email" {
  description = "Service account used by the Cloud Run service"
  value       = google_service_account.mlflow.email
}

output "image" {
  description = "Container image deployed to Cloud Run"
  value       = var.mlflow_image
}

output "region" {
  description = "Region where MLflow (Cloud Run) is deployed"
  value       = var.region
}
