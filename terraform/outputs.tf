output "dataset_id" {
  description = "BigQuery dataset ID (includes environment prefix)"
  value       = google_bigquery_dataset.blabin.dataset_id
}

output "dataset_full_id" {
  description = "Full BigQuery dataset ID (project:dataset)"
  value       = "${var.project_id}:${google_bigquery_dataset.blabin.dataset_id}"
}

output "dataset_location" {
  description = "BigQuery dataset location"
  value       = google_bigquery_dataset.blabin.location
}

output "environment" {
  description = "Current environment"
  value       = var.environment
}

output "tables_created" {
  description = "List of created table IDs"
  value = [
    google_bigquery_table.articles.table_id,
    google_bigquery_table.session_summaries.table_id,
    google_bigquery_table.sessions.table_id,
    google_bigquery_table.speakers.table_id,
  ]
}

output "table_full_names" {
  description = "Fully qualified table names (project.dataset.table)"
  value = {
    articles          = "${var.project_id}.${google_bigquery_dataset.blabin.dataset_id}.${google_bigquery_table.articles.table_id}"
    session_summaries = "${var.project_id}.${google_bigquery_dataset.blabin.dataset_id}.${google_bigquery_table.session_summaries.table_id}"
    sessions          = "${var.project_id}.${google_bigquery_dataset.blabin.dataset_id}.${google_bigquery_table.sessions.table_id}"
    speakers          = "${var.project_id}.${google_bigquery_dataset.blabin.dataset_id}.${google_bigquery_table.speakers.table_id}"
  }
}

output "python_config" {
  description = "Configuration values for Python .env file"
  value = {
    ENVIRONMENT       = var.environment
    GOOGLE_CLOUD_PROJECT    = var.project_id
    BIGQUERY_DATASET  = google_bigquery_dataset.blabin.dataset_id
    BIGQUERY_LOCATION = google_bigquery_dataset.blabin.location
  }
}

output "summary" {
  description = "Deployment summary"
  value = <<-EOT
    ========================================
    Blabin Infrastructure Deployed
    ========================================
    Environment:    ${var.environment}
    Dataset:        ${google_bigquery_dataset.blabin.dataset_id}
    Location:       ${google_bigquery_dataset.blabin.location}
    Tables Created: ${length([
      google_bigquery_table.articles.table_id,
      google_bigquery_table.session_summaries.table_id,
      google_bigquery_table.sessions.table_id,
      google_bigquery_table.speakers.table_id,
    ])}

    Next steps:
    1. Update your .env file with:
       ENVIRONMENT=${var.environment}
       BIGQUERY_DATASET=${google_bigquery_dataset.blabin.dataset_id}

    2. Run your application:
       python -m src.main --chat
    ========================================
  EOT
}
