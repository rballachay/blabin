terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.5"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required services (one-time)
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
    "artifactregistry.googleapis.com",
  ])
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# Service Account for MLflow
resource "google_service_account" "mlflow" {
  account_id   = "mlflow-server"
  display_name = "MLflow Server"
}

# Artifact bucket for MLflow
locals {
  bucket_name = coalesce(var.artifact_bucket_name, "mlflow-artifacts-${var.project_id}")
}

resource "google_storage_bucket" "artifacts" {
  name                        = local.bucket_name
  location                    = var.bucket_location
  uniform_bucket_level_access = true
  force_destroy               = false

  lifecycle_rule {
    action { type = "Delete" }
    condition { age = 365 }
  }

  depends_on = [google_project_service.apis]
}

# Grant MLflow SA access to bucket objects
resource "google_storage_bucket_iam_member" "mlflow_bucket_write" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.mlflow.email}"
}

# Cloud SQL Postgres for MLflow backend store
resource "google_sql_database_instance" "pg" {
  name             = "mlflow-pg"
  database_version = var.db_version
  region           = var.sql_region
  deletion_protection = var.deletion_protection

  settings {
    tier = var.db_tier

    ip_configuration {
      # Keep public IP enabled for simplicity with Cloud Run connector.
      # No authorized networks needed when using the connector.
      ipv4_enabled = true
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_sql_database" "mlflowdb" {
  name     = "mlflowdb"
  instance = google_sql_database_instance.pg.name
}

resource "random_password" "db_password" {
  length  = 20
  special = true
}

resource "google_sql_user" "mlflow" {
  instance = google_sql_database_instance.pg.name
  name     = "mlflow"
  password = random_password.db_password.result
}

# Allow the SA to connect to Cloud SQL
resource "google_project_iam_member" "mlflow_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.mlflow.email}"
}

data "google_project" "proj" {}

# Cloud Run service agent (pulls the image)
resource "google_project_iam_member" "cr_service_agent_ar_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:service-${data.google_project.proj.number}@serverless-robot-prod.iam.gserviceaccount.com"
}


# Cloud Run service for MLflow
resource "google_cloud_run_service" "mlflow" {
  name     = "mlflow"
  location = var.region

  // Service-level annotations (OK for ingress)
  metadata {
    annotations = {
      "run.googleapis.com/ingress" = "internal-and-cloud-load-balancing" // or "internal-and-cloud-load-balancing"
    }
  }

  template {
    metadata {
      annotations = {
        // Revision-level annotations (keep these here)
        "run.googleapis.com/cloudsql-instances" = google_sql_database_instance.pg.connection_name
        "autoscaling.knative.dev/minScale"      = "0"
        "autoscaling.knative.dev/maxScale"      = "1"
      }
    }
    spec {
      service_account_name = google_service_account.mlflow.email
      containers {
        image = var.mlflow_image
        command = ["mlflow"]
        args = [
          "server",
          "--backend-store-uri",
          "postgresql+psycopg2://mlflow:${urlencode(random_password.db_password.result)}@/${google_sql_database.mlflowdb.name}?host=/cloudsql/${google_sql_database_instance.pg.connection_name}",
          "--default-artifact-root",
          "gs://${google_storage_bucket.artifacts.name}",
          "--host", "0.0.0.0",
          "--port", "5000"
        ]
        ports { container_port = 5000 }
        resources { limits = { cpu = "1", memory = "1Gi" } }
      }
      timeout_seconds = 300
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [
    google_project_service.apis,
    google_project_iam_member.mlflow_sql_client,
    google_storage_bucket_iam_member.mlflow_bucket_write,
  ]
}

# IAM: who can invoke
resource "google_cloud_run_service_iam_member" "public" {
  count    = var.allow_unauthenticated ? 1 : 0
  location = google_cloud_run_service.mlflow.location
  service  = google_cloud_run_service.mlflow.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_service_iam_member" "invokers" {
  for_each = var.allow_unauthenticated ? toset([]) : toset(var.invoker_members)
  location = google_cloud_run_service.mlflow.location
  service  = google_cloud_run_service.mlflow.name
  role     = "roles/run.invoker"
  member   = each.value
}
