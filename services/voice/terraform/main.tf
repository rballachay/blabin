terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_artifact_registry_repository" "fastapi_registry" {
  provider      = google
  location      = var.region
  repository_id = "fastapi-containers"
  description   = "Docker images for FastAPI app"
  format        = "DOCKER"
  labels = {
    environment = var.environment
    app         = "fastapi"
  }
}

resource "google_artifact_registry_repository_iam_member" "fastapi_registry_writer" {
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.fastapi_registry.repository_id
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${var.service_account_email}"
}

resource "google_artifact_registry_repository_iam_member" "fastapi_registry_reader" {
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.fastapi_registry.repository_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${var.service_account_email}"
}

resource "google_cloud_run_service" "fastapi_service" {
  name     = "fastapi-service"
  location = var.region

  template {
    spec {
      container_concurrency = 1

      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/fastapi-containers/blabin-voice:latest"
        ports {
          container_port = 8000
        }

        resources {
          limits = {
            memory = "2Gi"
            cpu    = "2"
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service_iam_member" "fastapi_invoker" {
  service = google_cloud_run_service.fastapi_service.name
  location = google_cloud_run_service.fastapi_service.location
  role    = "roles/run.invoker"
  member  = "allUsers"
}

resource "google_service_account" "fastapi_service_account" {
  account_id   = "fastapi-service-account"
  display_name = "FastAPI Service Account"
  description  = "Service account for FastAPI application to access Google Cloud resources."
}

resource "google_project_iam_member" "artifact_registry_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.fastapi_service_account.email}"
}

resource "google_project_iam_member" "artifact_registry_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.fastapi_service_account.email}"
}

resource "google_project_iam_member" "cloud_run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.fastapi_service_account.email}"
}
