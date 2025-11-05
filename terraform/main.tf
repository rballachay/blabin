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

# BigQuery Dataset - name includes environment
resource "google_bigquery_dataset" "blabin" {
  dataset_id                  = "${var.environment}_blabin"  # dev_blabin, testing_blabin, prod_blabin
  friendly_name               = "Blabin Language Learning Data (${upper(var.environment)})"
  description                 = "Dataset for conversation sessions, mistakes, speakers, and news articles - ${var.environment} environment"
  location                    = var.location
  default_table_expiration_ms = null

  labels = {
    environment = var.environment
    app         = "blabin"
  }

  access {
    role          = "OWNER"
    user_by_email = var.dataset_owner_email
  }
}

# News Articles Table
resource "google_bigquery_table" "articles" {
  dataset_id = google_bigquery_dataset.blabin.dataset_id
  table_id   = "articles"

  deletion_protection = var.environment == "prod" ? true : false  # Protect prod only

  schema = jsonencode([
    {
      name = "id"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "source"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "title"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "link"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "published"
      type = "TIMESTAMP"
      mode = "NULLABLE"
    },
    {
      name = "text"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "fetched_at"
      type = "TIMESTAMP"
      mode = "NULLABLE"
    }
  ])

  time_partitioning {
    type  = "DAY"
    field = "fetched_at"
  }

  clustering = ["source"]
}

# Session Summaries (Mistakes) Table
resource "google_bigquery_table" "session_summaries" {
  dataset_id = google_bigquery_dataset.blabin.dataset_id
  table_id   = "session_summaries"

  deletion_protection = var.environment == "prod" ? true : false

  schema = jsonencode([
    {
      name = "session_id"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "records_json"
      type = "JSON"
      mode = "NULLABLE"
    },
    {
      name = "counts_json"
      type = "JSON"
      mode = "NULLABLE"
    },
    {
      name = "total_mistakes"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "level_cefr"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "level_confidence"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "level_method"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "level_window"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "level_explanation"
      type = "STRING"
      mode = "NULLABLE"
    }
  ])

  time_partitioning {
    type  = "DAY"
    field = "created_at"
  }

  clustering = ["session_id"]
}

# Sessions Table
resource "google_bigquery_table" "sessions" {
  dataset_id = google_bigquery_dataset.blabin.dataset_id
  table_id   = "sessions"

  deletion_protection = var.environment == "prod" ? true : false

  schema = jsonencode([
    {
      name = "session_id"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "created_at"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "ended_at"
      type = "TIMESTAMP"
      mode = "NULLABLE"
    },
    {
      name = "duration_sec"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "input_mode"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "turns_total"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "turns_user"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "turns_assistant"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "user_chars"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "assistant_chars"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "user_words"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "assistant_words"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "user_tokens_approx"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "assistant_tokens_approx"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "resp_latency_avg_ms"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "resp_latency_p95_ms"
      type = "FLOAT64"
      mode = "NULLABLE"
    },
    {
      name = "errors"
      type = "INT64"
      mode = "NULLABLE"
    },
    {
      name = "notes"
      type = "STRING"
      mode = "NULLABLE"
    }
  ])

  time_partitioning {
    type  = "DAY"
    field = "created_at"
  }

  clustering = ["session_id"]
}

# Speakers Table
resource "google_bigquery_table" "speakers" {
  dataset_id = google_bigquery_dataset.blabin.dataset_id
  table_id   = "speakers"

  deletion_protection = var.environment == "prod" ? true : false

  schema = jsonencode([
    {
      name = "id"
      type = "INT64"
      mode = "REQUIRED"
    },
    {
      name = "name"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "first_seen"
      type = "TIMESTAMP"
      mode = "NULLABLE"
    },
    {
      name = "last_seen"
      type = "TIMESTAMP"
      mode = "NULLABLE"
    },
    {
      name = "voice_signature"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "language_level"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "sample_count"
      type = "INT64"
      mode = "NULLABLE"
    }
  ])

  clustering = ["id"]
}
