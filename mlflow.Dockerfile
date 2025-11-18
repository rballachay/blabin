FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:slim
WORKDIR /app

RUN apt-get update && apt-get install -y google-cloud-cli-cloud-run-proxy

COPY scripts/mlflow-entrypoint.sh /app/scripts/
COPY .creds/gcp-sa-key.json /app/gcp-sa-key.json

ENV GCP_SA_KEY_FILE="/app/gcp-sa-key.json"
ENV GCP_REGION="us-central1"

ENTRYPOINT ["/bin/bash","/app/scripts/mlflow-entrypoint.sh"]
