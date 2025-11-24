FROM ghcr.io/mlflow/mlflow:v3.4.0

RUN apt-get -y update && \
    apt-get -y install python3-dev default-libmysqlclient-dev build-essential pkg-config && \
    pip install --upgrade pip && \
    pip install mysqlclient && \
    pip install psycopg2-binary && \
    pip install google-cloud-storage

CMD ["bash"]
