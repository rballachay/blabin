FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /workspaces/app

# Create a non-root user for development
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    &&  apt-get -y install portaudio19-dev \
    && apt install -y python3-pyaudio \
    && apt install -y unzip \
    && apt install -y pulseaudio-utils \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install Google Cloud SDK, neeeded for authentication with GCP
RUN apt-get install -y gnupg
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get update \
    && apt-get install -y google-cloud-cli \
    && apt-get install -y google-cloud-cli-cloud-run-proxy

# Make the version configurable
# https://github.com/hashicorp/terraform/releases
ARG TERRAFORM_VERSION=1.13.5

# Install Terraform as root user
RUN curl -fsSL https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip -o /tmp/terraform.zip \
  && unzip /tmp/terraform.zip -d /tmp \
  && mv /tmp/terraform /usr/local/bin/ \
  && rm -rf /tmp/* \
  && terraform version

# Switch to non-root user
USER $USERNAME

# Install auto-completions for non-root user
RUN terraform -install-autocomplete

# Install uv for the user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# connect docker to audio
ENV PULSE_SERVER=docker.for.mac.localhost:4713

# Add user's .local/bin to path for uv
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
