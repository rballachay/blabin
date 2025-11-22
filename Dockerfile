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

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential tzdata portaudio19-dev python3-pyaudio unzip pulseaudio-utils gnupg \
 && rm -rf /var/lib/apt/lists/*


# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /app

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
    && apt install -y tzdata \
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

# Install supercronic for cronjobs
RUN curl -L -o /usr/local/bin/supercronic \
https://github.com/aptible/supercronic/releases/latest/download/supercronic-linux-amd64

# Switch to non-root user
USER $USERNAME

# Install uv for the user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add user's .local/bin to path for uv
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

# copy the source in the container - don't mount, as it causes problems with macOS permissions
RUN sudo chown -R $USERNAME:$USERNAME /app
COPY --chown=$USERNAME:$USERNAME pyproject.toml /app/pyproject.toml
COPY --chown=$USERNAME:$USERNAME ./uv.lock /app/uv.lock

# sync dependencies
RUN uv sync

# copy the rest of the files
COPY --chown=$USERNAME:$USERNAME ./scripts /app/scripts
COPY --chown=$USERNAME:$USERNAME ./src /app/src
COPY --chown=$USERNAME:$USERNAME ./prompts /app/prompts

# connect docker to audio
ENV PULSE_SERVER=docker.for.mac.localhost:4713

RUN sudo chmod +x /app/scripts/blabin-entrypoint.sh
RUN sudo chmod +x /usr/local/bin/supercronic
