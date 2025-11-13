import json
import logging
import socket
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import mlflow
import requests


def _is_http_url(uri: str) -> bool:
    parsed = urlparse(uri)
    return parsed.scheme in ('http', 'https')


def _probe_http_server(tracking_uri: str, timeout: float = 3.0) -> bool:
    """
    Probe an MLflow server:
    1) GET root UI
    2) POST experiments/list (MLflow REST)
    Consider 2xx/3xx as up.
    """
    base = tracking_uri.rstrip('/') or tracking_uri

    # Attempt 1: GET root
    try:
        if requests:
            r = requests.get(base + '/', timeout=timeout, allow_redirects=True)
            if 200 <= r.status_code < 400:
                return True
        else:
            req = Request(base + '/', method='GET')
            with urlopen(req, timeout=timeout) as resp:
                if 200 <= resp.status < 400:
                    return True
    except Exception:
        pass

    # Attempt 2: POST experiments/list
    try:
        endpoint = base + '/api/2.0/mlflow/experiments/list'
        if requests:
            r = requests.post(endpoint, json={}, timeout=timeout)
            if 200 <= r.status_code < 400:
                return True
        else:
            data = json.dumps({}).encode('utf-8')
            req = Request(
                endpoint, data=data, method='POST', headers={'Content-Type': 'application/json'}
            )
            with urlopen(req, timeout=timeout) as resp:
                if 200 <= resp.status < 400:
                    return True
    except Exception:
        pass

    return False


def _is_mlflow_server_up(tracking_uri: str, timeout: float = 3.0) -> bool:
    """
    Return True if the MLflow tracking server at tracking_uri responds.
    For non-HTTP(S) URIs (e.g., local file store paths or file://), returns True.
    """
    if not _is_http_url(tracking_uri):
        return True  # local path or non-HTTP backend; skip probe

    # Avoid long DNS hangs
    socket.setdefaulttimeout(timeout)
    return _probe_http_server(tracking_uri, timeout=timeout)


def init_mlflow_autolog(
    tracking_uri: str,
    experiment_name: str,
    probe_timeout: float = 10.0,
) -> str | None:
    """
    Initialize MLflow with autologging after verifying the server is reachable for HTTP(S) URIs.
    For file paths and non-HTTP URIs, skip the probe and proceed.
    Raises RuntimeError if not configured or unreachable (HTTP(S) only).
    Returns the experiment name if initialized, else None.
    """
    if not tracking_uri:
        raise RuntimeError(
            'Tracking URI not set; set MLFLOW_URI_LOCAL (see terraform/mlflow/README.md).'
        )

    if not _is_mlflow_server_up(tracking_uri, timeout=probe_timeout):
        raise RuntimeError(f"MLflow server not reachable at '{tracking_uri}' (probe failed).")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Guard if langchain integration not present
    try:
        mlflow.langchain.autolog(silent=True)
    except AttributeError:
        logging.warning('mlflow.langchain.autolog not available; skipping.')

    logging.info(f"MLflow initialized at {tracking_uri} (experiment '{experiment_name}').")
    return experiment_name
