from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.entities import Run
from mlflow.tracking import MlflowClient


class PromptManager:
    """
    Lightweight prompt manager:
      - load prompts from local YAML files
      - publish prompts to MLflow (as artifacts + tags)
      - fetch latest prompt by name from MLflow
    """

    def __init__(
        self,
        experiment_name: str = 'prompts',
        tracking_uri: str | None = None,
    ):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.experiment = self._ensure_experiment_exists(experiment_name)
        self._local_dir = Path('prompts')
        self._local_dir.mkdir(exist_ok=True)

    def _ensure_experiment_exists(self, experiment_name: str):
        """Create MLflow experiment if it doesn't exist yet."""
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            exp_id = mlflow.create_experiment(experiment_name)
            experiment = mlflow.get_experiment(exp_id)
        else:
            exp_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        print(f"[PromptManager] Using MLflow experiment '{experiment_name}' (id={exp_id})")
        return experiment

    def load_local(self, path: Path) -> dict[str, Any]:
        return yaml.safe_load(path.read_text(encoding='utf-8'))

    def save_local(self, name: str, content: dict[str, Any], version: str | None = None) -> Path:
        fn = f'{name}{("_" + version) if version else ""}.yaml'
        p = self._local_dir / fn
        p.write_text(yaml.safe_dump(content), encoding='utf-8')
        return p

    def publish_to_mlflow(
        self,
        name: str,
        content: dict[str, Any],
        version: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        tags = tags or {}
        if version:
            tags['prompt_version'] = version
        tags['prompt_name'] = name

        with mlflow.start_run(experiment_id=self.experiment.experiment_id) as run:
            run_id = run.info.run_id
            mlflow.log_text(yaml.safe_dump(content), artifact_file=f'prompts/{name}.yaml')
            for k, v in tags.items():
                mlflow.set_tag(k, v)
            return run_id

    def get_latest_from_mlflow(self, name: str) -> dict[str, Any] | None:
        filter_str = f"tags.prompt_name = '{name}'"
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=filter_str,
            order_by=['attribute.start_time DESC'],
        )
        if isinstance(runs, list):
            if not runs:
                return None

            run_0 = runs[0]

            if isinstance(run_0, Run):
                run_id = run_0.run_id
        try:
            bpath = self.client.download_artifacts(run_id, f'prompts/{name}.yaml')
            return yaml.safe_load(Path(bpath).read_text(encoding='utf-8'))
        except Exception:
            return None

    def get_prompt(
        self, name: str, version: str | None = None, local_only: bool = False
    ) -> dict[str, Any] | None:
        if not local_only:
            ml = self.get_latest_from_mlflow(name)
            if ml:
                return ml

        if version:
            p = self._local_dir / f'{name}_{version}.yaml'
        else:
            p = self._local_dir / f'{name}.yaml'
            if not p.exists():
                matches = list(self._local_dir.glob(f'{name}*.yaml'))
                p = matches[0] if matches else p

        if p.exists():
            return self.load_local(p)
        return None
