import json
import os
import uuid
from pathlib import Path
from typing import Dict, Any, List

import docker
from docker.errors import DockerException

from config.logging_config import get_logger
from services.container_config import ContainerConfig

logger = get_logger("ContainerOrchestrator")


class ContainerOrchestrator:
    def __init__(self):
        logger.info("Initializing ContainerOrchestrator...")
        self.client = self._connect_client()
        self.image_name = ContainerConfig.IMAGE_NAME
        self.containers: Dict[str, Any] = {}
        self.host_backend_path = os.environ.get('HOST_BACKEND_PATH', '/app')
        self._log_init_status()
        self._cleanup_orphaned_containers()

    def __del__(self):
        self.cleanup_all()

    def _log_init_status(self):
        status = 'Connected' if self.client else 'Not available'
        logger.info(f"ContainerOrchestrator initialized | Docker client: {status}")
        logger.info(f"Host backend path: {self.host_backend_path}")

    def _connect_client(self):
        try:
            client = docker.from_env()
            logger.info("✅ Successfully connected to Docker daemon")
            return client
        except DockerException as e:
            logger.warning(f"⚠️ Could not connect to Docker daemon: {e}")
            return None

    def _require_client(self):
        if not self.client:
            raise RuntimeError('Docker daemon is not accessible from the backend container.')

    def _get_dockerfile_path(self) -> Path:
        return Path("/app/Dockerfile.sandbox")

    def _get_app_path(self) -> Path:
        return Path("/app")

    def build_sandbox_image(self):
        self._require_client()
        dockerfile_path = self._get_dockerfile_path()
        logger.info(f"🔨 Building sandbox image: {self.image_name}")
        logger.info(f"   Dockerfile path: {dockerfile_path}")

        if not dockerfile_path.exists():
            logger.error(f"❌ Dockerfile not found: {dockerfile_path}")
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

        self._build_image(dockerfile_path)

    def _build_image(self, dockerfile_path: Path):
        context_path = dockerfile_path.parent
        logger.info(f"   Build context: {context_path}")
        self.client.images.build(
            path=str(context_path),
            dockerfile=str(dockerfile_path.name),
            tag=self.image_name,
            rm=True
        )
        logger.info(f"✅ Successfully built sandbox image: {self.image_name}")

    def ensure_image_exists(self):
        self._require_client()
        logger.info(f"📦 Building/rebuilding sandbox image: {self.image_name}")
        self.build_sandbox_image()

    def _prepare_volumes(self) -> Dict[str, Dict[str, str]]:
        host_base = Path(self.host_backend_path)
        volumes = {
            str(host_base / "models"): {'bind': '/app/models', 'mode': 'ro'},
            str(host_base / "data"): {'bind': '/app/data', 'mode': 'ro'},
            str(host_base / "extractors"): {'bind': '/app/extractors', 'mode': 'ro'},
            str(host_base / "services"): {'bind': '/app/services', 'mode': 'ro'},
            str(host_base / "config"): {'bind': '/app/config', 'mode': 'ro'},
        }
        self._log_volumes(host_base)
        return volumes

    def _log_volumes(self, base: Path):
        logger.debug("   Volume mounts prepared (host paths):")
        logger.debug(f"      - {base / 'models'} -> /app/models (read-only)")
        logger.debug(f"      - {base / 'data'} -> /app/data (read-only)")
        logger.debug(f"      - {base / 'extractors'} -> /app/extractors (read-only)")
        logger.debug(f"      - {base / 'services'} -> /app/services (read-only)")
        logger.debug(f"      - {base / 'config'} -> /app/config (read-only)")

    def _prepare_environment(self, urls: List[str], models: List[str], mode: str, dataset: str) -> Dict[str, str]:
        env = {'URLS': json.dumps(urls), 'MODELS': json.dumps(models), 'MODE': mode, 'DATASET': dataset}
        self._log_environment(urls, models, mode, dataset)
        return env

    def _log_environment(self, urls: List[str], models: List[str], mode: str, dataset: str):
        logger.debug("   Environment variables:")
        logger.debug(f"      - URLS: {urls}")
        logger.debug(f"      - MODELS: {models}")
        logger.debug(f"      - MODE: {mode}")
        logger.debug(f"      - DATASET: {dataset}")

    def _log_container_creation(self, container_id: str, container_name: str, urls: List[str], models: List[str], mode: str, dataset: str):
        logger.info(f"{'='*60}")
        logger.info("🐳 CREATING SANDBOX CONTAINER")
        logger.info(f"   Container ID: {container_id}")
        logger.info(f"   Container Name: {container_name}")
        logger.info(f"   Image: {self.image_name}")
        logger.info(f"   URLs to analyze: {len(urls)}")
        for i, url in enumerate(urls, 1):
            logger.info(f"      {i}. {url}")
        logger.info(f"   Models: {models}")
        logger.info(f"   Mode: {mode}")
        logger.info(f"   Dataset: {dataset}")
        logger.info(f"{'='*60}")

    def _log_container_config(self):
        logger.info("   🔧 Container configuration:")
        logger.info(f"      - Memory limit: {ContainerConfig.MEMORY_LIMIT}")
        logger.info(f"      - CPU limit: {ContainerConfig.CPU_LIMIT} cores")
        logger.info(f"      - Network: isolated ({ContainerConfig.NETWORK_MODE})")

    def _run_container(self, container_name: str, environment: Dict, volumes: Dict):
        nano_cpus = int(ContainerConfig.CPU_LIMIT * 1e9)
        return self.client.containers.run(
            self.image_name,
            detach=True,
            environment=environment,
            volumes=volumes,
            mem_limit=ContainerConfig.MEMORY_LIMIT,
            nano_cpus=nano_cpus,
            network_mode=ContainerConfig.NETWORK_MODE,
            remove=False,
            name=container_name,
            auto_remove=False
        )

    def create_container(self, urls: List[str], models: List[str], mode: str = "optimized", dataset: str = 'current') -> str:
        self._require_client()
        self.ensure_image_exists()

        container_id = str(uuid.uuid4())
        container_name = f"phishing-sandbox-{container_id}"
        self._log_container_creation(container_id, container_name, urls, models, mode, dataset)

        environment = self._prepare_environment(urls, models, mode, dataset)
        volumes = self._prepare_volumes()
        self._log_container_config()

        container = self._run_container(container_name, environment, volumes)
        self.containers[container_id] = container

        logger.info("   ✅ Container created successfully!")
        logger.info(f"      Docker Container ID: {container.short_id}")
        logger.info(f"      Status: {container.status}")
        return container_id

    def _parse_container_logs(self, logs: str) -> Dict[str, Any]:
        try:
            result = json.loads(logs.split('\n')[-2])
            logger.debug("   📄 Container output parsed successfully")
            return result
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"   ❌ Failed to parse container output: {e}")
            logger.debug(f"   Raw logs: {logs[:500]}...")
            return {'success': False, 'error': 'Failed to parse container output', 'raw_logs': logs}

    def _log_result_status(self, parsed_result: Dict[str, Any]):
        if parsed_result.get('success'):
            results_count = len(parsed_result.get('results', []))
            logger.info(f"   ✅ Analysis completed: {results_count} URL(s) processed")
        else:
            logger.warning(f"   ⚠️ Container reported failure: {parsed_result.get('error', 'Unknown error')}")

    def wait_for_result(self, container_id: str, timeout: int = 300) -> Dict[str, Any]:
        self._require_client()
        if container_id not in self.containers:
            logger.error(f"❌ Container not found: {container_id}")
            raise ValueError(f"Container {container_id} not found")

        container = self.containers[container_id]
        logger.info(f"   ⏳ Waiting for container to complete (timeout: {timeout}s)...")

        try:
            return self._wait_and_parse(container, timeout)
        except Exception as e:
            logger.error(f"   ❌ Error waiting for container: {e}")
            return {'success': False, 'error': str(e), 'container_id': container_id}

    def _wait_and_parse(self, container, timeout: int) -> Dict[str, Any]:
        result = container.wait(timeout=timeout)
        exit_code = result.get('StatusCode', -1)
        logger.info(f"   📦 Container finished with exit code: {exit_code}")

        logs = container.logs().decode('utf-8')
        logger.debug(f"   📝 Retrieved {len(logs.split(chr(10)))} lines of container logs")

        parsed_result = self._parse_container_logs(logs)
        self._log_result_status(parsed_result)
        return parsed_result

    def cleanup_container(self, container_id: str):
        if not self.client or container_id not in self.containers:
            return

        container = self.containers[container_id]
        logger.info(f"   🧹 Cleaning up container: {container.name}")
        self._stop_container(container)
        self._remove_container(container)

        del self.containers[container_id]
        logger.info(f"   ✅ Container cleanup completed: {container_id}")

    def _stop_container(self, container):
        try:
            container.stop(timeout=10)
            logger.debug("      - Container stopped")
        except Exception as e:
            logger.debug(f"      - Stop failed (may already be stopped): {e}")

    def _remove_container(self, container):
        try:
            container.remove(force=True, v=True)
            logger.debug("      - Container and volumes removed")
        except Exception as e:
            logger.debug(f"      - Remove failed: {e}")

    def cleanup_all(self):
        if not self.client:
            return

        container_ids = list(self.containers.keys())
        logger.info(f"🧹 Cleaning up all containers ({len(container_ids)} total)")
        for container_id in container_ids:
            self.cleanup_container(container_id)
        logger.info("✅ All containers cleaned up")

    def _cleanup_orphaned_containers(self):
        if not self.client:
            return

        try:
            containers = self.client.containers.list(all=True, filters={'name': 'phishing-sandbox-'})
            if containers:
                logger.info(f"🧹 Found {len(containers)} orphaned sandbox container(s), cleaning up...")
                for container in containers:
                    try:
                        container.stop(timeout=5)
                    except Exception:
                        pass
                    try:
                        container.remove(force=True, v=True)
                        logger.info(f"   ✅ Removed orphaned container: {container.name}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Failed to remove orphaned container {container.name}: {e}")
        except Exception as e:
            logger.debug(f"Could not cleanup orphaned containers: {e}")

    def get_container_stats(self, container_id: str) -> Dict[str, Any]:
        if not self.client or container_id not in self.containers:
            return {}
        return self._fetch_stats(self.containers[container_id], container_id)

    def _fetch_stats(self, container, container_id: str) -> Dict[str, Any]:
        try:
            stats = container.stats(stream=False)
            logger.debug(f"   📊 Container stats retrieved for: {container_id}")
            return {
                'cpu_usage': stats.get('cpu_stats', {}),
                'memory_usage': stats.get('memory_stats', {}),
                'network': stats.get('networks', {})
            }
        except Exception as e:
            logger.debug(f"   ⚠️ Could not get container stats: {e}")
            return {}
