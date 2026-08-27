import json
import os
from threading import Lock
import uuid
from pathlib import Path
from typing import Dict, Any, List

import docker
from docker.errors import DockerException, ImageNotFound

from config.logging_config import get_logger
from services.container_config import ContainerConfig
from services.sandbox_client import SandboxClient

logger = get_logger("ContainerOrchestrator")


class DockerSandboxClient:
    """Docker SDK implementation of the sandbox client port."""

    def __init__(self):
        logger.info("Initializing DockerSandboxClient...")
        self.client = self._connect_client()
        self.image_name = ContainerConfig.IMAGE_NAME
        self.containers: Dict[str, Any] = {}
        self._request_environment: Dict[str, Dict[str, str]] = {}
        self._warm_container = None
        self._warm_lock = Lock()
        self._execution_lock = Lock()
        self.host_backend_path = os.environ.get('HOST_BACKEND_PATH', '/app')
        self._log_init_status()
        self._cleanup_orphaned_containers()

    def __del__(self):
        if getattr(self, 'client', None):
            self.cleanup_all()

    def _log_init_status(self):
        status = 'Connected' if self.client else 'Not available'
        logger.info(f"DockerSandboxClient initialized | Docker client: {status}")
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
        return Path("/app/Dockerfile")

    def _get_app_path(self) -> Path:
        return Path("/app")

    def build_sandbox_image(self):
        """Build the configured Docker sandbox image when missing or requested."""
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
            target="sandbox",
            rm=True
        )
        logger.info(f"✅ Successfully built sandbox image: {self.image_name}")

    def ensure_image_exists(self):
        """Reuse or build the configured sandbox image."""
        self._require_client()
        if os.environ.get('REBUILD_SANDBOX_IMAGE', '').lower() in {'1', 'true', 'yes'}:
            logger.info(f"📦 Forced rebuild of sandbox image: {self.image_name}")
            self.build_sandbox_image()
            return

        try:
            self.client.images.get(self.image_name)
            logger.info(f"📦 Reusing sandbox image: {self.image_name}")
        except ImageNotFound:
            logger.info(f"📦 Sandbox image missing, building: {self.image_name}")
            self.build_sandbox_image()

    def ensure_image(self):
        """Expose the sandbox port name while retaining the legacy alias."""
        return self.ensure_image_exists()

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

    def _prepare_environment(
        self, urls: List[str], models: List[str], mode: str, dataset: str,
    ) -> Dict[str, str]:
        env = {'URLS': json.dumps(urls), 'MODELS': json.dumps(models), 'MODE': mode, 'DATASET': dataset}
        self._log_environment(urls, models, mode, dataset)
        return env

    def _log_environment(self, urls: List[str], models: List[str], mode: str, dataset: str):
        logger.debug("   Environment variables:")
        logger.debug(f"      - URLS: {urls}")
        logger.debug(f"      - MODELS: {models}")
        logger.debug(f"      - MODE: {mode}")
        logger.debug(f"      - DATASET: {dataset}")

    def _log_container_creation(
        self, container_id: str, container_name: str, urls: List[str], models: List[str],
        mode: str, dataset: str,
    ):
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

    def _run_container(self, container_name: str, volumes: Dict):
        nano_cpus = int(ContainerConfig.CPU_LIMIT * 1e9)
        return self.client.containers.run(
            self.image_name,
            command=["python", "-c", "import time; time.sleep(2147483647)"],
            detach=True,
            volumes=volumes,
            mem_limit=ContainerConfig.MEMORY_LIMIT,
            nano_cpus=nano_cpus,
            network_mode=ContainerConfig.NETWORK_MODE,
            remove=False,
            name=container_name,
            auto_remove=False
        )

    def _container_running(self, container) -> bool:
        try:
            container.reload()
            return container.status == "running"
        except DockerException:
            return False

    def _get_warm_container(self, volumes: Dict):
        if self._warm_container and self._container_running(self._warm_container):
            return self._warm_container
        name = f"phishing-sandbox-warm-{uuid.uuid4()}"
        self._warm_container = self._run_container(name, volumes)
        return self._warm_container

    def create_container(
        self, urls: List[str], models: List[str], mode: str = "optimized", dataset: str = 'current',
    ) -> str:
        """Create one isolated Docker analysis container."""
        self._require_client()
        self.ensure_image_exists()
        request_id = str(uuid.uuid4())
        environment = self._prepare_environment(urls, models, mode, dataset)
        volumes = self._prepare_volumes()
        self._log_container_config()
        with self._warm_lock:
            container = self._get_warm_container(volumes)
        self.containers[request_id] = container
        self._request_environment[request_id] = environment
        logger.info("   ✅ Warm sandbox ready for request %s", request_id)
        return request_id

    def _parse_container_logs(self, logs: str) -> Dict[str, Any]:
        for line in reversed(logs.splitlines()):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(result, dict):
                logger.debug("   📄 Container output parsed successfully")
                return result

        logger.error("   ❌ Failed to parse container output: no JSON response found")
        logger.debug(f"   Raw logs: {logs[:500]}...")
        return {'success': False, 'error': 'Failed to parse container output', 'raw_logs': logs}

    def _log_result_status(self, parsed_result: Dict[str, Any]):
        if parsed_result.get('success'):
            results_count = len(parsed_result.get('results', []))
            logger.info(f"   ✅ Analysis completed: {results_count} URL(s) processed")
        else:
            logger.warning(f"   ⚠️ Container reported failure: {parsed_result.get('error', 'Unknown error')}")

    def wait_for_result(self, container_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for one container and parse its JSON result."""
        self._require_client()
        if container_id not in self.containers:
            logger.error(f"❌ Container not found: {container_id}")
            raise ValueError(f"Container {container_id} not found")

        container = self.containers[container_id]
        environment = self._request_environment[container_id]
        logger.info(f"   ⏳ Waiting for container to complete (timeout: {timeout}s)...")

        try:
            with self._execution_lock:
                return self._execute_request(container, environment, timeout)
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

    def _execute_request(self, container, environment, timeout: int):
        command = ["timeout", f"{timeout}s", "python", "sandbox_runner.py"]
        result = container.exec_run(command, environment=environment, workdir="/app")
        output = getattr(result, "output", b"")
        exit_code = getattr(result, "exit_code", 0)
        logs = output.decode("utf-8") if isinstance(output, bytes) else str(output)
        if exit_code:
            logger.warning("Sandbox request exited with code %s", exit_code)
        return self._parse_container_logs(logs)

    def cleanup_container(self, container_id: str):
        """Stop and remove one owned container if it still exists."""
        if not self.client or container_id not in self.containers:
            return

        logger.info(f"   🧹 Releasing sandbox request: {container_id}")
        del self.containers[container_id]
        self._request_environment.pop(container_id, None)

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
        """Stop and remove every container owned by this client."""
        if not self.client:
            return

        containers = self._unique_owned_containers()
        logger.info(f"🧹 Cleaning up all containers ({len(containers)} total)")
        for container in containers:
            self._stop_container(container)
            self._remove_container(container)
        self.containers.clear()
        self._request_environment.clear()
        self._warm_container = None
        logger.info("✅ All containers cleaned up")

    def _unique_owned_containers(self):
        containers = {container.id: container for container in self.containers.values()}
        if self._warm_container:
            containers[self._warm_container.id] = self._warm_container
        return list(containers.values())

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
                    except DockerException as exc:
                        logger.debug("Orphaned container already stopped: %s", exc)
                    try:
                        container.remove(force=True, v=True)
                        logger.info(f"   ✅ Removed orphaned container: {container.name}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Failed to remove orphaned container {container.name}: {e}")
        except Exception as e:
            logger.debug(f"Could not cleanup orphaned containers: {e}")

    def get_container_stats(self, container_id: str) -> Dict[str, Any]:
        """Return Docker CPU, memory and network statistics for a container."""
        if not self.client or container_id not in self.containers:
            return {}
        return self._fetch_stats(self.containers[container_id], container_id)

    def _fetch_stats(self, container, container_id: str) -> Dict[str, Any]:
        """Read resource statistics from a Docker container."""
        try:
            stats = container.stats(stream=False)
            return {
                "cpu_usage": stats.get("cpu_stats", {}),
                "memory_usage": stats.get("memory_stats", {}),
                "network": stats.get("networks", {}),
            }
        except Exception as exc:
            logger.debug("Could not get container stats: %s", exc)
            return {}


class ContainerOrchestrator:
    """Coordinate sandbox operations through an injectable client port."""

    def __init__(self, client: SandboxClient | None = None):
        """Create an orchestrator with Docker as the production default."""
        self.client = client or DockerSandboxClient()

    def ensure_image(self):
        """Delegate sandbox image preparation."""
        return self.client.ensure_image()

    def create_container(self, urls, models, mode="optimized", dataset="current"):
        """Delegate container creation to the sandbox client."""
        return self.client.create_container(urls, models, mode, dataset)

    def wait_for_result(self, container_id, timeout=300):
        """Delegate result collection to the sandbox client."""
        return self.client.wait_for_result(container_id, timeout)

    def cleanup_container(self, container_id):
        """Delegate one-container cleanup."""
        return self.client.cleanup_container(container_id)

    def cleanup_all(self):
        """Delegate cleanup of all containers."""
        return self.client.cleanup_all()

    @staticmethod
    def _parse_container_logs(logs: str) -> Dict[str, Any]:
        """Parse the last JSON object in container logs for compatibility tests."""
        for line in reversed(logs.splitlines()):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(result, dict):
                return result
        return {"success": False, "error": "Failed to parse container output", "raw_logs": logs}
