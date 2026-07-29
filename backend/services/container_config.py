"""Resource and network limits for sandbox containers."""


class ContainerConfig:
    """Static Docker sandbox configuration shared by orchestration code."""
    IMAGE_NAME = "phishing-detector-sandbox"
    MEMORY_LIMIT = '2g'
    CPU_LIMIT = 2.0
    NETWORK_MODE = 'bridge'
