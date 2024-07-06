from jax.lib import xla_bridge
from loguru import logger


def log_device():
    device = xla_bridge.get_backend().platform

    assert device in ["cpu", "gpu", "tpu"], logger.error(f"Unknown device: {device}")

    logger.info(f"XLA DEVICE: {device}")


if __name__ == "__main__":
    log_device()
