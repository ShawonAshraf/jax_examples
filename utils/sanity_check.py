import os

import jax
from jax.lib import xla_bridge
from loguru import logger


def main():
    logger.info("Generatintg PRNGKey")
    key = jax.random.PRNGKey(0)
    logger.info("Splitting PRNGKey")
    key, *subs = jax.random.split(key, 10)
    logger.success("All Good!")


if __name__ == "__main__":
    device = xla_bridge.get_backend().platform

    # especially for github runners
    if device == "cpu":
        logger.info("Setting JAX_PLATFORMS to cpu")
        os.environ["JAX_PLATFORMS"] = "cpu"

    main()
