import jax
from loguru import logger


def main():
    logger.info("Generating PRNGKey")
    key = jax.random.PRNGKey(0)
    logger.info("Splitting PRNGKey")
    key, *subs = jax.random.split(key, 10)
    logger.success("All Good!")


if __name__ == "__main__":
    main()
