"""Application logging configuration without logging secret values."""

import logging


def configure_logging(level: str) -> None:
    """Set the root logging level and a concise, consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )