"""Core package for the social media sentiment analysis project."""

from importlib import metadata


def get_version() -> str:
    """Return the installed package version if available."""

    try:
        return metadata.version("social-media-sentiment-analysis")
    except metadata.PackageNotFoundError:
        return "0.1.0-dev"


__all__ = ["get_version"]
