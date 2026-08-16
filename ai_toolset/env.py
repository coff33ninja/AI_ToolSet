"""Project-local .env loading.

Loads <repo root>/.env once per process, without overriding variables that
are already set in the environment (a shell export, the web UI, etc.). The
entry points (CLI, web, ui) call load_env() at startup; token consumers
(``ensure_diarize``, ``pipeline_kwargs``) also call it so direct library use
works without the CLI wrapper. See .env.example for the supported variables.
"""

from pathlib import Path

_loaded = False


def project_root():
    """Repo root (the directory containing the ai_toolset package)."""
    return Path(__file__).resolve().parent.parent


def load_env():
    """Load the repo's .env into os.environ once. Safe to call anywhere."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    env_path = project_root() / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        pass
