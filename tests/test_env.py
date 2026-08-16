"""Project setup: .env loading and env.example contract."""

import os

from ai_toolset.env import load_env, project_root


def test_project_root_is_repo_root():
    root = project_root()
    assert (root / "ai_toolset").is_dir()
    assert (root / "pyproject.toml").is_file()


def test_load_env_idempotent(monkeypatch):
    monkeypatch.delenv("AITOOLSET_TEST_KEY", raising=False)
    load_env()
    load_env()
    assert "AITOOLSET_TEST_KEY" not in os.environ


def test_env_example_contract():
    env_example = project_root() / ".env.example"
    assert env_example.is_file(), "missing .env.example"
    env = project_root() / ".env"
    example_keys = {
        line.split("=", 1)[0].strip()
        for line in env_example.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    assert example_keys, ".env.example has no KEY= entries"
    if env.is_file():
        env_keys = {
            line.split("=", 1)[0].strip()
            for line in env.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#") and "=" in line
        }
        assert env_keys.issubset(example_keys), (
            f".env keys not in .env.example: {env_keys - example_keys}"
        )
