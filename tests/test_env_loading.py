import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


def test_env_example_contains_provider_keys():
    values = dotenv_values(Path(__file__).parents[1] / ".env.example")
    assert "OPENAI_API_KEY" in values
    assert "OPENROUTER_API_KEY" in values
    assert values["OPENROUTER_MODEL"] == "openai/gpt-5.6-sol"


def test_exported_environment_has_priority(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_MODEL=from-file\n")
    monkeypatch.setenv("OPENROUTER_MODEL", "from-shell")
    load_dotenv(env_file, override=False)
    assert os.environ["OPENROUTER_MODEL"] == "from-shell"
