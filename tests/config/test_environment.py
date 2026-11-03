import os

import pytest

from robot_dock.config.environment import PokebotEnvironment, load_environment


def test_environment_does_not_override_process_values(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("ROBOT_DOCK_POKEBOT_API_KEY=file-value\n", encoding="utf-8")
    monkeypatch.setenv("ROBOT_DOCK_POKEBOT_API_KEY", "process-value")
    load_environment(path)
    assert PokebotEnvironment.read().api_key == "process-value"


def test_environment_keeps_literal_values_and_redacts_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOT_DOCK_POKEBOT_API_KEY", raising=False)
    monkeypatch.delenv("ROBOT_DOCK_POKEBOT_MODEL", raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "ROBOT_DOCK_POKEBOT_API_KEY=literal-${HOME}\nROBOT_DOCK_POKEBOT_MODEL=test-model\n"
    )
    # Restore process environment after testing the actual loader.
    with monkeypatch.context() as context:
        context.setattr(os, "environ", os.environ.copy())
        load_environment(path)
        settings = PokebotEnvironment.read()
        assert settings.api_key == "literal-${HOME}"
        assert settings.model == "test-model"
        assert settings.api_key not in repr(settings)


def test_missing_explicit_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_environment(tmp_path / "missing", required=True)


def test_vendor_environment_variable_remains_supported(monkeypatch):
    monkeypatch.delenv("ROBOT_DOCK_POKEBOT_API_KEY", raising=False)
    monkeypatch.setenv("PLATFORM_API_KEY", "vendor-value")
    assert PokebotEnvironment.read().api_key == "vendor-value"
