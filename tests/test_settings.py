from catalyst.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
