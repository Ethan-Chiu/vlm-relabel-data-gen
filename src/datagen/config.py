from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# Always resolve config files relative to the project root, not the cwd
_PROJECT_ROOT = Path(__file__).parent.parent.parent


class Config(BaseSettings):
    """Pipeline configuration.

    Priority (highest → lowest):
      1. Environment variables  (prefix: DATAGEN_)
      2. .env file
      3. config.toml
      4. Defaults below

    Example env overrides:
      DATAGEN_VLM_BACKEND=vllm
      DATAGEN_CONCURRENCY=32
      DATAGEN_GEMINI_API_KEY=your_key
    """

    model_config = SettingsConfigDict(
        toml_file=str(_PROJECT_ROOT / "config.toml"),
        env_prefix="DATAGEN_",
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
    )

    # --- Download ---
    output_dir: Path = Path("data/raw")
    timeout: int = 10

    # --- Storage ---
    metadata_path: Path = Path("data/metadata.parquet")
    relabeled_path: Path = Path("data/relabeled.parquet")

    # --- VLM ---
    vlm_backend: str = "gemini"          # "gemini" | "vllm"
    vlm_model: str = "gemini-2.0-flash"
    vlm_base_url: str = "http://localhost:8000/v1"
    vlm_prompt: str = "Describe this image in one concise sentence."
    concurrency: int = 8

    # --- Secrets (set via env or .env, never in config.toml) ---
    gemini_api_key: str = Field(default="", repr=False)
    openai_api_key: str = Field(default="", repr=False)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        **kwargs: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            kwargs["init_settings"],       # highest priority
            kwargs["env_settings"],        # shell env vars (DATAGEN_*)
            kwargs["dotenv_settings"],     # .env file
            TomlConfigSettingsSource(settings_cls),  # config.toml
        )                                  # defaults = lowest priority


def load(toml_file: str = "config.toml") -> Config:
    """Load config from a specific TOML file path."""
    toml_path = str((Path(toml_file) if Path(toml_file).is_absolute() else _PROJECT_ROOT / toml_file))

    class _Config(Config):
        model_config = SettingsConfigDict(
            toml_file=toml_path,
            env_prefix="DATAGEN_",
            env_file=str(_PROJECT_ROOT / ".env"),
            env_file_encoding="utf-8",
        )

    return _Config()
