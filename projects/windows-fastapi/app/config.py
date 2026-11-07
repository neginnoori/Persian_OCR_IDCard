from pathlib import Path
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='OCR_', env_file='.env', extra='ignore')
    api_key: SecretStr
    model_dir: Path = Path('./models')
    max_upload_mb: int = Field(default=15, ge=1, le=100)
    max_pixels: int = Field(default=25_000_000, ge=1024, le=100_000_000)
    max_pending: int = Field(default=2, ge=0, le=16)
    request_timeout_seconds: float = Field(default=180, gt=0, le=1800)
    upload_timeout_seconds: float = Field(default=30, gt=0, le=300)

    @field_validator('api_key')
    @classmethod
    def validate_key(cls, value):
        key = value.get_secret_value()
        if len(key) < 32 or not key.isascii() or key.strip() != key or key.startswith('CHANGE_ME'):
            raise ValueError('Set OCR_API_KEY to a random ASCII secret of at least 32 characters')
        return value

    @property
    def upload_bytes(self):
        return self.max_upload_mb * 1024 * 1024
