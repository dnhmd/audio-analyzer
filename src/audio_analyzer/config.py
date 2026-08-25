from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    sample_rate: int = 16000
    max_audio_duration_sec: float = 30.0
    max_inference_duration_sec: float = 3.0
    inference_timeout_sec: float = 10.0
    snr_threshold_db: float = 10.0
    silence_ratio_threshold: float = 0.8

settings = Settings()