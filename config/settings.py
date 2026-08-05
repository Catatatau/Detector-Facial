import json
import os
from pathlib import Path
from pydantic import BaseModel, Field

class CameraConfig(BaseModel):
    index: int = Field(default=0, description="Camera index (e.g., 0 for default webcam)")
    resolution: tuple[int, int] = Field(default=(1280, 720), description="Camera resolution (width, height)")
    frame_rate: int = Field(default=30, description="Target frame rate")

class ModelConfig(BaseModel):
    detection_backend: str = Field(default="yunet", description="Face detection backend (yunet, opencv, retinaface)")
    recognition_backend: str = Field(default="insightface", description="Recognition model backend")
    recognition_threshold: float = Field(default=0.5, description="Cosine similarity threshold for recognition")

class DatabaseConfig(BaseModel):
    location: str = Field(default="sqlite:///face_platform.db", description="Database connection string")

class SystemConfig(BaseModel):
    logging_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    thread_count: int = Field(default=4, description="Number of threads for processing")

class Settings(BaseModel):
    camera: CameraConfig = CameraConfig()
    models: ModelConfig = ModelConfig()
    database: DatabaseConfig = DatabaseConfig()
    system: SystemConfig = SystemConfig()

    @classmethod
    def load(cls, config_path: str = "config.json") -> "Settings":
        path = Path(config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return cls(**data)
        else:
            # If config does not exist, create default
            settings = cls()
            settings.save(config_path)
            return settings

    def save(self, config_path: str = "config.json"):
        path = Path(config_path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4))

# Create a global configuration instance
CONFIG_PATH = os.getenv("FACE_PLATFORM_CONFIG", "config.json")
config = Settings.load(CONFIG_PATH)
