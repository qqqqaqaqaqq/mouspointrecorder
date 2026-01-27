from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # DB
    DB_HOST: str
    DB_PORT: int = 5432
    DB_NAME: str = "postgres"
    DB_USER: str
    DB_PASSWORD: str
    
    SEQ_LEN : int = 300
    STRIDE : int = 50
    
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg2://"
            f"{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    JsonPath: str = "./"
    Recorder: str = "postgres"

    class Config:
        env_file = ".env"
        case_sensitive = True
    
settings = Settings()
