from dotenv import load_dotenv
import os
from constants import PROJECT_ROOT

load_dotenv(PROJECT_ROOT)

TELEGRAM_CONFIG = {
    "enabled": True,
    "token": os.getenv("TELEGRAM_TOKEN"),
    "chat_id": os.getenv("TELEGRAM_CHAT_ID"),
}

API_SERVER_CONFIG = {
    "enabled": True,
    "listen_ip_address": "127.0.0.1",
    "listen_port": 8080,
    "verbosity": "error",
    "enable_openapi": False,
    "jwt_secret_key": os.getenv("JWT_SECRET_KEY"),
    "ws_token": os.getenv("WS_TOKEN"),
    "CORS_origins": [],
    "username": os.getenv("API_USERNAME"),
    "password": os.getenv("API_PASSWORD"),
}
