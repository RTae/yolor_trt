from fastapi import FastAPI
from src.routes.routes import api_router

app = FastAPI(title="MaineCoon")

app.include_router(api_router)