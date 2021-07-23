from fastapi import APIRouter
from src.routes import model_route

api_router = APIRouter()
api_router.include_router(model_route.router, prefix="/model", tags=["model"])