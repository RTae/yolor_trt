import io
from celery.result import AsyncResult
from fastapi.responses import JSONResponse
from src.services.model_service import helper
from fastapi import APIRouter, File, UploadFile
from starlette.responses import StreamingResponse
from src.controllers.model_controller import controller_inference

router = APIRouter()

@router.post("/inference")
async def inference(image: UploadFile = File(...)):
    contents = await image.read()
    task_id = controller_inference(contents)
    return {'task_id': str(task_id), 'status': 'Processing'}

@router.get('/result/{task_id}')
async def result(task_id):
    task = AsyncResult(task_id)
    if not task.ready():
        return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
    result = helper().imgStr2imgByte(task.get())
    return StreamingResponse(io.BytesIO(result), media_type='image/jpg')