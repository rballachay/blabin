from fastapi import FastAPI
from src.api.routes import router as api_router

app = FastAPI()

app.include_router(api_router)


@app.get('/')
async def read_root():
    return {'message': 'Welcome to the FastAPI Cloud App!'}
