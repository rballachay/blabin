from api.routes import router as api_router
from fastapi import FastAPI

app = FastAPI()

app.include_router(api_router)


@app.get('/')
async def read_root():
    return {'message': 'Welcome to the FastAPI Cloud App!'}
