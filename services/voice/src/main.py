import torch
from fastapi import FastAPI
from resemblyzer import VoiceEncoder
from src.api.routes import router
from src.api.routes import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(title='Blabin Voice Service', version='1.0.0')

    @app.on_event('startup')
    async def _load_models():
        # Limit CPU threads to avoid contention
        torch.set_num_threads(1)
        # Load Silero VAD
        app.state.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad'
        )
        # Load resemblyzer encoder
        app.state.voice_encoder = VoiceEncoder()

    app.include_router(router, prefix='')
    return app


app = create_app()

app.include_router(api_router)


@app.get('/')
async def read_root():
    return {'message': 'Welcome to the FastAPI Cloud App!'}
