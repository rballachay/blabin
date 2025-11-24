from fastapi import APIRouter

router = APIRouter()


@router.get('/health')
async def health_check():
    return {'status': 'healthy'}


@router.post('/vad')
async def vad_endpoint(audio_data: bytes):
    # Placeholder for VAD processing logic
    return {'message': 'VAD processing not implemented yet.'}


@router.post('/recognize')
async def recognize_endpoint(audio_data: bytes):
    # Placeholder for voice recognition logic
    return {'message': 'Voice recognition not implemented yet.'}
