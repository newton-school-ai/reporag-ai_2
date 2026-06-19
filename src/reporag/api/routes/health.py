from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Simple health check endpoint returning status of the service."""
    return {"status": "healthy"}
