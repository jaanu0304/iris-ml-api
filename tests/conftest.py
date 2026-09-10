import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings


@pytest.fixture
def client():
    with TestClient(
        app,
        headers={"X-API-Key": settings.API_KEY}
    ) as test_client:
        yield test_client