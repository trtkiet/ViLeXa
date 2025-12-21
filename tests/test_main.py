import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

def test_healthcheck():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
def test_chat_endpoint():
    payload = {"message": "Hello, how are you?"}
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert isinstance(data["reply"], str)
    print("Chat reply:", data["reply"])
    