from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
from app.main import app
from app.config import settings
from app.retrieval import clear_embedding_cache

def test_cors_preflight_allows_vite_5174_origin():
    with TestClient(app) as client:
        response=client.options(
            "/api/health",
            headers={
                "Origin":"http://localhost:5174",
                "Access-Control-Request-Method":"GET",
                "Access-Control-Request-Headers":"authorization,content-type",
            },
        )
    assert response.status_code==200
    assert response.headers["access-control-allow-origin"]=="http://localhost:5174"
    assert response.headers["access-control-allow-credentials"]=="true"

def test_upload_and_ask_returns_grounded_source(monkeypatch):
    engine=create_engine("sqlite://",connect_args={"check_same_thread":False},poolclass=StaticPool)
    TestingSession=sessionmaker(bind=engine,autoflush=False,autocommit=False)
    Base.metadata.create_all(bind=engine)
    def override_db():
        db=TestingSession()
        try: yield db
        finally: db.close()
    app.dependency_overrides[get_db]=override_db
    monkeypatch.setattr(settings,"qdrant_url","http://127.0.0.1:1")
    monkeypatch.setattr(settings,"embedding_model","hashing-fallback");clear_embedding_cache()
    try:
        with TestClient(app) as client:
            upload=client.post("/api/documents/upload",files={"file":("policy.txt",b"Personal data must be retained for 90 days and encrypted at rest.","text/plain")})
            assert upload.status_code==200
            response=client.post("/api/assistant/ask",json={"question":"How long is personal data retained?"})
            assert response.status_code==200
            body=response.json()
            assert body["sources"][0]["filename"]=="policy.txt"
            assert "90 days" in body["answer"]
            assert body["provider"]=="extractive-fallback"
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
