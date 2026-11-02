from types import SimpleNamespace
from app.config import settings
from app.retrieval import bm25_scores,clear_embedding_cache,embed,index_qdrant_batch,_qdrant_client

def test_bm25_prefers_relevant_text():
    docs=[SimpleNamespace(content="Customer personal information retention is limited."),SimpleNamespace(content="Office lease rent escalates each year.")]
    scores=bm25_scores("personal information retention",docs)
    assert scores[0]>scores[1]

def test_qdrant_local_in_memory_upsert_and_search(monkeypatch):
    monkeypatch.setattr(settings,"qdrant_url","memory://")
    monkeypatch.setattr(settings,"qdrant_collection","test_vectors")
    monkeypatch.setattr(settings,"embedding_model","hashing-fallback")
    clear_embedding_cache();vector=embed("privacy encryption");
    assert index_qdrant_batch([(1001,vector,{"document_id":44,"page":2})])
    hits=_qdrant_client().query_points("test_vectors",query=vector,limit=1).points
    assert hits and hits[0].id==1001 and hits[0].payload["page"]==2
