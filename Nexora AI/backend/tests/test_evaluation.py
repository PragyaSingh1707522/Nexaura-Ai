from types import SimpleNamespace
from app.evaluate import _retrieval_metrics

def test_recall_mrr_and_ndcg_use_supplied_relevance_labels():
    results=[(0.8,SimpleNamespace(id=4)),(0.7,SimpleNamespace(id=8)),(0.5,SimpleNamespace(id=2))]
    metrics=_retrieval_metrics(results,[8,2],5)
    assert metrics["recall"]==1.0
    assert metrics["mrr"]==0.5
    assert 0<metrics["ndcg"]<1
