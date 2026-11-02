import os
import pytest
from app.retrieval import embed,cosine,embedding_info

@pytest.mark.skipif(os.getenv("RUN_SEMANTIC_TESTS")!="1",reason="set RUN_SEMANTIC_TESTS=1 to load the configured embedding model")
def test_configured_sentence_transformer_captures_paraphrase_similarity():
    embed("warm-up embedding")
    assert embedding_info()["semantic"] is True
    query=embed("A service provider must encrypt customer data stored on its systems.",is_query=True)
    paraphrase=embed("The vendor must protect information at rest using encryption.")
    unrelated=embed("The agreement renews automatically for one year unless terminated.")
    assert len(query)==384
    assert cosine(query,paraphrase)>cosine(query,unrelated)
