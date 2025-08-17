
from rag.embed_store import EmbedStore
def test_store_roundtrip(tmp_path):
    store = EmbedStore()
    store.add(["hello world","medical guideline text"], [{"source":"t1"},{"source":"t2"}])
    out = tmp_path / "idx"
    store.save(str(out))
    store2 = EmbedStore.load(str(out))
    res = store2.search("guideline", k=1)
    assert res and isinstance(res[0].score, float)
