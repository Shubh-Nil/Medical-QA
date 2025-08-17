
from eval.evaluate import extract_citations, has_references_block, grounding_score

def test_citation_and_grounding():
    text = "Use amoxicillin [1] and consider macrolides [2].\n\nReferences\n[1] ...\n[2] ..."
    assert extract_citations(text) == [1,2]
    assert has_references_block(text)
    assert grounding_score(text, retrieved_count=2) == 1.0
