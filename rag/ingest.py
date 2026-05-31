from datasets import load_dataset


def load_PubMedQA(subset: str, split: str = "train") -> dict:
    dataset = load_dataset("qiaojin/PubMedQA", subset, split=split)
    return dataset 


def chunking(dataset: dict, subset: str) -> list[dict]:
    # Each data in the dataset contains several text 'chunks'
    chunks = [] 
    for data in dataset:
        item = data['context']

        for index, section in enumerate(item['labels']):
            chunk = {
                "text": item['contexts'][index],
                "meta": {
                    "pubid": data['pubid'],
                    "labels": section,
                    "meshes": ", ".join(item['meshes']),
                    "subset": subset
                }
            }
            chunks.append(chunk)
    return chunks
