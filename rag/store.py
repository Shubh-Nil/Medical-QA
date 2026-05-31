import argparse
from ingest import load_PubMedQA, chunking
from database import vector_database


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, required=True, choices=["pqa_labeled", "pqa_artificial", "pqa_unlabeled"], help="PubMedQA subset to ingest")
    args = parser.parse_args()

    dataset = load_PubMedQA(args.subset)
    chunks = chunking(dataset, args.subset)
    
    database = vector_database(model_name="sentence-transformers/all-MiniLM-L6-v2",
                               database_path="database/chroma")
    database.add(chunks)
