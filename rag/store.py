from ingest import *
from database import *


if __name__ == '__main__':
    print("Enter the subset you want to add to the database")
    print("1 - Labeled")
    print("2 - Artificially labeled")
    print("3 - Unlabeled")
    choice = int(input("Enter the number"))

    if choice==1: subset = "pqa_labeled"
    elif choice==2: subset = "pqa_artificial"
    elif choice==3: subset = "pqa_unlabeled"
    else: 
        print("Incorrect choice selected")
        exit()

    dataset = load_PubMedQA(subset)
    chunks = chunking(dataset, subset)
    
    database = vector_database(model_name="sentence-transformers/all-MiniLM-L6-v2",
                               database_path="database/chroma")
    database.add(chunks)
