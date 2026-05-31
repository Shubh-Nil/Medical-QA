`conda create --name medqa python==3.12`

`conda activate medqa`
`pip install -r requirements.txt`

Create mode: To create vector database - 
    For local machine/ no cluster: `python rag/store.py`
    For cluster: `sbatch job.sh`    (comment out the Retrieve mode)

Retrieve mode: To inference on a query -
    For local machine/ no cluster: `python rag/inference.py`
    For cluster: `sbatch job.sh`    (comment out the Create mode)

**Optional:**
To load the dataset at a faster rate from HuggingFace, go from Unauthenticated mode to Authenticated mode
`echo export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" >> ~/.bashrc` 


`logs/medical_qa_xxxx.err` shows you the status of dataset loading from HuggingFace
`logs/medical_qa_xxxx.out` shows you the generated answer + reference by the LLM for your query
                        

2 files will be created in the `database/chroma/` directory
- *chroma.sqlite3*:   store text chunks and its metadata
                     (result of the function 'chunking' in the rag/ingest.py during 'Create mode')
- *abcd-12-vwxyz-34*: **vector database** which stores indices, text chunks, text embeddings and  metadata
                     (result of the function 'add' in the rag/database.py during 'Create mode')


All the models from HuggingFace will be stored at `~/.cache/huggingface/hub/`

PROBLEMS:
1 - Hallucination
Ways to avoid **"Hallucination"** in this pipeline:
    Vector Database — gives the model relevant and factual context
    Strict prompt — tells the model to stay within that context

Hallucination of answers:
The answer is citing original chunk but hallucinating an incorrect statement in the answer.

Hallucination of references: 
Some models are trained that inline citations like [1] comes with a reference at the end. 
So when you will ask the model to include inline citations in the answer, they will cite the original chunk. 
But generate a References section, and hallucinate a fake document corresponding to the original citation.
  - Explicitly mention not to generate References section.

2 - Answer cutoff mid-sentence
    Reduce the `num_references` or increase the `max_new_tokens`


