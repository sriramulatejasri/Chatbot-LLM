import pandas as pd
from sentence_transformers import SentenceTransformer

model=SentenceTransformer("all-MiniLM-L6-v2")
def load_data(csv_path='data\codebasics_faqs.csv'):
    df=pd.read_csv(csv_path,encoding='ISO-8859-1')
    prompts=df["prompt"].tolist()
    responses=df["response"].tolist()
    embeddings=model.encode(prompts,convert_to_tensor=True)
    return prompts,responses,embeddings
