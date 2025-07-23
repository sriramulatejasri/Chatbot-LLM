from sklearn.metrics.pairwise import cosine_similarity
import torch

def find_best_match(user_query,prompts,responses,embeddings,model):
    query_embedding=model.encode([user_query],convert_to_tensor=True)
    similarities=cosine_similarity(query_embedding,embeddings)[0]
    best_index=torch.argmax(torch.tensor(similarities)).item()
    return responses[best_index]