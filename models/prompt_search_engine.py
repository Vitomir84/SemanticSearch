from typing import Sequence, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class PromptSearchEngine:
    '''Instanciate the language model and index for searching the most similar prompts. Performs the semantic search.'''
    def __init__(self, model_name='bert-base-nli-mean-tokens'):
        print("Search engine started!")
        self.model = SentenceTransformer(model_name)
        # Initialize FAISS index with right number of dimensions
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.embedding_dimension)  # Euclidian distance index - brute force for small datasets
        self.prompts_track = []  # To keep track of original prompts for returning results


    def add_prompts_to_vector_database(self, prompts):
        print("Data encoding started...")
        embeddings = self.model.encode(prompts)
        self.index.add(np.array(embeddings).astype('float32'))  
        self.prompts_track.extend(prompts)
        print("Data encoding completed!")


    def most_similar(self, query, top_k=5):
        # Encode the 
        print('Finding the most similar vectors')
        query_embedding = self.model.encode([query]).astype('float32')
        
        # Optimizovana pretraga ali moramo promeniti vrstu indeksa za pretragu kod stvarne upotrebe
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve the corresponding prompts for the found indices
        similar_prompts = [self.prompts_track[idx] for idx in indices[0]]
        
        return similar_prompts, distances[0]  # Return both the similar prompts and their distances


    def cosine_similarity(self, query_vector, index):
        """Compute the cosine similarity between a query vector and a set of corpus vectors.
            Args: query_vector: The query vector to compare against the corpus vectors. corpus_vectors: The set of corpus vectors to compare against the query vector. 
            Returns: The cosine similarity between the query vector and the corpus vectors.
            """
        print('Searching for all similarities...')
        query_vector = np.array(query_vector).astype('float32')
        query_norm = query_vector / np.linalg.norm(query_vector)

        # Get all vectors from FAISS
        index_vectors = index.reconstruct_n(0, index.ntotal)  # Reconstruct all vectors in the index
        index_norms = np.linalg.norm(index_vectors, axis=1, keepdims=True)
        normalized_index_vectors = index_vectors / index_norms
        cosine_similarities = np.dot(normalized_index_vectors, query_norm.T)

        return cosine_similarities

                
    
      