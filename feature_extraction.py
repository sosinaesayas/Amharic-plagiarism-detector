import pandas as pd
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings
import numpy as np
from pathlib import Path

# Function to generate embeddings using Flair
def generate_flair_embeddings(text):
    embedding_model = 'am-forward'
    # Load Flair embeddings
    embeddings = FlairEmbeddings(embedding_model)
    vectors = []


    sentence = Sentence(text)
    embeddings.embed(sentence)
    # Use mean pooling to get a single vector for the sentence
    sentence_embedding = np.mean([token.embedding.numpy() for token in sentence], axis=0)
    
    return [sentence_embedding]

