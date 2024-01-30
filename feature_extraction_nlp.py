import numpy as np

def add_statistical_features(vectors):
    enhanced_vectors = []
    for vec in vectors:
        # Example statistical features
        mean = np.mean(vec)
        std = np.std(vec)
        max_value = np.max(vec)
        min_value = np.min(vec)

        # Create an enhanced vector
        enhanced_vector = np.append(vec, [mean, std, max_value, min_value])
        enhanced_vectors.append(enhanced_vector)
    
    return enhanced_vectors

