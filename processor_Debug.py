import csv
from normalizer import normalizer
from lemmatizer import lemmatize_text
from stopword_remover import remove_stopwords
from stemmizer import stem
from feature_extraction import generate_flair_embeddings
from feature_extraction_nlp import add_statistical_features

text = "ውሀ ቢወቅጡት እምቦጭ ፣ ሖኗል የእናንተ ነገር ነው ና ነይ"

def tokenize(text):
    return text.split()

def process_text(text):
    # normalize the text
    normalized_text = normalizer(text)

    print("normalizing text : " , normalized_text)
    # tokenize the text
    tokens = tokenize(normalized_text)
    print("tokenizing text : " , tokens)
    # lemmatize the text
    lemmatized_tokens = [lemmatize_text(token) for token in tokens]
    
    # join the tokens back to text 
    lemmatized_text = ' '.join(lemmatized_tokens)
    print("lemmatizing text : " , lemmatized_text)
    stopwords_removed_text = remove_stopwords(lemmatized_text)
    print("removing stopwords : " , stopwords_removed_text)


    print("generating flair embeddings")
    # flair embeddings
    flair_embeddings = generate_flair_embeddings(stopwords_removed_text)

    write_features_to_csv(flair_embeddings, "flair_embeddings.csv")
    print("flair embeddings generated --- adding statistical features")

    # add statistical features
    statistical_features = add_statistical_features(flair_embeddings)
    print("statistical features added")
    return statistical_features

def write_features_to_csv(features, filename="statistical_features.csv"):
    # Write the features to a CSV file
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write a header
        writer.writerow([ str(i) for i in range(len(features[0]))])
        # Write the features
        for feature in features:
            writer.writerow(feature)
    print(f"Features written to {filename}")

statistical_features = process_text(text)
write_features_to_csv(statistical_features)