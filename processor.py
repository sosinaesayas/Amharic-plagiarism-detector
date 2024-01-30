import csv
import pandas as pd
import os
from normalizer import normalizer
from lemmatizer import lemmatize_text
from stopword_remover import remove_stopwords
from stemmizer import stem
from feature_extraction import generate_flair_embeddings
from feature_extraction_nlp import add_statistical_features

def tokenize(text):
    return text.split()

def process_text(text):
    # Normalization, tokenization, lemmatization, and stopword removal
    normalized_text = normalizer(text)
    tokens = tokenize(normalized_text)
    lemmatized_tokens = [lemmatize_text(token) for token in tokens]
    lemmatized_text = ' '.join(lemmatized_tokens)
    stopwords_removed_text = remove_stopwords(lemmatized_text)
    return stopwords_removed_text


def clean_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            # Limit to only two fields per row
            if len(row) > 2:
                # Combine extra fields
                row = [row[0], ' '.join(row[1:])]
            writer.writerow(row)


def write_features_to_csv(features, filename):
    # Write the features to a CSV file
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([str(i) for i in range(len(features[0]))])  # Write header
        for feature in features:
            writer.writerow(feature)
def main():
    dataset_file = os.path.join(os.path.dirname(__file__), 'cleaned.csv')
    cleaned_dataset_file = os.path.join(os.path.dirname(__file__), 'cleaned.csv')
    embeddings_file = os.path.join(os.path.dirname(__file__), 'sample_flair_embeddings.csv')
    final_features_file = os.path.join(os.path.dirname(__file__), 'sample_statistical_features.csv')
    # clean_csv(dataset_file, cleaned_dataset_file)
    # Read dataset
    df = pd.read_csv(cleaned_dataset_file)

    # Initialize an empty list to store flair embeddings
    all_flair_embeddings = []
    print("Generating flair embeddings...")
    for index, row in df.iterrows():
        print("iterated " ,  index , " times" )
        processed_text = process_text(row['PostContent'])
        
        # Generate flair embeddings for each text
        flair_embeddings = generate_flair_embeddings(processed_text)
        all_flair_embeddings.extend(flair_embeddings)  # Store the embeddings

    # Write the accumulated flair embeddings to a CSV file
    write_features_to_csv(all_flair_embeddings, embeddings_file)

    print(f"Flair embeddings written to {embeddings_file}")
    # Read flair embeddings from file
    df = pd.read_csv(cleaned_dataset_file)

    flair_df = pd.read_csv(embeddings_file)

    final_df = pd.DataFrame()
    print("Adding statistical features...")
    for index, row in flair_df.iterrows():
        print("iterated " ,  index , " times")
        # Add statistical features to each embedding
        statistical_features = add_statistical_features([row.tolist()])
        final_row = pd.concat([pd.DataFrame(statistical_features), pd.DataFrame({'label': df.iloc[index]['label']}, index=[index])], axis=1)
        final_df = pd.concat([final_df, final_row])

    # Save the final features to CSV file
    final_df.to_csv(final_features_file, index=False)
    print(f"Final features written to {final_features_file}")

if __name__ == "__main__":
    main()
