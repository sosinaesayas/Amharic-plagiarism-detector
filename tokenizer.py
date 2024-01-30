import os

def tokenize_amharic(text):
    # Basic tokenization by splitting on spaces
    # This will separate the text into words
    return text.split()

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        for word in content:
            file.write(word + '\n')  # Writing each word in a new line

def main():
    input_dir = os.path.join(os.path.dirname(__file__), '..', 'segmented')  # Adjust as per your directory structure
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'tokenized')
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt'):  # Assuming the files are in .txt format
            file_path = os.path.join(input_dir, file_name)
            text = read_file(file_path)

            tokenized_text = tokenize_amharic(text)
            output_file_path = os.path.join(output_dir, file_name)
            write_file(output_file_path, tokenized_text)

if __name__ == "__main__":
    main()



 