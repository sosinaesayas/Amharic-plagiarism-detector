import os
import re
from amharicNormalizer import AmharicNormalizer
def normalizer(text):
    normalizer = AmharicNormalizer()

    cleaned_text = remove_selected_symbols(text)
    return normalizer.normalize(cleaned_text)


def remove_selected_symbols(text):
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)

    # Regular expression pattern that matches the specific symbols and English punctuation
    pattern = r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~፣።፤፡?]'

    # Using re.sub() to replace the matched characters with an empty string
    cleaned_text = re.sub(pattern, '', text)

    return cleaned_text
