import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# Create the stop-word list
stop_words = set(stopwords.words("english"))

# Create the lemmatizer
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Clean and preprocess resume text.
    """

    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 3. Tokenize the text
    tokens = word_tokenize(text)

    # 4. Remove stop words
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Lemmatize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 6. Convert the words back into a single string
    cleaned_text = " ".join(tokens)

    return cleaned_text
