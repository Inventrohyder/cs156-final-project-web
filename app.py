import joblib
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow import keras
from tensorflow.keras.models import Sequential

app = Flask(__name__)

vectorizer: CountVectorizer = CountVectorizer(
    decode_error="replace",
    vocabulary=joblib.load("vocabulary.pkl"),
    preprocessor=lambda x: x,
    tokenizer=lambda x: x,
)
model: Sequential = keras.models.load_model("model.h5")


@app.route("/")
def main():
    return render_template("index.html")


def review_to_words(review: str) -> list:
    """
    Cleans up a single review and returns the words that it contains.
    The cleanup includes:
    * removing HTML tags
    * converting to lowercase
    * removing stop words in English
    * getting the stem of the words (root)

    :param review: The review to process

    :returns a list of root words in lower case format that are cleaned
        from a review.
    """
    # Remove HTML tags
    text = BeautifulSoup(review, "html.parser").get_text()

    # Convert to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Split string into words
    words = text.split()

    # Remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]

    # stem
    words = [PorterStemmer().stem(w) for w in words]

    return words


@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]
    words = [review_to_words(review)]
    vector = vectorizer.transform(words)
    prediction = round(model.predict(vector)[0][0])
    return str(prediction)


if __name__ == "__main__":
    app.run()
