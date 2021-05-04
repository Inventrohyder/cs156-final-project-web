import pickle

import nltk
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

nltk.download("stopwords")

vectorizer: CountVectorizer = CountVectorizer(
    decode_error="replace",
    vocabulary=pickle.load(open("vocabulary.pkl", "rb")),
    preprocessor=lambda x: x,
    tokenizer=lambda x: x,
)
interpreter = tf.lite.Interpreter(model_path="model_quant.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


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
    vector = vectorizer.transform(words).toarray()
    input_data = np.array(vector, dtype=np.float32)
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    prediction = round(interpreter.get_tensor(output_index)[0][0])
    return str(prediction)


if __name__ == "__main__":
    app.run()
