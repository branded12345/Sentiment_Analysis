from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import logging
logging.basicConfig(filename="logfilename.log", level=logging.INFO)

nltk.download('stopwords')

stopwords_set = set(stopwords.words('english'))
emoticon_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')


app = Flask(__name__)
logging.info("Program Start")

# Load the sentiment analysis model and TF-IDF vectorizer
with open('models/clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

logging.info("Pickle test passed")




def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoticon_pattern.findall(text)
    text = re.sub('[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')
    prter = PorterStemmer()
    text = [prter.stem(word) for word in text.split() if word not in stopwords_set]

    return " ".join(text)

logging.info("Method passed")

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def analyze_sentiment():
    logging.info("Came inside")
    if request.method == 'POST':
        comment = request.form.get('comment')

        logging.info("Taken the comment")
        # Preprocess the comment
        preprocessed_comment = preprocessing(comment)

        logging.info("Preprocessing done")
        # Transform the preprocessed comment into a feature vector
        comment_vector = tfidf.transform([preprocessed_comment])

        logging.info("Vector Transform DOne")
        # Predict the sentiment
        sentiment = clf.predict(comment_vector)[0]

        logging.info("Prediction done and stored")

        return render_template('home.html', sentiment=sentiment)
    logging.info("Rendered out")
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
