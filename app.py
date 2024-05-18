from flask import Flask, render_template, request, jsonify
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

loaded_model = joblib.load('logistic_model.joblib')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]  # Remove non-alphabetic characters
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return ' '.join(words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    preprocessed_tweet = preprocess_text(tweet)
    prediction = loaded_model.predict([preprocessed_tweet])[0]
    if prediction == 0:
        result = "Hate Speech"
    elif prediction == 1:
        result = "Offensive Language"
    else:
        result = "Not Offensive"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
