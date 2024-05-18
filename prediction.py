# Load the saved model
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


# Assuming you have a new tweet as test data
new_tweet = "bitch"

# Preprocess the new tweet data similarly to your training data
preprocessed_tweet = preprocess_text(new_tweet)

# Use the loaded model to predict
prediction = loaded_model.predict([preprocessed_tweet])

# Map the prediction to the corresponding class label
if prediction == 0:
  print("Hate Speech")
elif prediction == 1:
  print("Offensive Language")
else:
  print("Not Offensive")
