from flask import Flask, render_template, request
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import joblib
from sklearn.feature_extraction.text import CountVectorizer 

app = Flask(__name__)

# Load the trained model and set up preprocessing functions
lr_model = joblib.load('random_forest_model.pkl')
cv = joblib.load('count_vectorizer.pkl')

def preprocess_url(url):
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    words_tokenized = tokenizer.tokenize(url)

    stemmer = SnowballStemmer("english")
    words_stemmed = [stemmer.stem(word) for word in words_tokenized]

    return ' '.join(words_stemmed)

def predict_url_status(url):
    preprocessed_url = preprocess_url(url)
    feature = cv.transform([preprocessed_url])
    prediction = lr_model.predict(feature)[0]

    return prediction

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling form submissions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        url_to_predict = request.form['url']
        print(f"Received URL: {url_to_predict}")  # Add this line for debugging
        prediction_result = predict_url_status(url_to_predict)

        if prediction_result == 'phishing':
            prediction_text = "Phishing"
            # features = "Feature1, Feature2, Feature3"  # Replace with your actual features
        else:
            prediction_text = "Legitimate"
            features = ""  # No features for legitimate sites

        print(f"Prediction Result: {prediction_text}")  # Add this line for debugging

        return render_template('result.html', prediction_text=prediction_text, url=url_to_predict)
    
if __name__ == '__main__':
    app.run(debug=True)
