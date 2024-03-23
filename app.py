from flask import Flask, request, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)

# Load your trained model
model = pickle.load(open('model.sav', 'rb'))
vectorizer=pickle.load(open('vectorizer_model.sav', 'rb'))
def preprocess(content):
    #stemming
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    #vectorizing
    X = vectorizer.transform([stemmed_content])
    return X
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        text=request.form['text']
        transformed_text=preprocess(text)
        
        # Make prediction
        prediction = model.predict(transformed_text)
        
        # Output prediction
        return render_template('index.html', prediction_text='Predicted Sentiment: {}'.format(prediction[0]))
    except Exception as e:
        return render_template('index.html', prediction_text='An error occurred: {}'.format(e))
if __name__ == "__main__":
    app.run(debug=True)