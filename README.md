# Sentiment Analyzer Flask App

## Description
This Flask application performs sentiment analysis on textual data. It uses a machine learning pipeline that includes preprocessing with stemming, feature extraction with vectorization, and classification using a Support Vector Classifier (SVC). The application is designed to classify sentiments as positive, negative, or neutral.

## Features
- **Stemming**: Processes text data to its root form.
- **NLTK**: Utilizes the Natural Language Toolkit for Python to handle text processing.
- **Vectorization**: Transforms text into numerical vectors using `vectorizer_model.sav`.
- **SVC Model**: Employs a pre-trained Support Vector Classifier stored in `model.sav` for sentiment classification.

## Installation
To run this application, follow these steps:

```bash
# Clone the repository
git clone https://github.com/khanyousufminhaj/Sentiment_analyzer.git
cd Sentiment_analyzer

# Install dependencies
pip install -r requirements.txt
```

Data
The sentiment analysis model was trained on the train.csv dataset, which contains labeled sentiment data.

Usage
To start the Flask app, run:
```bash
flask run
```

This will start a web server that you can access at http://localhost:5000 to interact with the sentiment analyzer.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.
