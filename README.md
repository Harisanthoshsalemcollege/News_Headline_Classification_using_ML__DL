# News Headline Classification Using Machine Learning and Deep Learning

## Domain
* Natural Language Processing (NLP)
* Machine Learning
* Deep Learning
* Text Classification

## Introduction
This project classifies short news headlines into four categories:
* World
* Sports
* Business
* Sci/Tech
Multiple machine learning and deep learning models were built and compared.
The final model is deployed using a Streamlit web application supporting both single and bulk predictions.

## Skills Takeaway From This Project
* NLP Text Cleaning
* Tokenization, Stopword Removal, Lemmatization
* TF-IDF Vectorization
* Machine Learning Model Building
* Deep Learning Model Building
* Model Evaluation
* Hyperparameter Tuning
* Streamlit Deployment
* Pickle-based Model Saving

## Technologies Used
* Programming
* Python
* Data Handling
* Pandas
* NumPy
* Machine Learning
* Scikit-Learn
* Random Forest Classifier
* Multinomial Naive Bayes
* Boosting
* XGBoost
* XGBoost Classifier
* Deep Learning
* TensorFlow / Keras
* LSTM
* SimpleRNN
* NLP Processing
* NLTK
* Tokenization
* Stopwords
* Lemmatization
* Visualization
* Matplotlib
* Seaborn
* Deployment & Saving
* Streamlit
* Pickle

## Project Workflow
1. Data Cleaning and Preprocessing
* Lowercasing text
* Removing punctuation, URLs, numbers, emojis
* Tokenizing words
* Removing stopwords
* Applying lemmatization
* Transforming text using TF-IDF
  
2. Exploratory Data Analysis (EDA)
* Word Cloud Visualization
* Count Plot for category distribution
* Unique sample headlines displayed

## Models Implemented
* Random Forest Classifier
* Multinomial Naive Bayes
* XGBoost Classifier
* LSTM Deep Learning Model
* SimpleRNN Recurrent Neural Network

## Model Evaluation Metrics
* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## Model Comparison Summary

* Random Forest Classifier – Good performance

* Multinomial Naive Bayes – Best overall accuracy

* XGBoost Classifier – Strong boosting model

* LSTM – Moderate sequence learning

* SimpleRNN – Moderate performance

## Best Model Selected:
* Tuned Multinomial Naive Bayes

## Hyperparameter Tuning
RandomizedSearchCV was used to tune Multinomial Naive Bayes parameters such as:
* Alpha
* Fit_prior
This improved model accuracy significantly.

## Streamlit Application
Key Features:
* Single headline prediction
* CSV file batch prediction
* Downloadable result file
* Easy-to-use interface
  
## Backend Operation
* Loads nb_model.pkl and tfidf.pkl
* Preprocesses new text input
* Predicts the news category

## Final Outcome
* Built a complete NLP classification pipeline
* Trained and compared five ML and DL models
* Identified the best-performing tuned Naive Bayes classifier
* Developed and deployed a Streamlit application
* Enabled real-time single and batch predictions
