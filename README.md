🧠 Multimodal Sentiment Analysis of Text using Machine Learning
📌 Project Overview
This project performs Sentiment Analysis on textual data by classifying user opinions into Positive, Negative, or Neutral sentiments. It is designed to process social media posts, reviews, and comments, including Hinglish (Hindi-English mixed language) text by translating it into English before analysis.

The project uses Natural Language Processing (NLP) techniques for preprocessing and Machine Learning algorithms for sentiment classification.

🚀 Features
📂 Load sentiment dataset from CSV
🌐 Translate Hinglish text to English
🧹 Text preprocessing and cleaning
🔤 Tokenization and stopwords removal
📊 TF-IDF feature extraction
🤖 Train multiple Machine Learning models
📈 Compare model performance
✅ Predict Positive, Negative, or Neutral sentiment
📉 Display evaluation metrics
🏗️ Project Workflow
Dataset
   │
   ▼
Load CSV File
   │
   ▼
Data Cleaning
(Remove URLs, Mentions, Punctuation, Special Characters)
   │
   ▼
Translate Hinglish → English
   │
   ▼
Text Preprocessing
(Tokenization, Lowercasing, Stopword Removal)
   │
   ▼
TF-IDF Feature Extraction
   │
   ▼
Train Machine Learning Models
   │
   ▼
Model Evaluation
   │
   ▼
Sentiment Prediction
📁 Dataset
The dataset contains textual reviews or tweets along with their sentiment labels.

Example:

Text	Sentiment
I love this movie	Positive
Worst product ever	Negative
It's okay	Neutral
⚙️ Technologies Used
Python
Pandas
NumPy
NLTK
TextBlob
Scikit-learn
Matplotlib
Seaborn
Google Translator API (for Hinglish translation)
📚 Machine Learning Models
The following algorithms were implemented and compared:

Logistic Regression
Multinomial Naive Bayes
Bernoulli Naive Bayes
Linear Support Vector Machine (Linear SVM)
SGD Classifier
🧹 Text Preprocessing
The preprocessing pipeline includes:

Convert text to lowercase
Remove URLs
Remove mentions (@username)
Remove punctuation
Remove numbers
Remove stopwords
Tokenization
Lemmatization/Stemming
Translate Hinglish to English
📊 Feature Extraction
The project uses TF-IDF (Term Frequency–Inverse Document Frequency) to convert textual data into numerical vectors that can be processed by Machine Learning models.

📈 Model Evaluation
The trained models are evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
💻 Installation
Clone the repository

git clone https://github.com/yourusername/multimodal-sentiment-analysis.git
Move into the project folder

cd multimodal-sentiment-analysis
Install dependencies

pip install -r requirements.txt
Run the project

python main.py
📂 Project Structure
Multimodal-Sentiment-Analysis/
│
├── dataset/
│   └── sentiment_dataset.csv
│
├── notebooks/
│   └── SentimentAnalysis.ipynb
│
├── models/
│
├── images/
│
├── requirements.txt
│
├── main.py
│
├── README.md
│
└── LICENSE
📌 Sample Input
Ye movie bahut acchi hai.
Translation
This movie is very good.
Prediction
Positive
📈 Future Improvements
Deep Learning (LSTM/Bi-LSTM)
BERT-based sentiment analysis
Real-time Twitter sentiment analysis
Flask/FastAPI web deployment
Multilingual sentiment analysis
Emotion detection (Happy, Angry, Sad, etc.)
✅ Advantages
Supports Hinglish text
Fast sentiment prediction
Easy to train and deploy
Multiple Machine Learning model comparison
Effective preprocessing pipeline
⚠️ Limitations
Translation errors may affect accuracy.
Sarcasm and irony are difficult to detect.
TF-IDF does not capture contextual meaning as effectively as transformer-based models.
Performance depends on dataset quality.
👨‍💻 Author
Sahil Sharma

B.Tech Computer Science Engineering

⭐ Acknowledgements
Scikit-learn
NLTK
TextBlob
Pandas
NumPy
Matplotlib
Seaborn
📜 License
This project is developed for educational and research purposes.

🌐 Live Demo Link : https://qkggmswlwwrt5whqcysckg.streamlit.app/
