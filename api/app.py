import mlflow
import dagshub
import joblib
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


def mlflow_init():
    """
    Initialize MLFlow Tracking Server
    """
    try:
        # Initialize Dagshub
        DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
        DAGSHUB_BUCKET = os.getenv("DAGSHUB_BUCKET")
        dagshub.init(repo_owner=DAGSHUB_USERNAME,
                     repo_name=DAGSHUB_BUCKET, mlflow=True)
        # Initialize mlflow tracking server
        mlflow.set_tracking_uri(
            f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_BUCKET}.mlflow")
        print("Succesfully initialized mlfow tracking server")
    except Exception as e:
        print(f"Error occured in initializing mlflow tracking server: {e}")


def load_model_vectorizer(model_name: str, model_version: str,
                          vectorizer_path: str):
    '''
    Load Model From MLFlow Registry and vectorizer
    '''
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        # model = mlflow.pyfunc.load_model(
        #     model_uri=f"models:/{model_name}/{model_version}")
        model = mlflow.sklearn.load_model(model_uri)
        print("Model Loaded Successfully")
        vectorizer = joblib.load(vectorizer_path)
        print("Vectorizer Loaded Successfully")
        return model, vectorizer
    except Exception as e:
        print(f"Error occured in loading model: {e}")


# Define the preprocessing function
def preprocess_comment(comment: str):
    """
    Preprocess the comment

    Args:
    comment: The comment to preprocess

    Returns:
    comment: The preprocessed comment
    """
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - \
            {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join(
            [word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word)
                            for word in comment.split()])
        return comment
    except Exception as e:
        print(f"Error occured in preprocessing comment: {e}")
        return comment


def predict(comment: str):
    """
    Predict sentiment of the comment

    Args:
    comment: The comment to predict sentiment

    Returns:
    sentiment: The sentiment of the comment
    """
    try:
        # Preprocess the comment
        comment = preprocess_comment(comment)

        # Vectorize the comment
        comment_vector = vectorizer.transform([comment])

        # Predict the sentiment
        sentiment = model.predict(comment_vector)
        confidence = model.predict_proba(comment_vector).max()

        if sentiment == 0:
            sentiment = "Negative"
        elif sentiment == 1:
            sentiment = "Neutral"
        else:
            sentiment = "Positive"

        return sentiment, confidence
    except Exception as e:
        print(f"Error occured in predicting sentiment: {e}")
        return "Neutral", 0.5


# Initialize Mlflow
mlflow_init()
# Load model and vectorizer
model, vectorizer = load_model_vectorizer(
    "yt_chrome_plugin_model", "2", "./vectorizer.pkl")


@app.get('/')
def read_root():
    return "Welcome to Youtube Sentiment Prediction API"


class Comment(BaseModel):
    comment: str


class Sentiment(BaseModel):
    sentiment: str
    confidence: float


@app.post('/predict')
def predict_req(Comment: Comment):
    sentiment, confidence = predict(Comment.comment)
    return {"sentiment": sentiment, "confidence": confidence}
