import mlflow
import dagshub
import joblib
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
import pandas as pd
import io

import warnings
warnings.simplefilter("ignore")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    # Allows all origins, or specify domains like ["http://example.com"]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)


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

        if sentiment == 0:
            sentiment = -1
        elif sentiment == 1:
            sentiment = 0
        else:
            sentiment = 1

        return sentiment
    except Exception as e:
        print(f"Error occured in predicting sentiment: {e}")
        return 0


# Initialize Mlflow
mlflow_init()
# Load model and vectorizer
model, vectorizer = load_model_vectorizer(
    "yt_chrome_plugin_model", "2", "./vectorizer.pkl")


@app.get('/')
def read_root():
    return "Welcome to Youtube Sentiment Prediction API"


class Comment(BaseModel):
    comments: list


@app.post('/predict')
def predict_req(Comment: Comment):
    comments = Comment.comments
    if not comments:
        return {"error": "No comments provided"}
    predictions = [predict(comment) for comment in comments]

    response = [{"comment": comment, "sentiment": sentiment}
                for comment, sentiment in
                zip(comments, predictions)]

    return response


class CommentData(BaseModel):
    text: str
    timestamp: str
    authorId: Optional[str] = 'Unknown'


class CommentsRequest(BaseModel):
    comments: List[CommentData]


@app.post('/predict_with_timestamps')
def predict_with_timestamps(CommentRequest: CommentsRequest):
    comments_data = CommentRequest.comments
    if not comments_data:
        return {"error": "No comments provided"}
    try:
        # Accessing .text, not item['text']
        comments = [item.text for item in comments_data]
        # Accessing .timestamp
        timestamps = [item.timestamp for item in comments_data]

        predictions = [predict(comment) for comment in comments]
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
                for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return response


class SentimentCounts(BaseModel):
    sentiment_counts: dict[str, int]


@app.post('/generate_chart')
def generate_chart(SentimentCounts: SentimentCounts):
    try:
        sentiment_counts = SentimentCounts.sentiment_counts

        if not sentiment_counts:
            return {"error": "No sentiment counts provided"}

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            return {"error": "Sentiment counts sum to zero"}

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.axis('equal')

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)  # Rewind the BytesIO object to the beginning
        plt.close()

        return StreamingResponse(img_io, media_type='image/png')
    except Exception as e:
        print(f"Error in /generate_chart: {e}")
        return {"error": f"Chart generation failed: {str(e)}"}


class SentimentDataRequest(BaseModel):
    sentiment_data: List[dict]  # List of sentiment data items


@app.post('/generate_trend_graph')
def generate_trend_graph(request: SentimentDataRequest):
    try:
        sentiment_data = request.sentiment_data
        print(sentiment_data)
        if not sentiment_data:
            return {"error": "No sentiment data provided"}

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample(
            'M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return StreamingResponse(img_io, media_type='image/png')
    except Exception as e:
        print(f"Error in /generate_trend_graph: {e}")
        return {"error": f"Trend graph generation failed: {str(e)}"}


@app.post("/generate_wordcloud")
def generate_wordcloud(Comment: Comment):
    try:
        comments = Comment.comments
        if not comments:
            return {"error": "No comments provided"}

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(
            comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return StreamingResponse(img_io, media_type='image/png')
    except Exception as e:
        print(f"Error in /generate_wordcloud: {e}")
        return {"error": f"Word cloud generation failed: {str(e)}"}
