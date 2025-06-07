from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from dotenv import load_dotenv
import os
import requests
import tweetnlp
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from collections import defaultdict

app = Flask(__name__)
CORS(app)

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API")
BASE_URL = os.getenv("BASE_URL")

model=tweetnlp.load_model('sentiment')

def extract_video_id_from_url(url):
    match = re.search(r"(?:v=|\/|youtu\.be\/|embed\/|shorts\/)([a-zA-Z0-9_-]{11})", url)
    if match !=None:
        return match.group(1)
    else:
        return None


def get_comments(video_id):
    ALL_COMMENTS = []
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": API_KEY,
        "maxResults": 100,
        "textFormat": "plainText"
    }

    while True:
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        for item in data.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            ALL_COMMENTS.append(comment)

        if "nextPageToken" in data:
            params["pageToken"] = data["nextPageToken"]
        else:
            break

    return ALL_COMMENTS


def get_sentiment(ALL_COMMENTS):
    data={}
    result=model.sentiment(ALL_COMMENTS, return_probability=True)
    for output in range(len(result)):
        data[output]=result[output]
    return data


def get_wordcloud(ALL_COMMENTS):
    combined_comments = " ".join(ALL_COMMENTS)
    wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=STOPWORDS,
            max_words=100,
            colormap='plasma'
        ).generate(combined_comments)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    
def get_things(ALL_COMMENTS,ALL_SENTIMENT_DATA):
       
    sentiment_count = {'positive': 0, 'neutral': 0, 'negative': 0}
    sum_probabilities = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}

    
    comment_by_sentiment = defaultdict(list)

    for i, result in ALL_SENTIMENT_DATA.items():
        label = result['label']
        sentiment_count[label] += 1

       
        for s in ['positive', 'neutral', 'negative']:
            sum_probabilities[s] += result['probability'][s]

        
        confidence = result['probability'][label]
        comment_by_sentiment[label].append((ALL_COMMENTS[i], confidence))

    total = len(ALL_COMMENTS)

    
    print("Sentiment Distribution:")
    for sentiment, count in sentiment_count.items():
        print(f"{sentiment.capitalize()}: {count} ({(count/total)*100:.2f}%)")

    
    print("\nAverage Sentiment Probabilities:")
    for sentiment, total_prob in sum_probabilities.items():
        print(f"{sentiment.capitalize()}: {total_prob/total:.2f}")

    
    print("\nTop 5 High-Confidence Comments per Sentiment:")
    for sentiment, comment_tuples in comment_by_sentiment.items():
        print(f"\n{sentiment.capitalize()} comments:")
        sorted_comments = sorted(comment_tuples, key=lambda x: x[1], reverse=True)
        for comment, confidence in sorted_comments[:5]:
            print(f"- ({confidence:.2f}) {comment}")

    
    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_count.values(), labels=sentiment_count.keys(), autopct='%1.1f%%', startangle=140, colors=['green', 'grey', 'red'])
    plt.title('Sentiment Distribution')
    plt.show()

@app.route('/')
def home():
    return "<h1>Hello, Flask is running on Render!</h1>"

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "This is a sample API response!"})


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()  
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400
    video_url = data.get("yt_url")
    video_id = extract_video_id_from_url(video_url)
    if(video_id==None):
        return jsonify({"error": "Invalid URL"}), 400
    
    ALL_COMMENTS = get_comments(video_id)
    ALL_SENTIMENT_DATA=get_sentiment(ALL_COMMENTS)

    for i in range(5):
        print(ALL_SENTIMENT_DATA[i])

    get_wordcloud(ALL_COMMENTS)

    get_things(ALL_COMMENTS,ALL_SENTIMENT_DATA)



    return jsonify({"message": f"This is a sample API response for query! {video_id}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  
