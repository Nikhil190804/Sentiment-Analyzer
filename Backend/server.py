from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from dotenv import load_dotenv
import os
import requests
import tweetnlp
from wordcloud import WordCloud,STOPWORDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict,Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import math
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
import io
import base64

plt.rcParams['font.family'] = 'Segoe UI Emoji'
app = Flask(__name__)
CORS(app)
load_dotenv(override=True)
API_KEY = os.getenv("YOUTUBE_API")
BASE_URL = os.getenv("BASE_URL")

SENTIMENT_MODEL=tweetnlp.load_model('sentiment')
OFFENSIVE_MODEL = tweetnlp.load_model('offensive')
EMOJI_MODEL = tweetnlp.load_model('emoji')

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct"
)
chat_model = ChatHuggingFace(llm=llm)
CHAT_HISTORY=[]

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
    result=SENTIMENT_MODEL.sentiment(ALL_COMMENTS, return_probability=True)
    for output in range(len(result)):
        data[output]=result[output]
    return data


def get_offensive_language_detection(ALL_COMMENTS):
    data={}
    result=OFFENSIVE_MODEL.offensive(ALL_COMMENTS, return_probability=True)
    for output in range(len(result)):
        data[output]=result[output]
    return data


def get_emoji(ALL_COMMENTS):
    data={}
    result=EMOJI_MODEL.emoji(ALL_COMMENTS)
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
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='PNG')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return img_base64


def get_sentiment_analysis(ALL_COMMENTS, ALL_SENTIMENT_DATA):
    sentiment_count = {'positive': 0, 'neutral': 0, 'negative': 0}
    sum_probabilities = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
    comment_by_sentiment = defaultdict(list)

    output_lines = []

    for i, result in ALL_SENTIMENT_DATA.items():
        label = result['label']
        sentiment_count[label] += 1
        for s in ['positive', 'neutral', 'negative']:
            sum_probabilities[s] += result['probability'][s]
        confidence = result['probability'][label]
        comment_by_sentiment[label].append((ALL_COMMENTS[i], confidence))

    total = len(ALL_COMMENTS)

    output_lines.append("Sentiment Distribution:")
    for sentiment, count in sentiment_count.items():
        output_lines.append(f"{sentiment.capitalize()}: {count} ({(count/total)*100:.2f}%)")

    output_lines.append("\nAverage Sentiment Probabilities:")
    for sentiment, total_prob in sum_probabilities.items():
        output_lines.append(f"{sentiment.capitalize()}: {total_prob/total:.2f}")

    output_lines.append("\nTop 5 High-Confidence Comments per Sentiment:")
    top_5_by_sentiment = {}
    for sentiment, comment_tuples in comment_by_sentiment.items():
        output_lines.append(f"\n{sentiment.capitalize()} comments:")
        sorted_comments = sorted(comment_tuples, key=lambda x: x[1], reverse=True)
        top_comments = []
        for comment, confidence in sorted_comments[:5]:
            output_lines.append(f"- ({confidence:.2f}) {comment}")
            top_comments.append(comment)
        top_5_by_sentiment[sentiment] = top_comments

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sentiment_count.values(),
        labels=sentiment_count.keys(),
        autopct='%1.1f%%',
        startangle=140,
        colors=['green', 'grey', 'red']
    )
    ax.set_title('Sentiment Distribution')

    buf = io.BytesIO()
    plt.savefig(buf, format='PNG')
    plt.close(fig)
    buf.seek(0)
    pie_chart_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return {
        "print_output": "\n".join(output_lines),
        "top_5_comments": top_5_by_sentiment,
        "pie_chart": pie_chart_base64
    }


def get_offensive_language_analysis(ALL_COMMENTS, ALL_OFFENSIVE_DATA):
    total = len(ALL_OFFENSIVE_DATA)

    offensive_comments = {i: v for i, v in ALL_OFFENSIVE_DATA.items() if v['label'] == 'offensive'}
    non_offensive_comments = {i: v for i, v in ALL_OFFENSIVE_DATA.items() if v['label'] == 'non-offensive'}

    offensive_percent = (len(offensive_comments) / total) * 100
    non_offensive_percent = 100 - offensive_percent

    top_offensive = sorted(
        [(idx, data['probability']['offensive'], ALL_COMMENTS[idx])
         for idx, data in offensive_comments.items()],
        key=lambda x: x[1], reverse=True
    )[:5]

    
    output_lines = []
    output_lines.append(f"Total comments analyzed: {total}")
    output_lines.append(f"Offensive comments: {len(offensive_comments)} ({offensive_percent:.2f}%)")
    output_lines.append(f"Non-Offensive comments: {len(non_offensive_comments)} ({non_offensive_percent:.2f}%)\n")

    output_lines.append("Top 5 Offensive Comments:")
    top_comments_cleaned = []
    for idx, prob, text in top_offensive:
        output_lines.append(f"{text}  -->  {prob:.2%} offensive")
        top_comments_cleaned.append({
            "comment": text,
            "confidence": round(prob, 4)
        })

    fig, ax = plt.subplots(figsize=(7, 7))
    labels = ['Offensive', 'Non-Offensive']
    sizes = [offensive_percent, non_offensive_percent]
    colors = ['#f39c12', '#3498db']
    explode = (0.1, 0)

    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        shadow=True,
        startangle=140,
        textprops={'fontsize': 14}
    )
    ax.set_title('Offensive Language Detection Results', fontsize=16)
    ax.axis('equal')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='PNG')
    plt.close(fig)
    buf.seek(0)
    pie_chart_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return {
        "print_output": "\n".join(output_lines),
        "top_5_offensive_comments": top_comments_cleaned,
        "pie_chart": pie_chart_base64
    }


def get_emoji_analysis(ALL_COMMENTS, ALL_EMOJI_DATA):
    emojis = [v['label'] for v in ALL_EMOJI_DATA.values()]
    total = len(emojis)
    emoji_counts = Counter(emojis)
    sorted_emojis = emoji_counts.most_common()

    output_lines = []
    output_lines.append(f"Total comments analyzed: {total}")
    output_lines.append("\nTop 5 Most Common Emojis:")
    
    top_5_emojis = []
    for emoji, count in sorted_emojis[:5]:
        percentage = (count / total) * 100
        output_lines.append(f"{emoji}  →  {count} comments ({percentage:.2f}%)")
        top_5_emojis.append({
            "emoji": emoji,
            "count": count,
            "percentage": round(percentage, 2)
        })

    labels = [f"{emoji}" for emoji, _ in sorted_emojis[:8]]
    sizes = [count for _, count in sorted_emojis[:8]]
    explode = [0.1 if i == 0 else 0 for i in range(len(labels))]
    colors = plt.cm.tab20.colors[:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        sizes,
        labels=labels,
        explode=explode,
        autopct='%1.1f%%',
        startangle=140,
        shadow=True,
        colors=colors,
        textprops={'fontsize': 14}
    )
    ax.set_title("Emoji Sentiment Representation from Comments", fontsize=16)
    ax.axis('equal')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='PNG')
    plt.close(fig)
    buf.seek(0)
    pie_chart_base64 = base64.b64encode(buf.read()).decode('utf-8')

    dominant_emoji = sorted_emojis[0][0] if sorted_emojis else None
    output_lines.append(f"\n🎯 Dominant Emoji for This Video: {dominant_emoji}")

    return {
        "print_output": "\n".join(output_lines),
        "top_5_emojis": top_5_emojis,
        "dominant_emoji": dominant_emoji,
        "pie_chart": pie_chart_base64
    }


def get_representative_comments(ALL_COMMENTS, num_clusters=None):
    if not ALL_COMMENTS or len(ALL_COMMENTS) < 2:
        return ALL_COMMENTS

    if num_clusters is None:
        num_clusters = min(20, max(2, int(math.sqrt(len(ALL_COMMENTS)))))

    if len(ALL_COMMENTS) < num_clusters:
        return ALL_COMMENTS

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(ALL_COMMENTS)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    representative_comments = []
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_id]

        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]

        representative_comments.append(ALL_COMMENTS[closest_idx])

    return representative_comments


def comments_summarizer_by_llm(COMMENTS_DATA):

    QUERY = ""

    for sentiment, comments in COMMENTS_DATA.items():
        comments_text = "\n".join(f"- {comment}" for comment in comments)
        QUERY += (
            f"\nSentiment: {sentiment.capitalize()}\n"
            f"Comments:\n{comments_text}\n"
        )

    PROMPT_TEMPLATE = (
        "You are an expert insights generator.\n"
        "Your task is to analyze viewer feedback based on their sentiment. "
        "You will be given three sets of comments: Positive, Negative, and Neutral.\n"
        "For each sentiment group:\n"
        "- Summarize the general sentiment in a short, engaging paragraph.\n"
        "- Provide two key takeaways or observations based on the comments.\n\n"
        "Finally, provide 1-2 overall insights that reflect the broader themes across all comments.\n\n"
        f"{QUERY}\n\n"
        "Format:\n"
        "## Positive Insights\n"
        "[Paragraph]\n"
        "- Key Takeaway 1\n"
        "- Key Takeaway 2\n\n"
        "## Negative Insights\n"
        "[Paragraph]\n"
        "- Key Takeaway 1\n"
        "- Key Takeaway 2\n\n"
        "## Neutral Insights\n"
        "[Paragraph]\n"
        "- Key Takeaway 1\n"
        "- Key Takeaway 2\n\n"
        "## Overall Insights\n"
        "[What viewers are saying in general]"
    )

    messages = [
        SystemMessage(content="You are a helpful AI."),
        HumanMessage(content=PROMPT_TEMPLATE)
    ]

    response = chat_model.invoke(messages)
    return response.content


@app.route('/')
def home():
    return "<h1>Hello, Flask is running on Render!</h1>"

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "This is a sample API response!"})


@app.route('/question',methods=['POST'])
def chatbot():
    data = request.get_json()  
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400
    question = data.get("question")
    CHAT_HISTORY.append(HumanMessage(content=question))
    response = chat_model.invoke(CHAT_HISTORY).content
    CHAT_HISTORY.append(AIMessage(content=response))
    return jsonify({"answer": response}), 200


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
    REPRESENTATIVE_COMMENTS=get_representative_comments(ALL_COMMENTS)
    REPRESENTATIVE_COMMENTS_ALL = ""
    for comment in REPRESENTATIVE_COMMENTS:
        REPRESENTATIVE_COMMENTS_ALL+="-  "+comment+"\n"

    ALL_SENTIMENT_DATA=get_sentiment(ALL_COMMENTS)
    ALL_OFFENSIVE_SPEECH_DATA = get_offensive_language_detection(ALL_COMMENTS)
    ALL_EMOJI_DETECTION_DATA=get_emoji(ALL_COMMENTS)

    WORDCLOUD_IMAGE=get_wordcloud(ALL_COMMENTS)

    SENTIMENT_ANALYSIS_DATA=get_sentiment_analysis(ALL_COMMENTS,ALL_SENTIMENT_DATA)
    TOP_5_SENTIMENT_DATA=SENTIMENT_ANALYSIS_DATA["top_5_comments"]
    OFFENSIVE_LANGUAGE_ANALYSIS=get_offensive_language_analysis(ALL_COMMENTS,ALL_OFFENSIVE_SPEECH_DATA)
    EMOJI_ANALYSIS_DATA=get_emoji_analysis(ALL_COMMENTS,ALL_EMOJI_DETECTION_DATA)

    LLM_OUTPUT=comments_summarizer_by_llm(TOP_5_SENTIMENT_DATA)

    system_message_new = SystemMessage(
        content= f"""
        You are a helpful and grounded assistant. You have been given a list of representative viewer comments from a video. This means the list captures the most relevant and commonly discussed topics, but does not contain every comment.

        You must answer user questions strictly based on the provided representative comments. Do not rely on outside knowledge, assumptions, or hallucinations.

        If the answer to the user's question cannot be reasonably inferred from the given comments, clearly respond with something like:
        - "There is no mention of that in the comments."
        - "I couldn't find any information related to that topic."
        - "The provided comments do not cover that subject."

        Before answering, you should carefully review all representative comments to identify patterns or relevant details. If multiple perspectives appear, mention both. Always keep your answers grounded and evidence-based.

        Representative Comments:
        {REPRESENTATIVE_COMMENTS_ALL}

        """
    )
    CHAT_HISTORY.append(system_message_new)

    RESPONSE_DICT = {}
    RESPONSE_DICT["word_cloud"]=WORDCLOUD_IMAGE
    RESPONSE_DICT["sentiment_output"]=SENTIMENT_ANALYSIS_DATA["print_output"]
    RESPONSE_DICT["sentiment_graph"]=SENTIMENT_ANALYSIS_DATA["pie_chart"]
    RESPONSE_DICT["offensive_language_output"]=OFFENSIVE_LANGUAGE_ANALYSIS["print_output"]
    RESPONSE_DICT["offensive_language_graph"]=OFFENSIVE_LANGUAGE_ANALYSIS["pie_chart"]
    RESPONSE_DICT["emoji_output"]=EMOJI_ANALYSIS_DATA["print_output"]
    RESPONSE_DICT["emoji_graph"]=EMOJI_ANALYSIS_DATA["pie_chart"]
    RESPONSE_DICT["llm_output"]=LLM_OUTPUT


    #for k,v in RESPONSE_DICT.items():
    #    print(k,"-----")
    #    print(v)
    #    print("\n\n")

    return jsonify(RESPONSE_DICT)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  
