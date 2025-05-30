from flask import Flask, request, jsonify
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)


def extract_video_id_from_url(url):
    match = re.search(r"(?:v=|\/|youtu\.be\/|embed\/|shorts\/)([a-zA-Z0-9_-]{11})", url)
    if match !=None:
        return match.group(1)
    else:
        return None


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
    return jsonify({"message": f"This is a sample API response for query! {video_id}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  
