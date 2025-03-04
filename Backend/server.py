from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "<h1>Hello, Flask is running on Render!</h1>"

@app.route('/test', methods=['GET'])
def api():
    return jsonify({"message": "This is a sample API response!"})


@app.route('/query', methods=['POST'])
def final_test_api():
    return jsonify({"message": "This is a sample API response for query!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  
