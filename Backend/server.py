from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>Hello, Flask is running on Render!</h1>"

@app.route('/test', methods=['GET'])
def api():
    return jsonify({"message": "This is a sample API response!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  
