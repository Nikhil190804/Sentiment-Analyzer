# 📺 YouTube Comment Analyzer & Chatbot 🤖

This project analyzes YouTube comments using AI to provide insights and enables chat-based interaction with the comments. It leverages several state-of-the-art NLP models to understand sentiment, detect offensive language, identify emotions, and generate summaries, as well as allowing users to "talk" to the comment section.

## ✨ Features

*   🔗 **YouTube Video URL Input:** Accepts standard and shortened YouTube video links (e.g., `youtube.com/watch?v=`, `youtu.be/`, `youtube.com/shorts/`).
*   💬 **Comment Fetching:** Retrieves all comments from the provided YouTube video.
*   😊 **Sentiment Analysis:**
    *   Model: `Twitter-roBERTa-base for Sentiment Analysis - UPDATED (2022)`.
    *   Categorizes comments into positive, negative, or neutral.
    *   Displays the top 5 comments for each sentiment category.
*   🤬 **Offensive Language Detection:**
    *   Model: `Twitter-roBERTa-base for Offensive Language Identification`.
    *   Identifies comments containing offensive language.
*   ემო **Emotion/Emoji Analysis:**
    *   Model: `twitter-roberta-base-emotion-multilabel-latest`.
    *   Assigns emojis to comments based on detected emotions.
    *   Identifies the dominant emoji for the video.
*   📊 **Visualizations:** Interactive pie charts for sentiment, offensive language, and emoji distributions.
*   📝 **AI-Generated Insights:**
    *   LLM: `meta-llama/Llama-3.3-70B-Instruct` (via Langchain).
    *   Generates a summary of key takeaways and insights from the analyzed comment data.
*   🤖 **Comment Chatbot:**
    *   Allows users to "talk" to the YouTube comments.
    *   Asks questions and receives answers based on the video's comment section.

## ⚙️ How It Works / Technical Details

### Comment Processing Pipeline:
1.  User provides a YouTube video URL.
2.  The backend (Flask) fetches all comments using the YouTube Data API v3.
3.  Comments are processed by three `tweetnlp` models for sentiment, offensive language, and emoji tagging.

### LLM-Powered Insights Generation:
1.  Key data points (top comments per sentiment, distribution percentages of sentiment, offensive language, and emojis) are fed to the `meta-llama/Llama-3.3-70B-Instruct` LLM via Langchain.
2.  The LLM generates a human-readable summary of insights and key takeaways from this data.

### Interactive Comment Chatbot:
1.  **Embedding Generation:** All fetched comments are converted into numerical representations (embeddings) using the `all-MiniLM-L6-v2` model from SentenceTransformer.
2.  **Comment Clustering:** The KMeans algorithm clusters the comment embeddings. The number of clusters `x` is dynamically determined by `min(20, max(2, int(math.sqrt(len(ALL_COMMENTS)))))`. This ensures a selection of diverse yet concise representative comments from the entire comment section.
3.  **LLM Interaction:** The `meta-llama/Llama-3.3-70B-Instruct` LLM (via Langchain) is provided with these `x` representative comments (cluster centers) as context. Users can then ask questions, and the LLM answers based *only* on this curated comment data. This approach significantly reduces the token count for the LLM prompt, making it more efficient and focused.

### Deep Learning Models Used:
1.  `tweetnlp/Twitter-roBERTa-base for Sentiment Analysis - UPDATED (2022)`: For classifying comment sentiment.
2.  `tweetnlp/Twitter-roBERTa-base for Offensive Language Identification`: For detecting offensive content.
3.  `tweetnlp/twitter-roberta-base-emotion-multilabel-latest`: For identifying emotions and assigning corresponding emojis.
4.  `sentence-transformers/all-MiniLM-L6-v2`: For generating dense vector embeddings of comments.
5.  `meta-llama/Llama-3.3-70B-Instruct`: For generating insights and powering the chatbot.

## 🚀 Key Technical Highlights

*   Engineered a full-stack GenAI system for YouTube comment analysis using 5 transformer models (3 NLP classifiers, SentenceTransformer, LLaMA-3.3-70B) for sentiment, offensiveness, and emoji-based emotion tagging on large-scale comment data.
*   Optimized LLM input by clustering √n comments (where n is the total number of comments) via KMeans on dense SentenceTransformer embeddings. This reduces the prompt size by approximately 90% while preserving semantic coverage for downstream tasks like insight generation and chatbot interaction.
*   Implemented a LangChain pipeline with LLaMA-3.3-70B to generate natural language insights and enable a chatbot, letting users interactively query comment data via an LLM-driven dialogue system grounded in the clustered comment representations.

## 🛠️ Technologies Used

*   **Frontend:** React.js
*   **Backend:** Flask (Python)
*   **NLP Models:**
    *   `tweetnlp`
        *   `Twitter-roBERTa-base for Sentiment Analysis - UPDATED (2022)`
        *   `Twitter-roBERTa-base for Offensive Language Identification`
        *   `twitter-roberta-base-emotion-multilabel-latest`
    *   `SentenceTransformers` (`all-MiniLM-L6-v2`)
    *   HuggingFace Transformers (`meta-llama/Llama-3.3-70B-Instruct`)
*   **Core Libraries:**
    *   `Langchain`
    *   `scikit-learn` (KMeans)
    *   `Flask-CORS`
    *   `python-dotenv`
    *   `requests`
    *   `wordcloud`
    *   `matplotlib`
*   **Development Tools:**
    *   `vite`
    *   `eslint`

## 📋 Setup and Installation

### Prerequisites:
*   Python 3.8+
*   Node.js and npm/yarn
*   Access to YouTube Data API v3 (API Key)
*   Hugging Face API Token (for Llama 3.3)

### Backend (`Backend/` directory):
1.  Navigate to the `Backend` directory: `cd Backend`
2.  Create a virtual environment: `python -m venv venv`
3.  Activate it:
    *   Windows: `venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`
4.  Install dependencies: `pip install -r requirements.txt`
5.  Create a `.env` file. You can copy `.env.example` if it exists, or create one manually.
6.  Add your API keys to the `.env` file:
    ```env
    YOUTUBE_API="YOUR_YOUTUBE_API_KEY"
    HUGGINGFACEHUB_API_TOKEN="YOUR_HUGGINGFACE_API_TOKEN"
    BASE_URL="https://www.googleapis.com/youtube/v3/commentThreads"
    ```

### Frontend (`Frontend/` directory):
1.  Navigate to the `Frontend` directory: `cd Frontend`
2.  Install dependencies: `npm install` (or `yarn install`)

### Running the Application:
1.  **Start the Backend server:** In the `Backend` directory, activate your virtual environment and run:
    ```bash
    python server.py
    ```
2.  **Start the Frontend development server:** In the `Frontend` directory, run:
    ```bash
    npm run dev
    ```
    (or `yarn dev`)
3.  Open your browser and go to `http://localhost:5173` (or the port specified by Vite).

## ▶️ Usage

1.  Open the web application in your browser.
2.  Find a YouTube video you want to analyze.
3.  Copy the video's URL. Standard links (`https://www.youtube.com/watch?v=...`), shortened links (`https://youtu.be/...`), and shorts links (`https://www.youtube.com/shorts/...`) are supported.
4.  Paste the URL into the input field on the web page and click the "Analyze" button.
5.  Wait for the analysis to complete. You will see loaders and progress updates. This might take a few moments depending on the number of comments.
6.  Explore the results:
    *   View the interactive pie charts showing distributions for sentiment, offensive language, and emojis.
    *   Read the top 5 comments for positive, negative, and neutral sentiment categories.
    *   See the dominant emoji identified from the comments.
    *   Read the AI-generated insights that summarize the key discussion points in the comments.
7.  Interact with the Chatbot:
    *   Type your questions about the video's comments into the chat input field (e.g., "What are people saying about the new feature?", "Are there any funny comments?", "What is the general opinion on the topic?").
    *   The chatbot will provide answers based on the representative comments extracted and processed from the video.

## 🎬 Video Demo

**(Coming Soon!)** A video demonstration will be added here to showcase the project in action.
