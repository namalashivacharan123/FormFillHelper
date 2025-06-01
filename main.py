from flask import Flask, request, jsonify
import csv
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

qa_csv = 'qa.csv'
questions = []
answers = []

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_and_embed_data():
    global questions, answers, question_embeddings
    questions = []
    answers = []
    try:
        with open(qa_csv, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                if len(row) >= 2:
                    questions.append(row[0])
                    answers.append(row[1])
        # Compute embeddings for all questions
        question_embeddings = model.encode(questions, convert_to_tensor=True)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        questions = []
        answers = []
        question_embeddings = None

load_and_embed_data()


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Please provide a 'question' field in JSON."}), 400

    user_question = data['question'].strip()
    user_embedding = model.encode(user_question, convert_to_tensor=True)

    # Compute cosine similarities
    cos_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]

    # Get top 3 matches
    top_results = torch.topk(cos_scores, k=3)

    matched_answers = []
    for idx in top_results.indices:
        matched_answers.append(answers[idx])

    return jsonify(matched_answers)

if __name__ == '__main__':
    app.run(debug=True)