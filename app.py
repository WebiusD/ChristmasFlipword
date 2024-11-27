from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv
import os
import numpy as np

import openai
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity 

load_dotenv()

app = Flask(__name__)

# Dummy in-memory storage
players = []
target_word = ""
# Path to the file storing used words
USED_WORDS_FILE = 'used_words.txt'

# Function to load used words from the file
def load_used_words():
    if os.path.exists(USED_WORDS_FILE):
        with open(USED_WORDS_FILE, 'r') as f:
            return set(f.read().splitlines())
    return set()

# Function to save used words to the file
def save_used_words():
    with open(USED_WORDS_FILE, 'w') as f:
        for word in used_words:
            f.write(f"{word}\n")

# To store previously generated words
used_words = load_used_words()

# OpenAI API configuration
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_new_target_word():
    global used_words
    exclusion_list = ", ".join(used_words)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Du generierst einzelnen Wörter für ein Spiel."},
            {"role": "user", "content": f"""Gib mir ein einzelnes Wort, dass mit Weihnachten zu tun hat. 
             Die folgenden Wörter wurden bereits verwendet und dürfen nicht wiederholt werden: {exclusion_list}."""}
        ],
        temperature=0.7,  # Creativity level
        max_tokens=10  # Enough for a single word
    )
    new_word = response.choices[0].message.content.strip()
    used_words.add(new_word)
    save_used_words()

    return new_word

def compute_scores(sim_word_emb, opp_word_emb, tar_word_emb):
    similarity_score = cosine_similarity([tar_word_emb], [sim_word_emb])[0][0]
    opposite_score = cosine_similarity([tar_word_emb], [opp_word_emb])[0][0]

    return similarity_score * 100.0, opposite_score * 100.0

# You may want to implement the functions `get_optimal_similar_word` and `get_optimal_opposite_word`
def get_optimal_similar_word():
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Du durchsuchst den Raum der Wort Embeddings nach einem Wort, dass einem bestimmten Begriff möglichst ähnlich ist."},
            {"role": "user", "content": f"""Welches Wort liegt im Embedding Raum dem Begriff '{target_word} am nächsten? Gib nur das Ergebniswort an."""}
        ],
        temperature=0.7,  # Creativity level
        max_tokens=10  # Enough for a single word
    )
    return response.choices[0].message.content.strip()

def get_optimal_opposite_word():
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Du durchsuchst den Raum der Wort Embeddings nach einem Wort, dass einem bestimmten Begriff möglichst entgegengerichtet ist."},
            {"role": "user", "content": f"""Welches Wort ist dem Begriff '{target_word}, im Embedding Raum, genau entgegengerichtet? Gib nur das Ergebniswort an."""}
        ],
        temperature=0.7,  # Creativity level
        max_tokens=10  # Enough for a single word
    )
    return response.choices[0].message.content.strip()

@app.route('/')
def setup():
    return render_template('setup.html', players=players)

@app.route('/new_setup')
def new_setup():
    players = []
    return redirect(url_for('setup'))

@app.route('/add_player', methods=['POST'])
def add_player():
    name = request.form['name']
    if name:
        players.append({'name': name, 'total_score': 0, 'round_score': 0, 'sim_score': 0, 'opp_score': 0})
    return redirect(url_for('setup'))

@app.route('/start_game', methods=['POST'])
def start_game():
    global target_word
    target_word = get_new_target_word()
    
    # reset the scores of the previous round:
    for player in players:
        player["sim_score"] = 0
        player["opp_score"] = 0
        player["round_score"] = 0

    return redirect(url_for('game'))

@app.route('/game')
def game():
    return render_template('game.html', players=players, target_word=target_word)

@app.route('/submit_round', methods=['POST'])
def submit_round():
    # Handle submissions and calculate scores:
    global tar_word_emb
    tar_word_emb = get_embedding(target_word)
    
    for player in players:
        sim_word = request.form.get(f"{player['name']}_similar")
        opp_word = request.form.get(f"{player['name']}_opposite")
        sim_word_emb = get_embedding(sim_word)
        opp_word_emb = get_embedding(opp_word)

        # TODO: show the sim_score, opp_score and round_score below the players.
        sim_score, opp_score = compute_scores(sim_word_emb, opp_word_emb, tar_word_emb)
        round_score = sim_score - opp_score

        # display the scores:
        player["sim_score"] = sim_score
        player["opp_score"] = opp_score
        player["round_score"] = round_score
        player["total_score"] += round_score

    return redirect(url_for('game'))

@app.route('/get_optimal_answer', methods=['GET'])
def get_optimal_answer():
    # Fetch the optimal answer from the model (similar and opposite words)
    optimal_similar_word = get_optimal_similar_word()
    optimal_opposite_word = get_optimal_opposite_word()

    # Return the data as JSON
    return {
        'similar_word': optimal_similar_word,
        'opposite_word': optimal_opposite_word
    }

if __name__ == '__main__':
    app.run(debug=True)
