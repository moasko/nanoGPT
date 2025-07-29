from flask import Flask, request, jsonify
import torch
from model import GPTConfig, GPT
from sample import sample

app = Flask(__name__)

# Charger modèle (à adapter selon ton config et chemin)
device = 'cpu'
model_path = "out-sqlbot/model.pt"

config = GPTConfig(
    vocab_size=50304,  # adapte si besoin
    block_size=128,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
)

model = GPT(config)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

@app.route('/generate_sql', methods=['POST'])
def generate_sql():
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Aucun prompt fourni"}), 400

    with torch.no_grad():
        sql = sample(model, device=device, start=prompt, max_new_tokens=128)

    return jsonify({"sql": sql})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
