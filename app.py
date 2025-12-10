from flask import Flask, render_template, request
import numpy as np
import pickle
import torch
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

# ----------------- Load Crop Recommendation Model -----------------
model = pickle.load(open("models/crop-recommendation/model.pkl", "rb"))
sc = pickle.load(open("models/crop-recommendation/standscaler.pkl", "rb"))
ms = pickle.load(open("models/crop-recommendation/minmaxscaler.pkl", "rb"))

crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
    10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean",
    18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
    21: "Chickpea", 22: "Coffee"
}

# ----------------- Load AI Assistant Models -----------------
tokenizer = GPT2Tokenizer.from_pretrained("models/tokenizer")
vanilla_model = GPT2LMHeadModel.from_pretrained("gpt2")
finetuned_model = GPT2LMHeadModel.from_pretrained("models/gpt2-finetuned-agri")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vanilla_model.to(device).eval()
finetuned_model.to(device).eval()

def generate_response(model, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=150,  # Increased to let sentences finish naturally
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def complete_sentence(text):
    """Ensure the text ends with a complete sentence."""
    sentences = re.findall(r'[^.!?]*[.!?]', text)
    if sentences:
        return ''.join(sentences).strip()
    return text.strip()

def clean_response(raw_output, prompt):
    cleaned = raw_output.replace(prompt, "").strip()
    if "Answer:" in cleaned:
        cleaned = cleaned.split("Answer:")[-1].strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = complete_sentence(cleaned)
    return cleaned

# ----------------- Routes -----------------
@app.route("/")
def home():
    return render_template("index.html")

# Crop Recommendation Page
@app.route("/crop", methods=["GET", "POST"])
def crop():
    result = None
    if request.method == "POST":
        N = request.form["Nitrogen"]
        P = request.form["Phosporus"]
        K = request.form["Potassium"]
        temp = request.form["Temperature"]
        humidity = request.form["Humidity"]
        ph = request.form["Ph"]
        rainfall = request.form["Rainfall"]

        features = np.array([N, P, K, temp, humidity, ph, rainfall]).reshape(1, -1)
        scaled_features = ms.transform(features)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        crop = crop_dict.get(prediction[0], "Unknown")
        result = f"{crop} is the best crop to be cultivated right there"

    return render_template("crop.html", result=result)

# AI Assistant Page
@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    question, vanilla_output, finetuned_output = "", "", ""
    if request.method == "POST":
        question = request.form["question"]
        vanilla_prompt = f"Question: {question} Answer:"
        finetuned_prompt = f"<|startoftext|>Question: {question} Answer:"

        raw_vanilla = generate_response(vanilla_model, vanilla_prompt)
        raw_finetuned = generate_response(finetuned_model, finetuned_prompt)

        vanilla_output = clean_response(raw_vanilla, vanilla_prompt)
        finetuned_output = clean_response(raw_finetuned, finetuned_prompt)

    return render_template("chatbot.html",
                           question=question,
                           vanilla_output=vanilla_output,
                           finetuned_output=finetuned_output)

if __name__ == "__main__":
    app.run(debug=True)
