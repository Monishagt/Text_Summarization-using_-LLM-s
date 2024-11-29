from flask import Flask, request, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer

# Initialize the Flask app
app = Flask(__name__)

# Load the BART model and tokenizer from the folder where you unzipped it
model_path = 'bart_summary_model'  # Path to the folder containing your model
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()  # Get the JSON payload
    dialogue = data.get('dialogue', '')  # Extract dialogue from the request

    # Tokenize the input dialogue
    inputs = tokenizer(dialogue, max_length=1024, truncation=True, return_tensors="pt")

    # Generate the summary
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=50,  # Max length for the generated summary
        min_length=25,  # Min length for the generated summary
        length_penalty=2.0,  # Penalize long summaries
        num_beams=4,  # Number of beams for beam search
    )

    # Decode the generated summary and return it
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True)
