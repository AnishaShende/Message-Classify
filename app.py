from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__)

# Load model and tokenizer
MODEL_NAME = "AnishaShende/message-classifier-distilbert"

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Get labels from model config
id2label = model.config.id2label

# Define your custom label mapping
# Replace these with your actual category names
label_mapping = {
    "LABEL_0": "announcement",
    "LABEL_1": "casual",
    "LABEL_2": "greeting",
    "LABEL_3": "important",
    "LABEL_4": "reminder",
    "LABEL_5": "spam"
}

@app.route('/classify', methods=['POST'])
def classify_message():
    """Classify a single message"""
    # Get message from request
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "Please provide a message to classify"}), 400
    
    message = data['message']
    result = _classify_single_message(message)
    return jsonify(result)

@app.route('/classify_batch', methods=['POST'])
def classify_batch():
    """Classify an array of messages"""
    # Get messages from request
    data = request.json
    if not data or 'messages' not in data:
        return jsonify({"error": "Please provide an array of messages to classify"}), 400
    
    messages = data['messages']
    
    # Validate input
    if not isinstance(messages, list):
        return jsonify({"error": "The 'messages' field must be an array"}), 400
    
    if len(messages) == 0:
        return jsonify({"error": "The messages array cannot be empty"}), 400
    
    results = []
    for message in messages:
        if not isinstance(message, str):
            # Skip non-string messages with a warning
            results.append({
                "message": str(message),
                "error": "Invalid message type. Expected string."
            })
            continue
        
        result = _classify_single_message(message)
        results.append(result)
    
    return jsonify({
        "results": results,
        "count": len(results)
    })

def _classify_single_message(message):
    """Helper function to classify a single message"""
    # Tokenize message
    inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True).to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get predicted class and confidence
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class_id].item()
    
    # Get class name from id
    predicted_class = id2label[predicted_class_id]
    # Get the human-readable label name
    predicted_class_name = label_mapping.get(predicted_class, predicted_class)
    
    # Return prediction
    return {
        "message": message,
        "predicted_class": predicted_class,
        "predicted_class_name": predicted_class_name,
        "confidence": confidence,
        "all_probabilities": {id2label[i]: prob.item() for i, prob in enumerate(probabilities[0])},
        "all_probability_names": {label_mapping.get(id2label[i], id2label[i]): prob.item() for i, prob in enumerate(probabilities[0])}
    }

@app.route('/batch_classify', methods=['POST'])
def batch_classify():
    """Alternative endpoint name for batch classification"""
    return classify_batch()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model": MODEL_NAME,
        "device": device.type
    })

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    # Print startup message
    print(f"Starting Message Classification API on port {port}")
    print(f"Using device: {device}")
    app.run(host='0.0.0.0', port=port)
