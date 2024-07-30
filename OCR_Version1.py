from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime
import logging
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import requests
import traceback

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Telegram Bot API settings
TELEGRAM_BOT_TOKEN = "7496198379:AAEi9VUwukzbbal92rG9xuGXIXdahB4T6Kc"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Gemini AI settings
gemini_api_key = "AIzaSyDeXUlxp9OatBZVTXiPqa3rMC4w1Po3A6w"
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

@app.route('/scan_receipt', methods=['POST'])
def scan_receipt():
    logging.info("Received request to scan receipt")
    if 'image' not in request.files:
        logging.error("No image file uploaded")
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files['image']
    chat_id = request.form.get('chat_id')
    
    if not chat_id:
        logging.error("No chat_id provided")
        return jsonify({"error": "No chat_id provided"}), 400
    
    logging.info(f"Processing image for chat_id: {chat_id}")
    
    try:
        image_content = image_file.read()
        logging.info(f"Image size: {len(image_content)} bytes")
        
        extracted_text = extract_text_from_image(image_content)
        if not extracted_text:
            logging.error("Failed to extract text from the image")
            return jsonify({"error": "Failed to extract text from the image"}), 500

        logging.info(f"Extracted text: {extracted_text}")

        cleaned_data_dict = process_extracted_text(extracted_text)
        logging.info(f"Processed data: {cleaned_data_dict}")
        
        # Send the extracted data to Telegram
        send_message_to_telegram(chat_id, json.dumps(cleaned_data_dict, indent=2))

        return jsonify({"message": "Data extracted and sent to Telegram successfully", "data": cleaned_data_dict}), 200

    except Exception as e:
        logging.error(f"Error processing receipt: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": f"Failed to process receipt data: {str(e)}"}), 500

def extract_text_from_image(image_content):
    try:
        image = Image.open(BytesIO(image_content))
        logging.info(f"Image format: {image.format}, size: {image.size}")
        image_bytes = BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()
        response = model.generate_content(["Extract all text from this image:", image_bytes])
        logging.info(f"Gemini AI response: {response.text}")
        return response.text
    except Exception as e:
        logging.error(f"Error in extract_text_from_image: {str(e)}")
        logging.error(traceback.format_exc())
        return None

def process_extracted_text(extracted_text):
    try:
        response = model.generate_content([
            "input: ",
            extracted_text,
            "output: please extract and format the following data from the text:",
            "PLATE NO: [VALUE], DATE: [VALUE], Previous KM: [VALUE], Actual KM: [VALUE], QTY: [VALUE], TOTAL: [VALUE], AMOUNT: [VALUE]"
        ])
        generated_text = response.text
        logging.info(f"Processed text: {generated_text}")

        # ... (rest of the function remains the same)

    except Exception as e:
        logging.error(f"Error in process_extracted_text: {str(e)}")
        logging.error(traceback.format_exc())
        return {}

def send_message_to_telegram(chat_id, message):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logging.info(f"Message sent to Telegram. Response: {response.text}")
    except requests.RequestException as e:
        logging.error(f"Failed to send message to Telegram: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)