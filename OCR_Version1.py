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

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

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
    
    logging.info(f"Processing image for chat_id: {chat_id}")
    if not chat_id:
        return jsonify({"error": "No chat_id provided"}), 400
    
    try:
        image_content = image_file.read()
        extracted_text = extract_text_from_image(image_content)
        if not extracted_text:
            return jsonify({"error": "Failed to extract text from the image"}), 500

        cleaned_data_dict = process_extracted_text(extracted_text)
        
        # Send the extracted data to Telegram
        send_message_to_telegram(chat_id, json.dumps(cleaned_data_dict, indent=2))

        return jsonify({"message": "Data extracted and sent to Telegram successfully", "data": cleaned_data_dict}), 200

    except Exception as e:
        logging.error(f"Error processing receipt: {e}")
        return jsonify({"error": "Failed to process receipt data"}), 500

def extract_text_from_image(image_content):
    image = Image.open(BytesIO(image_content))
    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    response = model.generate_content(["Extract all text from this image:", image_bytes])
    return response.text

def process_extracted_text(extracted_text):
    response = model.generate_content([
        "input: ",
        extracted_text,
        "output: please extract and format the following data from the text:",
        "PLATE NO: [VALUE], DATE: [VALUE], Previous KM: [VALUE], Actual KM: [VALUE], QTY: [VALUE], TOTAL: [VALUE], AMOUNT: [VALUE]"
    ])
    generated_text = response.text

    try:
        list_of_dicts = json.loads(f"[{generated_text}]")
        data_dict = {k: v for d in list_of_dicts for k, v in d.items()}
    except json.JSONDecodeError:
        logging.error(f"JSON Decode Error. Generated text: {generated_text}")
        data_dict = {}

    cleaned_data_dict = {key.replace(":", "").strip(): (value.strip("$, ") if value is not None else None)
                         for key, value in data_dict.items()}

    if "DATE" in cleaned_data_dict and cleaned_data_dict["DATE"]:
        date_str = cleaned_data_dict["DATE"].replace(": ", "").strip()
        try:
            cleaned_data_dict["DATE"] = datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
        except ValueError:
            logging.error(f"Date conversion error. Original date: {date_str}")
            cleaned_data_dict["DATE"] = None

    for key in ["TOTAL", "AMOUNT", "QTY"]:
        if key in cleaned_data_dict and cleaned_data_dict[key]:
            cleaned_data_dict[key] = cleaned_data_dict[key].replace("$", "").replace(",", ".")

    return cleaned_data_dict

def send_message_to_telegram(chat_id, message):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        logging.error(f"Failed to send message to Telegram: {response.text}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)