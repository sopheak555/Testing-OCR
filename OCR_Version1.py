from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime
import logging
import pyodbc
import google.generativeai as genai
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure database connection
try:
    connection = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                                'SERVER=DESKTOP-34UAA17;'
                                'DATABASE=Car Information;'
                                'Trusted_Connection=yes;')
    cursor = connection.cursor()
    logging.info("Connection to SQL Server database established successfully.")
except Exception as e:
    logging.error(f"Error connecting to SQL Server database: {e}")

# Configure Gemini AI
gemini_api_key = "AIzaSyDeXUlxp9OatBZVTXiPqa3rMC4w1Po3A6w"
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

@app.route('/scan_receipt', methods=['POST'])
def scan_receipt():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files['image']
    
    try:
        image_content = image_file.read()
        extracted_text = extract_text_from_image(image_content)
        if not extracted_text:
            return jsonify({"error": "Failed to extract text from the image"}), 500

        cleaned_data_dict = process_extracted_text(extracted_text)
        insert_data_to_database(cleaned_data_dict)

        return jsonify({"message": "Data extracted and inserted successfully", "data": cleaned_data_dict}), 200

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
    
    # Assign Driver Name (replace with actual logic to get username)
    cleaned_data_dict["Driver Name"] = "YourUsername"

    # Convert date string to SQL Server date format
    if "DATE" in cleaned_data_dict and cleaned_data_dict["DATE"]:
        date_str = cleaned_data_dict["DATE"].replace(": ", "").strip()
        try:
            cleaned_data_dict["DATE"] = datetime.strptime(date_str, "%d/%m/%Y").strftime("%Y-%m-%d")
        except ValueError:
            logging.error(f"Date conversion error. Original date: {date_str}")
            cleaned_data_dict["DATE"] = None

    # Clean and convert numeric values
    for key in ["TOTAL", "AMOUNT", "QTY"]:
        if key in cleaned_data_dict and cleaned_data_dict[key]:
            cleaned_data_dict[key] = cleaned_data_dict[key].replace("$", "").replace(",", ".")

    return cleaned_data_dict

def insert_data_to_database(cleaned_data_dict):
    transaction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    plate_no = cleaned_data_dict.get("PLATE NO", "")
    date = cleaned_data_dict.get("DATE", "1900-01-01")
    previous_km = float(cleaned_data_dict.get("Previous KM", 0))
    actual_km = float(cleaned_data_dict.get("Actual KM", 0))
    qty = float(cleaned_data_dict.get("QTY", 0.0))
    total = float(cleaned_data_dict.get("TOTAL", 0.0))
    amount = float(cleaned_data_dict.get("AMOUNT", 0.0))
    driver_name = cleaned_data_dict.get("Driver Name", "")

    cursor.execute('''INSERT INTO Tbl_TotalEnergies([PLATE_NO], [DATE], [Previous_KM], [Actual_KM], [QTY], [TOTAL], [AMOUNT], [Driver_Name], [transaction_date])
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                  (plate_no, date, previous_km, actual_km, qty, total, amount, driver_name, transaction_date))
    connection.commit()

if __name__ == "__main__":
    app.run(debug=True, ssl_context='adhoc')  # Use HTTPS for production