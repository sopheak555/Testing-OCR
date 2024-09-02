import os
import json
import pandas as pd
from datetime import datetime
import google.generativeai as genai
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import logging
import pyodbc
import asyncio
from aiohttp import web
import aiohttp_cors
from PIL import Image
import pytesseract
import base64
import io
import signal
import platform
import re

# Set your Gemini API key
gemini_api_key = ""

# Configure the Google Generative AI
genai.configure(api_key=gemini_api_key)

# Define your Telegram bot token
telegram_bot_token = "6418111603:AAG2Sep82eQwD2V8oFKGDtMPq85rjGZpLCs"



# Database connection (replace with your actual connection details)
connection = pyodbc.connect('DRIVER={"--"};'
                            'SERVER=;'
                            'DATABASE=;'
                            'Trusted_Connection=yes;')
cursor = connection.cursor()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! Send me an image and I'll extract text from it.")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    username = user.username if user.username else user.first_name

    try:
        # Download the image
        photo_file = await update.message.photo[-1].get_file()
        photo_path = os.path.join("downloads", f"{photo_file.file_unique_id}.jpg")
        await photo_file.download_to_drive(photo_path)

        # Read the image file and encode it to base64
        with open(photo_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Send image to server for processing
        async with aiohttp.ClientSession() as session:
            async with session.post('http://localhost:8080/process_image', 
                                    json={'image_data': f"data:image/jpeg;base64,{encoded_string}"}) as response:
                if response.status == 200:
                    result = await response.json()
                    extracted_data = result['extracted_data']
                    
                    logger.info(f"Extracted data: {extracted_data}")
                    
                    # Insert data into database
                    success, message = insert_data_to_db(extracted_data, username)
                    if success:
                        await update.message.reply_text(message)
                        logger.info("Data insertion successful")
                    else:
                        await update.message.reply_text(f"Failed to insert data: {message}")
                        logger.error(f"Data insertion failed: {message}")
                    
                    # Format and send extracted data to user
                    formatted_message = "Extracted data:\n"
                    for key, value in extracted_data.items():
                        formatted_message += f"- {key}: {value}\n"
                    
                    await update.message.reply_text(formatted_message)
                else:
                    error_message = await response.text()
                    logger.error(f"Server responded with status {response.status}: {error_message}")
                    await update.message.reply_text(f"Failed to process the image. Error: {error_message}")
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        await update.message.reply_text(f"An error occurred while processing the image: {str(e)}")
        
def insert_data_to_db(data, username):
    try:
        # Validate data
        required_fields = ['Plate No', 'Date', 'QTY', 'TOTAL', 'AMOUNT']
        for field in required_fields:
            if field not in data or data[field] is None:
                raise ValueError(f"Missing required field: {field}")

        # Log the data being inserted
        logger.info(f"Attempting to insert data: {data}")

        # Handle GLOBAL field
        avg_month, avg_week, avg_day = '', '', ''
        if "GLOBAL" in data:
            global_values = data["GLOBAL"]
            if isinstance(global_values, str):
                global_values = [global_values]
            
            for value in global_values:
                match = re.search(r'\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*/?([MWD])', value)
                if match:
                    amount, period = match.groups()
                    amount = amount.replace(',', '')  # Remove comma from amount
                    if period == 'M':
                        avg_month = f"${amount}M"
                    elif period == 'W':
                        avg_week = f"${amount}W"
                    elif period == 'D':
                        avg_day = f"${amount}D"

        # Prepare the data, handling potential null values
        plate_no = data.get('Plate No')
        date = datetime.strptime(data.get('Date'), '%Y-%m-%d %H:%M:%S') if data.get('Date') else None
        previous_km = float(data.get('Previous KM')) if data.get('Previous KM') else None
        actual_km = float(data.get('Actual KM')) if data.get('Actual KM') else None
        qty = float(data.get('QTY'))
        total = float(data.get('TOTAL'))
        amount = float(data.get('AMOUNT'))
        unit_price = float(data.get('Unit Price')) if data.get('Unit Price') else None
        
        # Log the prepared data
        logger.info(f"Prepared data for insertion: plate_no={plate_no}, date={date}, previous_km={previous_km}, actual_km={actual_km}, qty={qty}, total={total}, amount={amount}, unit_price={unit_price}")

        # Execute the query with explicit parameter types
        cursor.execute('''INSERT INTO Tbl_TotalEnergies(
            [PLATE_NO], [DATE], [Previous_KM], [Actual_KM], [QTY], [TOTAL], [AMOUNT], 
            [Driver_Name], [transaction_date], [UnitPrice], [Company], [Location], 
            [Consum], [Product_Name], [Avg_month], [Avg_week], [Avg_day], [TR_FREQ])
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
            (plate_no, date, previous_km, actual_km, qty, total, amount,
             username, datetime.now(), unit_price, data.get('Company Name'),
             data.get('Location'), str(data.get('Consum')), data.get('Product'),
             avg_month, avg_week, avg_day, data.get('TR_FREQ')))
        
        # Log the SQL query
        logger.info(f"Executed SQL query: {cursor.query}")
        
        connection.commit()
        logger.info(f"Successfully inserted data for plate number: {plate_no}")
        return True, "Data inserted successfully."
    except ValueError as ve:
        error_message = f"Validation error: {str(ve)}"
        logger.error(error_message)
        return False, error_message
    except pyodbc.Error as e:
        error_message = f"Database error: {str(e)}"
        logger.error(error_message)
        connection.rollback()
        return False, error_message
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        logger.error(error_message)
        logger.error(f"Problematic data: {data}")
        connection.rollback()
        return False, error_message
    
async def process_image(request):
    try:
        # Retrieve image data from the request
        data = await request.json()
        image_data = data.get('image_data')
        
        if not image_data:
            raise ValueError("No image data received")

        logging.info("Received image data")

        # Decode base64 image data
        image_data = image_data.split(',')[1]  # Remove the "data:image/jpeg;base64," part
        image_bytes = base64.b64decode(image_data)

        # Open the image with PIL
        image = Image.open(io.BytesIO(image_bytes))

        logging.info("Image opened successfully")

        # Prepare the prompt for the Gemini model
        response = model.generate_content([
            "input: ",
            image,
            """output: 
You are a highly accurate OCR and data extraction tool. 
Please carefully extract the following information from the image provided the data must be correct values:
- Location: [Extract any location information visible in the image, e.g., city, address, etc.]
- Company Name: [Extract the name of the company if present in the image]
- Product: [Extract the name of the product or service mentioned]
- Date: [Extract the date, ensuring it's in the format YYYY-MM-DD HH:MM:SS]
- Plate No: [Extract values of PLATE_NO from image]
- QTY: [Extract the quantity as a number]
- TOTAL: [Extract the total amount as a number, excluding currency symbols]
- AMOUNT: [Extract the amount as a number, excluding currency symbols]
- Unit Price: [Extract the unit price as a number, excluding currency symbols]
- Previous KM: [Extract the previous kilometer reading as a number]
- Actual KM: [Extract the actual kilometer reading as a number]
- Consum: [Extract all consumption values in the format #L/##Km, use an array if there are multiple values]
- GLOBAL: [Find and extract any text lines starting with 'GLOBAL' followed by a currency value]
- TR FREQ: [Extract any text following 'TR FREQ' and format it as 'X/D']
If a data point is not found or cannot be extracted, set the corresponding field to null. Use numbers where applicable and avoid quotes around numeric values."""
        ])

        # Process the response
        if not response.candidates:
            raise ValueError("No candidates found in the response")

        extracted_data_text = response.candidates[0].content.parts[0].text
        extracted_data = json.loads(extracted_data_text.strip("```json\n").strip("\n```"))
        
        logging.info(f"Raw extracted data: {extracted_data}")

        # Process GLOBAL field
        processed_data = process_global_field(extracted_data)
        
        logging.info(f"Processed data: {processed_data}")
        
        return web.json_response({'extracted_data': processed_data})

    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {str(e)}")
        return web.json_response({'error': 'Invalid JSON data'}, status=400)
    except ValueError as e:
        logging.error(f"Value error: {str(e)}")
        return web.json_response({'error': str(e)}, status=400)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        return web.json_response({'error': 'An unexpected error occurred'}, status=500)

def process_global_field(data):
    if "GLOBAL" not in data:
        return data

    global_values = data["GLOBAL"]
    if isinstance(global_values, str):
        global_values = [global_values]

    avg_month, avg_week, avg_day = '', '', ''
    for value in global_values:
        match = re.search(r'\$?\s*(\d+(?:[.,]\d{2})?)\s*/?([MWD])', value)
        if match:
            amount, period = match.groups()
            amount = amount.replace(',', '').replace('.', '')  # Remove comma and period
            amount = f"{int(amount):,d}"  # Format with comma as thousands separator
            if period == 'M':
                avg_month = f"${amount}M"
            elif period == 'W':
                avg_week = f"${amount}W"
            elif period == 'D':
                avg_day = f"${amount}D"

    # Update data with new fields
    data['Avg_month'] = avg_month
    data['Avg_week'] = avg_week
    data['Avg_day'] = avg_day
    
    # Remove the original GLOBAL field
    del data['GLOBAL']

    return data
    print(data)

async def start_server():
    app = web.Application()
    cors = aiohttp_cors.setup(app)
    
    route = app.router.add_post('/process_image', process_image)
    
    cors.add(route, {
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()
    logger.info("Server started on http://localhost:8080")
    return runner, site

async def main():
    # Create an event to signal when to stop the application
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Received stop signal, shutting down...")
        stop_event.set()

    loop = asyncio.get_running_loop()

    # Check if the platform is not Windows
    if platform.system() != 'Windows':
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    else:
        logger.warning("Signal handling not supported on Windows, press Ctrl+C to stop the bot.")

    # Initialize the Telegram bot application
    application = ApplicationBuilder().token(telegram_bot_token).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    
    # Start the web server
    runner, site = await start_server()
    
    # Start the Telegram bot
    await application.initialize()
    await application.start()
    await application.updater.start_polling()

    logger.info("Bot started. Press Ctrl+C to stop.")

    # Wait for the stop event
    try:
        await stop_event.wait()
    finally:
        # Cleanup
        logger.info("Stopping the bot and server...")
        await application.stop()
        await runner.cleanup()
        logger.info("Bot and server stopped.")

if __name__ == "__main__":
    asyncio.run(main())
