import http.server
import socketserver
import json
import base64
import os
import google.generativeai as genai
from PIL import Image
import io

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Text Extractor</title>
    <script src="https://telegram.org/js/telegram-web-app.js"></script>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
        #result { margin-top: 20px; border: 1px solid #ccc; padding: 10px; min-height: 100px; }
        #loader { display: none; text-align: center; margin-top: 20px; }
        #scanButton { display: block; margin: 20px auto; padding: 10px 20px; font-size: 16px; }
        #fileInput { display: none; }
    </style>
</head>
<body>
    <h1>Image Text Extractor</h1>
    <button id="scanButton">Capture Image</button>
    <input type="file" id="fileInput" accept="image/*" capture="environment">
    <div id="loader">Processing...</div>
    <div id="result"></div>

    <script>
        let tg = window.Telegram.WebApp;
        tg.expand();

        document.getElementById('scanButton').addEventListener('click', function() {
            document.getElementById('fileInput').click();
        });

        document.getElementById('fileInput').addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    extractText(e.target.result.split(',')[1]);
                };
                reader.readAsDataURL(file);
            }
        });

        async function extractText(imageData) {
            document.getElementById('loader').style.display = 'block';
            document.getElementById('result').innerText = '';

            try {
                const response = await fetch('/extract_text', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({image: imageData})
                });
                const data = await response.json();
                document.getElementById('result').innerText = data.text || data.error;
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('result').innerText = 'An error occurred while extracting text.';
            } finally {
                document.getElementById('loader').style.display = 'none';
            }
        }

        tg.ready();
    </script>
</body>
</html>
"""

# Configure Gemini
genai.configure(api_key=os.environ.get("GOOGLE_AI_STUDIO_API_KEY"))
model = genai.GenerativeModel('gemini-pro-vision')

def extract_text_with_gemini(image_data):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        response = model.generate_content(["Extract all the text from this image:", image])
        return response.text
    except Exception as e:
        print(f"Error extracting text with Gemini: {str(e)}")
        return None

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML.encode())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/extract_text':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            try:
                extracted_text = extract_text_with_gemini(data['image'])
                if extracted_text:
                    result = {'text': extracted_text}
                else:
                    result = {'text': 'No text found in the image'}

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404, "Not Found")

def run_server():
    PORT = int(os.environ.get('PORT', 8000))
    with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()

if __name__ == '__main__':
    run_server()