import http.server
import socketserver
import json
import base64
from google.cloud import vision
from google.oauth2 import service_account
import os

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
        #camera { width: 100%; max-width: 400px; height: 300px; border: 2px solid #ccc; margin: 20px auto; }
        #scanButton { display: block; margin: 20px auto; padding: 10px 20px; font-size: 16px; }
    </style>
</head>
<body>
    <h1>Image Text Extractor</h1>
    <video id="camera" autoplay playsinline></video>
    <canvas id="canvas" style="display:none;"></canvas>
    <button id="scanButton" onclick="captureImage()">Scan Image</button>
    <div id="loader">Processing...</div>
    <div id="result"></div>

    <script>
        let tg = window.Telegram.WebApp;
        tg.expand();

        const video = document.getElementById('camera');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        async function setupCamera() {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
            video.srcObject = stream;
        }

        setupCamera();

        function captureImage() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            const imageData = canvas.toDataURL('image/jpeg').split(',')[1];
            extractText(imageData);
        }

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

# Function to get Vision API client
def get_vision_client():
    try:
        credentials_path = 'credentials.json'
        if os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            return vision.ImageAnnotatorClient(credentials=credentials)
        return vision.ImageAnnotatorClient()
    except Exception as e:
        print(f"Error setting up Vision API client: {str(e)}")
        return None

# Custom request handler
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
            
            client = get_vision_client()
            if not client:
                self.send_error(500, "Failed to initialize Vision API client")
                return

            try:
                image = vision.Image(content=base64.b64decode(data['image']))
                response = client.text_detection(image=image)
                texts = response.text_annotations

                if texts:
                    extracted_text = texts[0].description
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

# Run the server
# Replace the last part with this:
httpd = None

def run_server():
    global httpd
    PORT = int(os.environ.get('PORT', 8000))
    httpd = socketserver.TCPServer(("", PORT), RequestHandler)
    print(f"Serving at port {PORT}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
else:
    run_server()