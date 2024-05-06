from flask import Flask, request, send_file
import numpy as np
from PIL import Image
import io
import imageio
from fusion import mds07_fusion  # Import the image enhancement function

app = Flask(__name__)

# Route for uploading the image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return 'No image selected', 400
    
    # Read the uploaded image
    img = imageio.imread(image_file)

    # Process the image
    enhanced_img = mds07_fusion(img)

    # Save the enhanced image to a BytesIO object
    enhanced_img_bytes = io.BytesIO()
    Image.fromarray(enhanced_img).save(enhanced_img_bytes, format='PNG')
    enhanced_img_bytes.seek(0)

    # Return the enhanced image
    return send_file(enhanced_img_bytes, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode