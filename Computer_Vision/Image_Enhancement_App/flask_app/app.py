import os
import io
import hashlib
import collections
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import tensorflow as tf
import tensorflow_hub as hub

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

app = Flask(__name__)

# Security: Limit upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Load ESRGAN model globally
print("Loading ESRGAN Model from TF Hub. This may take a moment...")
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(SAVED_MODEL_PATH)
print("Model loaded successfully.")
print("Go to the Webpage http://127.0.0.1:5000")

# Advanced Data Structure: LRU Cache for Enhanced Images (Max 20 items to prevent RAM bloat)
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = collections.OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Pop least recently used (first item)
            self.cache.popitem(last=False)

image_cache = LRUCache(capacity=20)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def preprocess_image(image_bytes):
    """ Decode and preprocess the raw image bytes into a TF tensor. """
    hr_image = tf.image.decode_image(image_bytes)
    # Remove alpha channel if present
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    # Crop to ensure dimensions are multiples of 4 (ESRGAN requirement)
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def process_tensor_tiled(tensor, tile_size=128):
    """
    Dynamic Algorithm: Tessellates the image into patches, inferences individually, 
    and stitches them together to prevent Out-Of-Memory exceptions on high-res images.
    """
    _, h, w, c = tensor.shape
    # If the image is small enough, process it entirely
    if h * w <= 256 * 256:
        return model(tensor)
    
    upscale_factor = 4
    out_h, out_w = h * upscale_factor, w * upscale_factor
    output_tensor = np.zeros((1, out_h, out_w, c), dtype=np.float32)
    
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            end_i = min(i + tile_size, h)
            end_j = min(j + tile_size, w)
            tile = tensor[:, i:end_i, j:end_j, :]
            
            # Pad tile if it's smaller than a multiple of 4 (edges)
            pad_h = (4 - (tile.shape[1] % 4)) % 4
            pad_w = (4 - (tile.shape[2] % 4)) % 4
            if pad_h > 0 or pad_w > 0:
                tile = tf.pad(tile, [[0,0], [0,pad_h], [0,pad_w], [0,0]], mode="REFLECT")
                
            enhanced_tile = model(tile)
            
            e_h = (end_i - i) * upscale_factor
            e_w = (end_j - j) * upscale_factor
            
            # Crop padding back off
            enhanced_tile = enhanced_tile[:, :e_h, :e_w, :]
            
            output_tensor[:, i*upscale_factor:end_i*upscale_factor, j*upscale_factor:end_j*upscale_factor, :] = enhanced_tile.numpy()
            
    return tf.convert_to_tensor(output_tensor)

def finalize_image(tensor):
    """ Converts processed tensor back into an in-memory JPEG byte stream. """
    tensor = tf.squeeze(tensor)
    tensor = tf.clip_by_value(tensor, 0, 255)
    image = Image.fromarray(tf.cast(tensor, tf.uint8).numpy())
    
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG', quality=95)
    img_io.seek(0)
    return img_io

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    if 'file' not in request.files:
        return jsonify({'error': 'No file parameter attached.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file extension. Only PNG, JPG, JPEG, WEBP allowed.'}), 400
        
    try:
        file_bytes = file.read()
        file_hash = generate_file_hash(file_bytes)
        
        # Check LRU Cache
        cached_result = image_cache.get(file_hash)
        if cached_result:
            print(f"Cache HIT for {file_hash}")
            cached_io = io.BytesIO(cached_result)
            return send_file(cached_io, mimetype='image/jpeg')

        print(f"Cache MISS for {file_hash}. Running heavy inference...")
        hr_image = preprocess_image(file_bytes)
        
        # Apply the Dynamic Tiling Algorithm for processing
        fake_image = process_tensor_tiled(hr_image, tile_size=128)
        
        # Convert to stream
        out_stream = finalize_image(fake_image)
        
        # Cache the result bytes
        image_cache.put(file_hash, out_stream.getvalue())
        
        return send_file(out_stream, mimetype='image/jpeg')

    except Exception as e:
        print(f"Error during enhancement: {e}")
        return jsonify({'error': 'Failed to process image.'}), 500

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({'error': 'File is too large. Maximum size is 16MB.'}), 413

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)