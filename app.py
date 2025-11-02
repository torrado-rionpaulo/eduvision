# ============================ ResNet50 Multi-Task Backend (Flask) — HYBRID ANALYSIS VERSION ============================
import os, io, sys
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import math
import base64

# --- Imports for K-Means/Color Analysis ---
from sklearn.cluster import KMeans
from skimage.color import rgb2lab # Requires scikit-image

# --- Imports for Flask and CORS ---
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --------------------- CONFIG ---------------------
IMG_SIZE = (224, 224)
# K-Means analysis size (smaller for speed and stable segmentation)
KMEANS_SIZE = (300, 300) 
CVD_CLASSES = ['none','protan','deutan','tritan','achroma']
CVD_IDX = {k:i for i,k in enumerate(CVD_CLASSES)}
SIMULATION_KINDS = ['protan', 'deutan', 'tritan'] # The 3 simulations to generate

# Calibration & human-like blend (Replace T_ACC with your actual calibrated value!)
T_ACC_DEFAULT = 2.850 
ACCESS_ALPHA = 0.85
QUAL_BONUS = 0.35

# Weights path (Assumes weights file is in the same directory as this script)
best_model_path = "content/resnet50_accessibility.weights.h5" 

# Global variables for the model and calibration temperature
model = None
T_ACC = T_ACC_DEFAULT

# K-Means Target Colors
EXPECTED_CLASSES = {
    "red": np.array([255, 0, 0]),
    "orange": np.array([255, 165, 0]),
    "yellow": np.array([255, 255, 0]),
    "green": np.array([0, 128, 0]),
    "lime": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "black": np.array([0, 0, 0]),
    "white": np.array([255, 255, 255])
}

# --------------------- CVD SIMULATORS (Unchanged) ---------------------
# ... (_to_linear_rgb, _to_srgb, CVD_MATS, simulate_cvd are unchanged)

def _to_linear_rgb(arr):
    arr = arr.astype(np.float32) / 255.0
    mask = arr <= 0.04045
    out = np.empty_like(arr, dtype=np.float32)
    out[mask] = arr[mask] / 12.92
    out[~mask] = ((arr[~mask] + 0.055) / 1.055) ** 2.4
    return out

def _to_srgb(arr_lin):
    mask = arr_lin <= 0.0031308
    out = np.empty_like(arr_lin, dtype=np.float32)
    out[mask] = 12.92 * arr_lin[mask]
    out[~mask] = 1.055 * (np.clip(arr_lin[~mask], 0, 1) ** (1/2.4)) - 0.055
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

CVD_MATS = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]], dtype=np.float32),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]], dtype=np.float32),
    "tritan": np.array([[1.255528, -0.076749, -0.178779],
                        [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]], dtype=np.float32),
}


def simulate_cvd(pil_img, kind="deutan"):
    arr = np.array(pil_img).astype(np.uint8)
    lin = _to_linear_rgb(arr)
    M = CVD_MATS.get(kind, CVD_MATS["deutan"])
    out = lin @ M.T
    return Image.fromarray(_to_srgb(out))

# --------------------- RESNET UTILITIES (Unchanged) ---------------------
# ... (percent_to_label, calibrate_prob, final_accessibility_percent, load_rgb_white_bg_resized, pil_to_base64)

def percent_to_label(pct):
    if pct >= 50.0: return "Accessible"
    elif pct >= 20.0: return "Partially accessible"
    else: return "Not accessible"

def calibrate_prob(p, T):
    eps = 1e-6
    p = float(np.clip(p, eps, 1-eps))
    logit = math.log(p/(1-p))
    return 1.0 / (1.0 + math.exp(-logit/float(T)))

def final_accessibility_percent(raw_prob, qual_est, T):
    p_cal = calibrate_prob(raw_prob, T)
    score = ACCESS_ALPHA * p_cal + QUAL_BONUS * float(np.clip(qual_est,0,1))
    score = float(np.clip(score, 0.0, 1.0))
    return 100.0 * score

def load_rgb_white_bg_resized(pil_img, size=IMG_SIZE):
    img = ImageOps.exif_transpose(pil_img)
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255,255,255,255))
        img = Image.alpha_composite(white, rgba).convert("RGB")
    else:
        img = img.convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    return img

def pil_to_base64(pil_img, format='JPEG'):
    buffered = io.BytesIO()
    # Ensure it is RGB before saving to JPEG
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    pil_img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --------------------- K-MEANS UTILITIES (Added for Color Analysis) ---------------------

def segment_image(img_array, n_segments=7):
    """Segments image using K-Means clustering in RGB space."""
    pixels = img_array.reshape(-1, 3)
    # n_init='auto' used for modern scikit-learn compatibility
    kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init='auto').fit(pixels)
    labels = kmeans.labels_.reshape(img_array.shape[:2])
    centers = kmeans.cluster_centers_.astype(int)
    masks = {i: (labels == i) for i in range(n_segments)}
    return labels, centers, masks

def classify_segments_lab(centers, expected_classes):
    """Classifies K-Means cluster centers to the closest named color in LAB space."""
    centers_lab = rgb2lab(np.array(centers).reshape(-1, 1, 3)).reshape(-1, 3)
    expected_lab = {name: rgb2lab(np.array(color).reshape(1, 1, 3)).reshape(3) 
                    for name, color in expected_classes.items()}
    def closest_color(center_lab):
        return min(expected_lab.items(), key=lambda kv: np.linalg.norm(center_lab - kv[1]))[0]
    return [closest_color(center_lab) for center_lab in centers_lab]

def compute_accuracy(predicted, expected):
    """Calculates the percentage of correctly classified segments."""
    correct = sum(p == e for p, e in zip(predicted, expected))
    # Use 99.9% instead of 100% to ensure "perfect" ground truth classification isn't confusing
    acc = round((correct / len(predicted)) * 100, 2)
    return 99.9 if acc == 100.0 else acc

def find_misclassifications(filtered_labels, filtered_masks, original_labels, original_masks):
    """Identifies which original segments are misread under the CVD filter."""
    misreads = []
    for f_idx, f_mask in filtered_masks.items():
        # Find the original segment that has the largest overlap with the filtered segment
        overlap_scores = {
            o_idx: np.logical_and(f_mask, original_masks[o_idx]).sum()
            for o_idx in original_masks
        }
        best_match_idx = max(overlap_scores, key=overlap_scores.get)
        expected = original_labels[best_match_idx]
        predicted = filtered_labels[f_idx]

        if expected != predicted:
            misreads.append(f"{expected.capitalize()} misread as {predicted.capitalize()}")
    return misreads

def get_kmeans_input_array(pil_img, size=KMEANS_SIZE):
    """Loads and preprocesses the image specifically for K-Means analysis."""
    img = ImageOps.exif_transpose(pil_img).convert("RGB").resize(size, Image.BILINEAR)
    img_array = np.array(img)
    # Simple power-law correction/normalization for better segmentation
    return np.clip((img_array / 255.0) ** 0.9 * 255, 0, 255).astype(np.uint8)

# --------------------- MODEL DEFINITION & LOADING ---------------------
# ... (build_model_for_inference, init_model_and_calibration, predict_access_and_cvd are unchanged)

def build_model_for_inference():
    """Defines the ResNet50 architecture with three custom heads."""
    backbone = ResNet50(include_top=False, weights=None, input_shape=(*IMG_SIZE, 3), pooling="avg")
    backbone.trainable = False

    inp = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = backbone(inp, training=False) 

    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(384, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x, training=False)
    x = tf.keras.layers.Dropout(0.25)(x)

    acc_out = tf.keras.layers.Dense(1, activation="sigmoid", name="acc_out", dtype="float32")(x)
    cvd_logits = tf.keras.layers.Dense(256, activation="relu")(x)
    cvd_logits = tf.keras.layers.Dropout(0.25)(cvd_logits)
    cvd_out = tf.keras.layers.Dense(len(CVD_CLASSES), activation="softmax", name="cvd_out", dtype="float32")(cvd_logits)
    qual_out = tf.keras.layers.Dense(1, activation="sigmoid", name="qual_out", dtype="float32")(x)

    model = models.Model(inp, [acc_out, cvd_out, qual_out], name="resnet50_accessibility_cvd_qual")
    return model

def init_model_and_calibration():
    """Initializes and loads model weights, and sets calibration temperature."""
    global model, T_ACC
    model = build_model_for_inference()
    
    if os.path.exists(best_model_path):
        try:
            dummy_input = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
            model.predict(dummy_input, verbose=0)
            
            model.load_weights(best_model_path)
            print(f"✅ Model weights loaded successfully from {best_model_path}.")
        except Exception as e:
            print(f"❌ Error loading weights: {e}. Model will use random initialization.")
    else:
        print(f"⚠️ Warning: Weights file not found at {best_model_path}. Model will use random initialization.")

    # Use the T_ACC value from your prompt's config
    T_ACC = T_ACC_DEFAULT 
    print(f"🌡️ Calibration Temperature (T_ACC) set to: {T_ACC:.3f}")

def predict_access_and_cvd(pil_img):
    """Runs inference on a PIL image and returns analysis results."""
    global model, T_ACC

    pil_img_resized = load_rgb_white_bg_resized(pil_img)
    ar = np.array(pil_img_resized, dtype=np.float32)
    ar = np.expand_dims(ar, 0)
    ar = preprocess_input(ar)
    
    # pa: prob_acc (1x1), pc: prob_cvd (1x5), pq: prob_qual (1x1)
    pa, pc, pq = model.predict(ar, verbose=0)
    
    # Extract results
    raw_p = float(pa.ravel()[0])
    cvd_idx = int(np.argmax(pc.ravel()))
    cvd_name = CVD_CLASSES[cvd_idx]
    qual_est = float(pq.ravel()[0])
    
    pct = final_accessibility_percent(raw_p, qual_est, T_ACC)
    label = percent_to_label(pct)

    # Format CVD probability distribution
    cvd_probs = {}
    pc_list = pc.ravel().tolist()
    for i, class_name in enumerate(CVD_CLASSES):
        cvd_probs[class_name] = round(pc_list[i], 3)
    
    return {
        'accessibility_percent': float(f"{pct:.2f}"),
        'verdict': label,
        'cvd_class': cvd_name,
        'cvd_probabilities': cvd_probs,
        'raw_probability': float(f"{raw_p:.3f}"),
    }


# --------------------- FLASK APPLICATION (UPDATED) ---------------------

app = Flask(__name__)
CORS(app) 

@app.route('/analysis')
def index():
    return render_template('index.html')

@app.route('/') 
def homepage():
    return render_template('homepage.html')

@app.route('/classify-image', methods=['POST'])
def classify_image_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        try:
            # 1. Load the original image
            img_stream = io.BytesIO(file.read())
            original_pil = Image.open(img_stream).convert("RGB")
            
            # 2. K-Means analysis for Ground Truth (Original Image)
            kmeans_img_array = get_kmeans_input_array(original_pil)
            labels_orig, centers_orig, masks_orig = segment_image(kmeans_img_array)
            classes_orig = classify_segments_lab(centers_orig, EXPECTED_CLASSES)
            
            results = {}
            
            # 3. Analyze and encode the Normal Image (Combination of ResNet & K-Means Ground Truth)
            normal_analysis = predict_access_and_cvd(original_pil)
            
            results['Normal'] = {
                'analysis': normal_analysis,
                # K-Means metrics for normal vision (Ground Truth)
                'color_metrics': {
                    'identified_segments': list(set(classes_orig)), # List unique identified segments
                    'classification_accuracy': compute_accuracy(classes_orig, classes_orig), 
                    'misclassified_segments': [], # None for ground truth
                },
                'b64_img': pil_to_base64(original_pil)
            }
            
            # 4. Generate, analyze, and encode the 3 simulated images
            for kind in SIMULATION_KINDS:
                # Generate simulation (uses the RGB original_pil)
                simulated_pil = simulate_cvd(original_pil, kind=kind)
                
                # ResNet analysis on simulated image
                simulated_analysis = predict_access_and_cvd(simulated_pil)
                
                # K-Means analysis on simulated image
                kmeans_sim_array = get_kmeans_input_array(simulated_pil)
                labels_sim, centers_sim, masks_sim = segment_image(kmeans_sim_array)
                classes_sim = classify_segments_lab(centers_sim, EXPECTED_CLASSES)
                
                # Compare simulated classification against original ground truth
                sim_acc = compute_accuracy(classes_sim, classes_orig)
                misreads = find_misclassifications(classes_sim, masks_sim, classes_orig, masks_orig)
                
                # Store results and encoded image
                results[kind.capitalize()] = {
                    'analysis': simulated_analysis,
                    'color_metrics': {
                        'identified_segments': list(set(classes_sim)),
                        'classification_accuracy': sim_acc,
                        'misclassified_segments': misreads,
                    },
                    'b64_img': pil_to_base64(simulated_pil)
                }

            # 5. Return combined results
            return jsonify({
                'filename': secure_filename(file.filename),
                'simulations': results
            })

        except Exception as e:
            print(f"Classification error: {e}", file=sys.stderr)
            return jsonify({'error': f'An error occurred during classification: {e}'}), 500
            
    return jsonify({'error': 'Invalid file type. Accepted: jpg, png, bmp, tif'}), 400

if __name__ == '__main__':
    init_model_and_calibration()
    print("🚀 Starting Flask server on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)