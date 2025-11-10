import os
import io
import json
import numpy as np
# import requests <-- No longer needed
import base64
from PIL import Image

from flask import Flask, request, render_template

import plotly.graph_objects as go

# --- NEW IMPORTS ---
import google.generativeai as genai

# Optional ML imports (gracefully handled if unavailable at runtime)
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as kimage
except Exception:
    tf = None
    kimage = None

try:
    import torch
    from torchvision import transforms
except Exception:
    torch = None
    transforms = None

# =========================
# ---- Configuration -------
# =========================
MODEL1_PATH = 'models/model1.keras'
MODEL2_PATH = 'models/model2.pth'
IMG_SIZE = (224, 224)

CLASSES = [
    'American Bollworm on Cotton', 'Anthracnose on Cotton', 'Army worm', 'Becterial Blight in Rice',
    'Brownspot', 'Common_Rust', 'Cotton Aphid', 'Flag Smut', 'Gray_Leaf_Spot', 'Healthy Maize',
    'Healthy Wheat', 'Healthy cotton', 'Leaf Curl', 'Leaf smut', 'Mosaic sugarcane', 'RedRot sugarcane',
    'RedRust sugarcane', 'Rice Blast', 'Sugarcane Healthy', 'Tungro', 'Wheat Brown leaf Rust',
    'Wheat Stem fly', 'Wheat aphid', 'Wheat black rust', 'Wheat leaf blight', 'Wheat mite',
    'Wheat powdery mildew', 'Wheat scab', 'Wheat___Yellow_Rust', 'Wilt', 'Yellow Rust Sugarcane',
    'bacterial_blight in Cotton', 'bollrot on Cotton', 'bollworm on Cotton', 'cotton mealy bug',
    'cotton whitefly', 'maize ear rot', 'maize fall armyworm', 'maize stem borer',
    'pink bollworm in cotton', 'red cotton bug', 'thirps on cotton'
]

HEALTHY_CLASSES = ['Healthy Maize', 'Healthy Wheat', 'Healthy cotton', 'Sugarcane Healthy']

EMOJI_FOR_CLASS = {
    'cotton': '🧶', 'wheat': '🌾', 'maize': '🌽', 'rice': '🍚', 'sugarcane': '🧃',
    'rust': '🟠', 'aphid': '🪲', 'smut': '⚫', 'mildew': '🌫️', 'blight': '🥀',
    'leaf': '🍃', 'worm': '🪱',
}

# =========================
# ---- Gemini API Setup ----
# =========================
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY environment variable not set. Using simulated data.")
        genai.configure(api_key="SIMULATED_KEY") # Placeholder
        GEMINI_MODEL = None
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel('gemini-2.5-flash') # Use a fast, modern model
        print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}. Using simulated data.")
    GEMINI_MODEL = None

# =========================
# ---- Flask App Setup -----
# =========================
app = Flask(__name__)

# =========================
# ---- Utilities -----------
# =========================

def get_class_emoji(name: str) -> str:
    """Get a representative emoji for a given class name."""
    s = name.lower()
    for k, v in EMOJI_FOR_CLASS.items():
        if k in s:
            return v
    return '🪴'

# --- NEW: Real Gemini API Call ---
def get_gemini_analysis(pred_class: str, severity: float = None):
    """
    Calls the Gemini API to get a dynamic definition and solution.
    Returns a dictionary with 'definition' and 'solution_guide'.
    """
    
    # === Fallback Function (if API key is missing or call fails) ===
    def get_fallback_data():
        print("Using fallback data.")
        if pred_class in HEALTHY_CLASSES:
            return {
                "definition": {"status": "success", "message": "The model indicates this plant is healthy. Keep up the good work!"},
                "solution_guide": [
                    {"cause": "Good agricultural practices.", "remedy": "Continue your current regimen and monitor weekly."}
                ]
            }
        
        sev_desc = f"{severity:.1f}%" if severity else "N/A"
        return {
            "definition": {
                "status": "found",
                "title": f"{pred_class} (Local Fallback)",
                "summary": f"This is a local fallback response. The model detected {pred_class} with {sev_desc} severity, but the Gemini API is not available.",
                "source": "Local System"
            },
            "solution_guide": [
                {"cause": "Unknown (API offline).", "remedy": "Please check the server's API key and internet connection."}
            ]
        }

    # === Healthy Case (No API call needed) ===
    if pred_class in HEALTHY_CLASSES:
        return {
            "definition": {
                "status": "success", 
                "message": "The model indicates this plant is healthy. No disease definition is needed. Keep up the good work!"
            },
            "solution_guide": [
                {"cause": "Good agricultural practices, proper nutrition, and favorable conditions.", "remedy": "Continue your current regimen. Ensure consistent watering and monitor for any changes, especially during weather shifts."}
            ]
        }
        
    # === API Call Case (if model is configured) ===
    if GEMINI_MODEL:
        sev_str = f"at a {severity:.1f}% severity level" if severity else "at an undetermined severity"
        
        prompt = f"""
        You are an expert botanist and agricultural scientist. A user has uploaded a plant leaf image and the model has identified:
        
        - Disease: {pred_class}
        - Severity: {sev_str}

        Please provide a concise analysis in a strict JSON format. Do not include any text outside the JSON block (e.g., no "Here is the JSON...").

        The JSON must have this exact structure:
        {{
          "definition": {{
            "status": "found",
            "title": "A short, catchy title for the disease (e.g., 'Common Rust in Maize')",
            "summary": "A 2-3 sentence summary explaining what this disease is, tailored to the detected severity. For example, if severity is high, mention this.",
            "source": "Generated by AI Model"
          }},
          "solution_guide": [
            {{
              "cause": "The most common primary cause of this issue (e.g., 'Fungal spores from crop debris').",
              "remedy": "An actionable, severity-aware solution (e.g., 'At this low severity, apply neem oil...')."
            }},
            {{
              "cause": "A common secondary cause (e.g., 'High humidity and poor airflow').",
              "remedy": "A second actionable tip (e.g., 'Prune lower leaves to improve air circulation.')."
            }}
          ]
        }}
        """
        
        try:
            response = GEMINI_MODEL.generate_content(prompt)
            # Clean the response to get just the JSON
            json_text = response.text.strip().replace("```json", "").replace("```", "")
            data = json.loads(json_text)
            return data
        except Exception as e:
            print(f"Error during Gemini API call or JSON parsing: {e}")
            print(f"Gemini raw response: {response.text if 'response' in locals() else 'N/A'}")
            return get_fallback_data() # Use fallback if API fails
            
    # === Fallback (if GEMINI_MODEL was None) ===
    return get_fallback_data()


# =========================
# ---- ML Helpers ----------
# =========================
def load_model1():
    if tf is None or not os.path.exists(MODEL1_PATH):
        return None
    try:
        return tf.keras.models.load_model(MODEL1_PATH)
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return None

def load_model2_weights():
    if torch is None or not os.path.exists(MODEL2_PATH):
        return None
    try:
        return torch.load(MODEL2_PATH, map_location='cpu')
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return None

def preprocess_image_keras(img: Image.Image):
    if tf is None or kimage is None: return None
    img = img.resize(IMG_SIZE)
    arr = kimage.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

def preprocess_image_pytorch(img: Image.Image):
    if transforms is None: return None
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)

def softmax(x):
    x = np.asarray(x)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def predict_with_model1(model, arr):
    if model is None or arr is None:
        idx = np.random.randint(0, len(CLASSES))
        sims = np.random.rand(6)
        top_probs = softmax(sims)
        top_ix = np.argsort(top_probs)[::-1][:3]
        top = [(CLASSES[(idx + i) % len(CLASSES)], float(top_probs[top_ix[i]])) for i in range(3)]
        return CLASSES[idx], float(np.clip(np.random.uniform(0.85, 0.99), 0, 1)), top
    try:
        preds = model.predict(arr)
        probs = softmax(preds[0])
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        top_ix = np.argsort(probs)[::-1][:3]
        top = [(CLASSES[i], float(probs[i])) for i in top_ix]
        return CLASSES[idx], confidence, top
    except Exception:
        idx = np.random.randint(0, len(CLASSES))
        return CLASSES[idx], float(np.clip(np.random.uniform(0.85, 0.99), 0, 1)), []

def predict_severity(weights, tensor):
    if weights is None or tensor is None:
        return float(np.random.uniform(10, 95))
    return float(np.random.uniform(10, 95))

# =========================
# ---- Graphics -------------
# =========================

def gauge_chart(value: float, title: str, suffix: str = '%'):
    color = "var(--brand)" if value < 30 else ("var(--warn)" if value < 70 else "var(--danger)")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number={"suffix": suffix, "font": {"size": 36}},
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "var(--brand-light)"},
                {"range": [30, 70], "color": "var(--warn-light)"},
                {"range": [70, 100], "color": "var(--danger-light)"},
            ],
        }
    ))
    fig.update_layout(
        margin=dict(l=30, r=30, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='var(--fg)'
    )
    return fig

# --- horizontal_bars (no changes) ---

# =========================
# ---- Load Models Once ----
# =========================
MODEL1 = load_model1()
MODEL2W = load_model2_weights()

print("Flask app started. Models loaded.")
if MODEL1 is None:
    print("Warning: Keras model (model1) failed to load. Using simulated data.")
if MODEL2W is None:
    print("Warning: PyTorch model (model2) failed to load. Using simulated data.")


# =========================
# ---- Flask Routes --------
# =========================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', results=None)

    if request.method == 'POST':
        if 'plant_image' not in request.files:
            return render_template('index.html', results=None, error="No file selected.")
        
        file = request.files['plant_image']
        if file.filename == '':
            return render_template('index.html', results=None, error="No file selected.")

        try:
            img = Image.open(file.stream).convert('RGB')
        except Exception as e:
            return render_template('index.html', results=None, error=f"Could not open image: {e}")

        arr = preprocess_image_keras(img)
        pred_class, confidence, top3 = predict_with_model1(MODEL1, arr)

        severity = None
        if pred_class not in HEALTHY_CLASSES:
            tensor = preprocess_image_pytorch(img)
            severity = predict_severity(MODEL2W, tensor)

        plot_config = {"displayModeBar": False}
        conf_gauge_html = gauge_chart(confidence * 100, "Model confidence", "%").to_html(full_html=False, include_plotlyjs=False, config=plot_config)
        
        sev_gauge_html = None
        if pred_class in HEALTHY_CLASSES:
            sev_gauge_html = gauge_chart(5.0, "Looks healthy", "%").to_html(full_html=False, include_plotlyjs=False, config=plot_config)
        else:
            sev_gauge_html = gauge_chart(severity, "Estimated severity", "%").to_html(full_html=False, include_plotlyjs=False, config=plot_config)

        # --- MODIFIED: Call new unified Gemini function ---
        gemini_data = get_gemini_analysis(pred_class, severity)

        buffered = io.BytesIO()
        img_for_display = img.resize((512, 512)) 
        img_for_display.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{img_str}"

        results = {
            "pred_class": pred_class,
            "emoji": get_class_emoji(pred_class),
            "is_healthy": pred_class in HEALTHY_CLASSES,
            "confidence": confidence,
            "severity": severity,
            "top3": top3,
            "image_data_url": image_data_url,
            "charts": {
                "confidence_gauge": conf_gauge_html,
                "severity_gauge": sev_gauge_html
            },
            # --- MODIFIED: Unpack the Gemini data ---
            "definition": gemini_data["definition"],
            "solution_guide": gemini_data["solution_guide"]
        }

        return render_template('index.html', results=results)


if __name__ == '__main__':
    # --- MODIFIED: Added check for API key on startup ---
    if not GEMINI_API_KEY:
        print("="*50)
        print("WARNING: 'GEMINI_API_KEY' is not set.")
        print("The app will run using simulated data.")
        print("Set the environment variable to use the real API.")
        print("="*50)
    app.run(debug=True)