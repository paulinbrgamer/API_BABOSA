from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import base64
import gdown
import os

app = Flask(__name__)

# Habilitar CORS
CORS(app)

MODEL_URL = 'https://drive.google.com/uc?id=1J5h9BfADV2c8Sw7KOkHh0tBqU3j_bvlW'

def download_model():
    if not os.path.exists("model.h5"):
        print("Baixando o modelo do Google Drive...")
        gdown.download(MODEL_URL, "model.h5", quiet=False)
        print("Modelo baixado com sucesso.")

download_model()

# Carregar o modelo pré-treinado
model = load_model('model.h5')
class_names = ['healthy_leaf', 'rot', 'rust']
class_names_pt = ['folha_saudável', 'podre', 'ferrugem']


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("image")
    if data:
        img_data = base64.b64decode(data)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (244, 244))
        img_array = np.expand_dims(img, axis=0)
        img_array = img_array / 255.0

        preds = model.predict(img_array)
        predicted_class = np.argmax(preds, axis=1)
        predicted_label = class_names[predicted_class[0]]
        predicted_label_pt = class_names_pt[predicted_class[0]]

        return jsonify({
            "prediction_en": predicted_label,
            "prediction_pt": predicted_label_pt
        })
    
    else:
        return jsonify({"error": "Image not received"}), 400

if __name__ == "__main__":
    app.run(debug=True)
