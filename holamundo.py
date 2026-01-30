from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from flask_cors import CORS



# =========================
# CONFIG
# =========================
MODEL_PATH = "model_c2f.tflite"

app = Flask(__name__)

# =========================
# ENABLE CORS
# =========================
CORS(app)  # <-- ESTA LÍNEA ES CLAVE

# =========================
# LOAD TFLITE MODEL
# =========================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return "Flask + TFLite API running"

@app.route("/predict/<float:celsius>", methods=["POST"])
def predict_get(celsius):

    input_data = np.array([[celsius]], dtype=np.float32)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    fahrenheit = float(output[0][0])

    return jsonify({
        "celsius": celsius,
        "fahrenheit": fahrenheit
    })


    # =========================
    # PREPARE INPUT
    # shape: (1,1)
    # =========================
    input_data = np.array([[celsius]], dtype=np.float32)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    fahrenheit = float(output[0][0])

    return jsonify({
        "celsius": celsius,
        "fahrenheit": fahrenheit
    })

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(debug=True)

