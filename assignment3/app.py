"""
ML Deployment - Assignment 3
Flask Prediction API

Endpoints:
  GET  /                  -> Welcome message (Q1)
  POST /predict-price     -> Calculate discounted price (Q2)
  POST /predict-delivery  -> Predict delivery time using ML model (Q3 + Q4)
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# --- Load the trained delivery model at startup (Q3) ---
delivery_model = joblib.load('assignment3/delivery_model.joblib')


# =============================================
# Q1: Root URL - Welcome Message
# =============================================
@app.route('/')
def home():
    return 'Welcome to the Prediction API'


# =============================================
# Q2: Predict Price after Discount
# =============================================
@app.route('/predict-price', methods=['POST'])
def predict_price():
    data = request.get_json()

    # Validate input
    if not data or 'base_price' not in data or 'discount' not in data:
        return jsonify({
            'error': 'Please provide both "base_price" and "discount" in JSON body'
        }), 400

    base_price = data['base_price']
    discount = data['discount']

    # Calculate final price
    final_price = base_price - (base_price * discount / 100)

    return jsonify({
        'base_price': base_price,
        'discount_percent': discount,
        'final_price': round(final_price, 2)
    })


# =============================================
# Q3 + Q4: Predict Delivery Time with ML Model
#           Returns prediction + friendly message
# =============================================
@app.route('/predict-delivery', methods=['POST'])
def predict_delivery():
    data = request.get_json()

    # Validate input
    if not data or 'distance_km' not in data or 'order_size' not in data:
        return jsonify({
            'error': 'Please provide "distance_km" and "order_size" in JSON body'
        }), 400

    distance_km = data['distance_km']
    order_size = data['order_size']

    # Make prediction using the loaded model
    input_features = np.array([[distance_km, order_size]])
    predicted_time = delivery_model.predict(input_features)[0]
    predicted_time = round(predicted_time, 1)

    # Q4: Return JSON with predicted value + custom friendly message
    # (Similar to Swiggy/Zomato estimated delivery time)
    if predicted_time <= 20:
        message = f"🚀 Lightning fast! Your order will arrive in ~{predicted_time} mins."
    elif predicted_time <= 35:
        message = f"🛵 On its way! Estimated delivery in ~{predicted_time} mins. Hang tight!"
    else:
        message = f"📦 Your order is being prepared. Expected delivery in ~{predicted_time} mins. Thanks for your patience!"

    return jsonify({
        'distance_km': distance_km,
        'order_size': order_size,
        'predicted_delivery_time_min': predicted_time,
        'message': message
    })


# =============================================
# Run the app
# =============================================
if __name__ == '__main__':
    print("\n--- Flask Prediction API ---")
    print("Endpoints:")
    print("  GET  http://127.0.0.1:5000/")
    print("  POST http://127.0.0.1:5000/predict-price")
    print("  POST http://127.0.0.1:5000/predict-delivery")
    print("-" * 40)
    app.run(debug=True)
