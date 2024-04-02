from flask import  jsonify, request, Flask
from flask_cors import CORS,cross_origin

from brain_tumor_lib import *

# Create a Flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Define a route and its handler
@app.route('/')
@cross_origin()
def intro():
    return 'Brain Tumor Detection!'

@app.route("/get-prediction", methods=["POST"])
@cross_origin()
def get_predictions():
    body = request.json
    if not body:
        return jsonify({"message": "Cannot decode json from body"}), 422
    else:
        print("RECEIVED REQUEST")
        image = body.get("image")
        patient_name = body.get("name") 
        if not image:
            return jsonify({"message": "image not found"}), 422
        if not patient_name:
            return jsonify({"message": "patient name not found"}), 422

        image_segmented_base64, prediction, tumor_description = brain_tumor_predict(image, patient_name)

        final_json = jsonify({"segmentedImage": image_segmented_base64,
                        "prediction": prediction,
                        "description": tumor_description
                        })

        return final_json, 200
    
@app.route("/get-all-records", methods = ["POST"])
@cross_origin()
def get_past_data():
    body = request.json
    if not body:
        return jsonify({"message": "Cannot decode json from body"}), 422
    else:
        history = get_all_records()
    return history

@app.route("/get-patient-history", methods=["POST"])
@cross_origin()
def get_patient_history():
    body = request.json
    if not body or "name" not in body:
        return jsonify({"message": "Cannot decode json from body or missing 'name' field"}), 422
    
    patient_name = body.get("name")
    patient_history = fetch_patient_history(patient_name)
    if patient_history:
        return jsonify(patient_history)
    else:
        return jsonify({"message": "Patient history not found"}), 404

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)