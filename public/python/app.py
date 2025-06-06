from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.SymptomSuggestion import predict_disease
import google.generativeai as genai
import os
from pathlib import Path
import logging
import json
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent.parent / 'data'

# Configure Gemini API
try:
    genai.configure(api_key='AIzaSyCTq5sFfuSWeyzQqZKO-jONuMmzQK7Yv50')
    gemini_model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config={
            "temperature": 0.5,
            "max_output_tokens": 800,
        }
    )
    logger.info("✅ Gemini API configured successfully")
except Exception as e:
    logger.error(f"❌ Gemini API configuration failed: {e}")
    gemini_model = None

def extract_bullet_points(text, section_name):
    points = []
    lines = text.split('\n')
    found_section = False
    
    for line in lines:
        line = line.strip()
        if section_name.lower() in line.lower():
            found_section = True
            continue
        if found_section and line.startswith('-'):
            points.append(line[1:].strip())
        elif found_section and not line:
            break
            
    return points[:5]

def get_detailed_treatment_info(disease):
    if not gemini_model:
        return None
        
    try:
        response = gemini_model.generate_content(
            f"Provide clinical guidance for {disease} with:\n"
            "5 PREVENTION methods (each starting with '-')\n"
            "5 TREATMENT methods (each starting with '-')\n"
            "Format exactly like this example:\n"
            "PREVENTION:\n"
            "- Wash hands frequently\n"
            "- Avoid sick people\n"
            "- Get vaccinated\n"
            "- Practice good hygiene\n"
            "- Boost immune system\n"
            "TREATMENT:\n"
            "- Get plenty of rest\n"
            "- Stay hydrated\n"
            "- Take fever reducers\n"
            "- Use cough medicine\n"
            "- See doctor if severe\n"
        )
        
        text = response.text
        prevention = []
        treatment = []
        
        if "PREVENTION:" in text:
            prevention_section = text.split("PREVENTION:")[1].split("TREATMENT:")[0] if "TREATMENT:" in text else text.split("PREVENTION:")[1]
            prevention = [line.strip() for line in prevention_section.split('\n') 
                         if line.strip().startswith('-')][:5]
        
        if "TREATMENT:" in text:
            treatment_section = text.split("TREATMENT:")[1]
            treatment = [line.strip() for line in treatment_section.split('\n') 
                        if line.strip().startswith('-')][:5]
        
        if not prevention or not treatment:
            prevention = extract_bullet_points(text, "prevention")
            treatment = extract_bullet_points(text, "treatment")
            
        default_prevention = [
            "- Wash hands frequently",
            "- Avoid close contact with sick individuals",
            "- Maintain a healthy immune system",
            "- Keep vaccinations up to date",
            "- Practice good respiratory hygiene"
        ]
        
        default_treatment = [
            "- Get plenty of rest",
            "- Stay well hydrated",
            "- Use over-the-counter symptom relief",
            "- Take prescribed medications",
            "- Monitor for complications"
        ]
        
        return {
            "prevention": prevention if len(prevention) == 5 else default_prevention,
            "treatment": treatment if len(treatment) == 5 else default_treatment
        }
        
    except Exception as e:
        logger.error(f"Gemini treatment error: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '').strip()
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
            
        logger.info(f"📊 Received symptoms: {symptoms}")
        
        predictions = predict_disease(symptoms)
        if not predictions or 'error' in predictions[0]:
            return jsonify({'error': 'Prediction failed', 'details': predictions}), 500
        
        # Filter out low confidence predictions
        filtered = [p for p in predictions if p.get('confidence', 0) >= 30]
        
        # Only normalize if we have multiple good predictions
        if len(filtered) > 1:
            total_confidence = sum(p['confidence'] for p in filtered)
            if total_confidence > 0:
                # Scale to sum to 100 while maintaining relative weights
                scale_factor = 100 / total_confidence
                for p in filtered:
                    p['confidence'] = min(99, round(p['confidence'] * scale_factor, 1))
                
                # Adjust last item to make sum exactly 100
                current_sum = sum(p['confidence'] for p in filtered)
                if current_sum != 100:
                    filtered[-1]['confidence'] = round(filtered[-1]['confidence'] + (100 - current_sum), 1)
        
        return jsonify(filtered[:3])
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'fallback': [
                {"disease": "Common Cold", "confidence": 65.0},
                {"disease": "Flu", "confidence": 50.0},
                {"disease": "Allergy", "confidence": 40.0}
            ]
        }), 500

@app.route('/details', methods=['POST'])
def details():
    try:
        data = request.get_json()
        disease = data.get('disease', '').strip()
        
        if not disease:
            return jsonify({'error': 'No disease provided'}), 400
            
        treatment_info = get_detailed_treatment_info(disease) or {
            'prevention': [
                "- Wash hands frequently with soap",
                "- Avoid touching face with unwashed hands",
                "- Disinfect frequently touched surfaces",
                "- Maintain healthy diet and exercise",
                "- Get adequate sleep and rest"
            ],
            'treatment': [
                "- Rest and stay hydrated",
                "- Use saline nasal spray or drops",
                "- Take acetaminophen for fever/pain",
                "- Use throat lozenges for sore throat",
                "- Consider steam inhalation for congestion"
            ]
        }
        
        return jsonify({
            'disease': disease,
            'prevention': treatment_info['prevention'],
            'treatment': treatment_info['treatment']
        })
        
    except Exception as e:
        logger.error(f"Details error: {e}")
        return jsonify({
            'error': str(e),
            'fallback': True
        }), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)