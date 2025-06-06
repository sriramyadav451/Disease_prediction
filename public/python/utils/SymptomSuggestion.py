import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from pathlib import Path
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

# Common diseases with their typical symptoms
COMMON_DISEASES = {
    "Common Cold": ["cough", "sore_throat", "runny_nose", "sneezing", "congestion", "mild_fever"],
    "Flu": ["fever", "headache", "muscle_aches", "fatigue", "chills", "cough"],
    "Allergy": ["sneezing", "itchy_eyes", "runny_nose", "nasal_congestion"],
    "Sinusitis": ["facial_pain", "nasal_congestion", "headache", "postnasal_drip", "cough"],
    "Strep Throat": ["sore_throat", "fever", "swollen_tonsils", "white_patches"],
    "Bronchitis": ["cough", "mucus_production", "fatigue", "mild_fever", "chest_discomfort"]
}

# Symptom weights - higher means more specific to certain diseases
SYMPTOM_WEIGHTS = {
    # High-specificity symptoms
    "swollen_tonsils": 4.0,
    "white_patches_throat": 4.0,
    "chest_pain": 3.5,
    "shortness_of_breath": 3.5,
    
    # Moderate-specificity symptoms
    "fever": 2.5,
    "headache": 2.0,
    "muscle_aches": 2.0,
    "fatigue": 1.8,
    
    # Common symptoms
    "cough": 1.5,
    "sore_throat": 1.5,
    "runny_nose": 1.3,
    "sneezing": 1.2,
    
    # Default weight
    "_default": 1.0
}

def load_dataset():
    try:
        DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'cleaned_symptom_disease.csv'
        df = pd.read_csv(DATA_PATH)
        logger.info("✅ Dataset loaded successfully")
        
        disease_symptoms = defaultdict(list)
        symptom_diseases = defaultdict(list)
        
        for _, row in df.iterrows():
            disease = row["Disease"].strip().lower()
            symptoms = [s.strip().lower() for s in str(row["Symptoms"]).split(",")]
            disease_symptoms[disease] = symptoms
            for sym in symptoms:
                symptom_diseases[sym].append(disease)
        
        # Add common diseases if not in dataset
        for disease, symptoms in COMMON_DISEASES.items():
            dl = disease.lower()
            if dl not in disease_symptoms:
                disease_symptoms[dl] = [s.lower() for s in symptoms]
                for sym in symptoms:
                    symptom_diseases[sym.lower()].append(dl)
        
        return disease_symptoms, symptom_diseases
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        # Fallback to common diseases only
        disease_symptoms = {k.lower(): [s.lower() for s in v] for k,v in COMMON_DISEASES.items()}
        symptom_diseases = defaultdict(list)
        for disease, symptoms in disease_symptoms.items():
            for sym in symptoms:
                symptom_diseases[sym].append(disease)
        return disease_symptoms, symptom_diseases

disease_symptoms, symptom_diseases = load_dataset()

def process_symptoms(symptoms):
    """Normalize and standardize symptoms"""
    if not symptoms:
        return []
    
    processed = []
    for sym in symptoms.lower().replace(", ", ",").split(','):
        sym = sym.strip()
        if not sym:
            continue
        
        # Tokenize and lemmatize
        words = [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(sym)]
        processed_sym = '_'.join(words)
        
        # Common symptom normalizations
        if "headache" in processed_sym:
            processed_sym = "severe_headache" if "severe" in processed_sym else "headache"
        elif "fever" in processed_sym:
            processed_sym = "high_fever" if "high" in processed_sym or ">101" in processed_sym else "fever"
        elif "throat" in processed_sym and "sore" in processed_sym:
            processed_sym = "sore_throat"
        elif "nose" in processed_sym and "runny" in processed_sym:
            processed_sym = "runny_nose"
        
        processed.append(processed_sym)
    
    return processed

def calculate_confidence(matched_symptoms, disease_symptoms):
    """Calculate weighted confidence score"""
    total_weight = sum(SYMPTOM_WEIGHTS.get(s, 1.0) for s in disease_symptoms)
    matched_weight = sum(SYMPTOM_WEIGHTS.get(s, 1.0) for s in matched_symptoms)
    
    # Base confidence is percentage of matched symptom weight
    base_confidence = (matched_weight / total_weight) * 80  # Max 80% for partial matches
    
    # Bonus for matching high-specificity symptoms
    specificity_bonus = sum(SYMPTOM_WEIGHTS.get(s, 0) for s in matched_symptoms if SYMPTOM_WEIGHTS.get(s, 1.0) > 2.0)
    bonus = min(15, specificity_bonus * 2)  # Max 15% bonus
    
    return min(95, base_confidence + bonus)  # Cap at 95%

def predict_disease(symptoms):
    processed_symptoms = process_symptoms(symptoms)
    if not processed_symptoms:
        return [{"error": "No valid symptoms provided"}]

    results = []
    
    # First check for exact matches to common diseases
    for disease, symptoms in COMMON_DISEASES.items():
        disease_lower = disease.lower()
        if set(processed_symptoms) == set(s.lower() for s in symptoms):
            return [{
                "disease": disease,
                "confidence": 95.0,
                "match_type": "exact"
            }]
    
    # Then check for partial matches
    candidate_diseases = set()
    for sym in processed_symptoms:
        candidate_diseases.update(symptom_diseases.get(sym, []))
    
    for disease in candidate_diseases:
        disease_syms = disease_symptoms[disease]
        matched = [s for s in processed_symptoms if s in disease_syms]
        if matched:
            confidence = calculate_confidence(matched, disease_syms)
            results.append({
                "disease": disease.capitalize(),
                "confidence": round(confidence, 1),
                "match_type": "partial"
            })
    
    # Sort by confidence and return top 3
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Fallback to common illnesses if no good matches
    if not results or results[0]['confidence'] < 40:
        fallbacks = [
            {"disease": "Common Cold", "confidence": 65.0, "match_type": "fallback"},
            {"disease": "Flu", "confidence": 50.0, "match_type": "fallback"},
            {"disease": "Allergy", "confidence": 40.0, "match_type": "fallback"}
        ]
        return fallbacks
    
    return results[:3]