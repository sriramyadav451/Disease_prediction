import pandas as pd
import warnings
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from itertools import combinations
from sklearn.linear_model import LogisticRegression

warnings.simplefilter("ignore")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')

# Get absolute path to data files
BASE_DIR = Path(__file__).parent.parent.parent  # Goes up to 'public' level
DATA_PATH = BASE_DIR / 'data' / 'cleaned_symptom_disease.csv'

# Try loading dataset
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset loaded successfully from: {DATA_PATH}")
except Exception as e:
    print(f"❌ Error loading dataset from {DATA_PATH}: {e}")
    df = None

# Fallback to minimal dataset if main dataset fails
if df is None:
    print("⚠️ Using fallback minimal dataset")
    disease_symptom_map = {
        "Common Cold": ["cough", "sore throat", "runny nose", "sneezing"],
        "Influenza": ["fever", "headache", "muscle pain", "fatigue"],
        "Allergy": ["sneezing", "itchy eyes", "runny nose", "rash"],
        "Migraine": ["headache", "nausea", "sensitivity to light"],
        "Stomach Flu": ["diarrhea", "stomach pain", "nausea"]
    }
    all_symptoms = list({sym for syms in disease_symptom_map.values() for sym in syms})
else:
    # Extract symptoms from actual dataset
    disease_symptom_map = {}
    all_symptoms = set()
    
    for _, row in df.iterrows():
        symptoms = str(row["Symptoms"]).split(", ")
        disease_symptom_map[row["Disease"]] = symptoms
        all_symptoms.update(symptoms)

    all_symptoms = list(all_symptoms)

print(f"ℹ️ Total diseases: {len(disease_symptom_map)}")
print(f"ℹ️ Total symptoms: {len(all_symptoms)}")

# Symptom mapping for standardization
symptom_mapping = {
    "stomach pain": "stomach_pain",
    "sore throat": "sore_throat",
    "runny nose": "runny_nose",
    "muscle pain": "muscle_pain",
    "sensitivity to light": "light_sensitivity"
}

def map_symptoms(symptoms):
    """Map symptoms to standardized format."""
    return [symptom_mapping.get(sym.lower(), sym.lower()) for sym in symptoms]

def process_symptoms(symptoms):
    """Preprocess user input symptoms."""
    if not symptoms:
        return []

    # Standardize input
    user_symptoms = symptoms.lower().replace(", ", ",").split(',')
    processed = []
    
    for sym in user_symptoms:
        sym = sym.strip()
        # Lemmatize each word in multi-word symptoms
        words = [lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)]
        processed.append('_'.join(words))
    
    return map_symptoms(processed)

# Train model if we have proper dataset
model = None
if df is not None:
    try:
        # Prepare training data
        X = []
        Y = []
        
        for disease, symptoms in disease_symptom_map.items():
            row = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
            X.append(row)
            Y.append(disease)
        
        # Train logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X, Y)
        print("✅ Model trained successfully")
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        model = None

def predict_disease(symptoms):
    """Predict disease based on symptoms."""
    if not disease_symptom_map:
        return [{"error": "Diagnosis system not available"}]

    processed_symptoms = process_symptoms(symptoms)
    if not processed_symptoms:
        return [{"error": "No valid symptoms provided"}]

    print(f"🔍 Processing symptoms: {processed_symptoms}")

    # If using fallback dataset
    if df is None:
        matches = []
        for disease, disease_syms in disease_symptom_map.items():
            matched = len(set(processed_symptoms) & set(map_symptoms(disease_syms)))
            if matched > 0:
                confidence = min(100, matched * 30)  # Simple confidence calculation
                matches.append({
                    "disease": disease,
                    "confidence": confidence
                })
        return sorted(matches, key=lambda x: x["confidence"], reverse=True)[:3]

    # Full prediction with logistic regression
    if model:
        try:
            # Create feature vector
            sample_x = [1 if symptom in processed_symptoms else 0 for symptom in all_symptoms]
            
            # Get prediction probabilities
            prediction = model.predict_proba([sample_x])[0]
            diseases = model.classes_
            
            # Get top 3 predictions
            top_k = prediction.argsort()[-3:][::-1]
            results = [{
                "disease": diseases[i], 
                "confidence": round(prediction[i] * 100, 2)
            } for i in top_k]
            
            return sorted(results, key=lambda x: x['confidence'], reverse=True)
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return [{"error": "Prediction failed"}]

    # Final fallback
    return [{
        "disease": "Common Cold",
        "confidence": 75
    }, {
        "disease": "Allergy", 
        "confidence": 60
    }]