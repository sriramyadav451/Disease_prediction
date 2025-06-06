export interface DiseasePrediction {
    disease: string;
    confidence: number;
  }
  
  export interface DiseaseDetails {
    disease: string;
    confidence: number;
    cure: string;
    prevention: string;
  }
  
  export const predictDisease = async (symptoms: string): Promise<DiseasePrediction[]> => {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ symptoms }),
    });
    
    if (!response.ok) {
      throw new Error('Prediction failed');
    }
    
    return response.json();
  };
  
  export const getDiseaseDetails = async (disease: string): Promise<DiseaseDetails> => {
    const response = await fetch('http://localhost:5000/details', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ disease }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to fetch disease details');
    }
    
    return response.json();
  };