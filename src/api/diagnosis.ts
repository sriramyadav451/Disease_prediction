interface Disease {
    disease: string;
    confidence: number;
  }
  
  export interface DiagnosisResponse {
    results: Disease[];
  }
  
  export const predictDisease = async (symptoms: string): Promise<DiagnosisResponse> => {
    const response = await fetch(`http://localhost:8000/predict?symptoms=${encodeURIComponent(symptoms)}`);
    if (!response.ok) {
      throw new Error('Failed to fetch diagnosis');
    }
    return response.json();
  };