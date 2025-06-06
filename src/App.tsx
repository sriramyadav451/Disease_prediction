import React, { useState } from 'react';
import { Moon, Sun, Loader2, AlertCircle, ThumbsUp, ChevronRight, Activity, Shield, Pill } from 'lucide-react';
import { predictDisease, getDiseaseDetails, DiseasePrediction, DiseaseDetails } from './services/diagnosis';

function App() {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [symptoms, setSymptoms] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<DiseaseDetails[]>([]);
  const [selectedDisease, setSelectedDisease] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    try {
      setIsLoading(true);
      setResults([]);
      setSelectedDisease(null);
      setError(null);
      
      const predictions = await predictDisease(symptoms);
      
      const detailedResults = await Promise.all(
        predictions.map(async (prediction) => {
          const details = await getDiseaseDetails(prediction.disease);
          return {
            name: details.disease,
            confidence: prediction.confidence,
            prevention: details.prevention,
            cure: details.treatment
          };
        })
      );
      
      setResults(detailedResults);
    } catch (error) {
      console.error('Prediction error:', error);
      setError('Failed to get diagnosis. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDiseaseClick = async (diseaseName: string) => {
    if (selectedDisease === diseaseName) {
      setSelectedDisease(null);
      return;
    }
    
    setSelectedDisease(diseaseName);
    
    const disease = results.find(d => d.name === diseaseName);
    if (!disease?.prevention) {
      try {
        const details = await getDiseaseDetails(diseaseName);
        setResults(prev => 
          prev.map(d => 
            d.name === diseaseName 
              ? { ...d, prevention: details.prevention, cure: details.treatment } 
              : d
          )
        );
      } catch (error) {
        console.error('Failed to load details:', error);
      }
    }
  };

  return (
    <div className={`min-h-screen transition-colors duration-300 ${isDarkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="flex justify-between items-center mb-12">
          <div className="flex items-center space-x-3">
            <Activity className="w-8 h-8 text-blue-500" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 bg-clip-text text-transparent">
              Disease Diagnosis
            </h1>
          </div>
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className={`p-2 rounded-full transition-all duration-300 transform hover:scale-110 ${
              isDarkMode ? 'bg-gray-800 hover:bg-gray-700' : 'bg-gray-200 hover:bg-gray-300'
            }`}
          >
            {isDarkMode ? <Sun className="w-6 h-6" /> : <Moon className="w-6 h-6" />}
          </button>
        </div>

        {/* Input Section */}
        <div className={`p-6 rounded-xl shadow-lg mb-8 transition-colors duration-300 backdrop-blur-sm ${
          isDarkMode ? 'bg-gray-800/90' : 'bg-white/90'
        }`}>
          <div className="mb-6">
            <label className="block mb-2 text-lg font-medium">Enter Your Symptoms</label>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
              List your symptoms separated by commas (e.g., fever, cough, headache)
            </p>
            <input
              type="text"
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
              className={`w-full p-4 rounded-lg border transition-colors duration-300 ${
                isDarkMode 
                  ? 'bg-gray-700 border-gray-600 focus:border-blue-500 text-white' 
                  : 'bg-gray-50 border-gray-300 focus:border-blue-500'
              } focus:ring-2 focus:ring-blue-500 focus:outline-none`}
              placeholder="fever, cough, headache..."
            />
          </div>
          <button
            onClick={handlePredict}
            disabled={isLoading || !symptoms.trim()}
            className={`px-6 py-3 rounded-lg font-medium flex items-center justify-center space-x-2 w-full sm:w-auto
              transition-all duration-300 transform hover:scale-105 ${
              isLoading || !symptoms.trim()
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white shadow-lg hover:shadow-xl'
            }`}
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <>
                <AlertCircle className="w-5 h-5" />
                <span>Analyze Symptoms</span>
              </>
            )}
          </button>
        </div>

        {/* Error Message */}
        {error && (
          <div className={`p-4 mb-6 rounded-lg ${
            isDarkMode ? 'bg-red-900/50 text-red-200' : 'bg-red-100 text-red-800'
          }`}>
            {error}
          </div>
        )}

        {/* Results Section */}
        {results.length > 0 && (
          <div className={`p-6 rounded-xl shadow-lg transition-colors duration-300 ${
            isDarkMode ? 'bg-gray-800' : 'bg-white'
          }`}>
            <h2 className="text-2xl font-bold mb-6 flex items-center">
              <Activity className="w-6 h-6 mr-2 text-blue-500" />
              Diagnosis Results
            </h2>
            <div className="space-y-4">
              {results.map((disease) => (
                <div
                  key={disease.name}
                  className={`rounded-lg transition-all duration-300 overflow-hidden ${
                    selectedDisease === disease.name
                      ? isDarkMode ? 'bg-blue-900/50' : 'bg-blue-50'
                      : isDarkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-50 hover:bg-gray-100'
                  }`}
                >
                  <div 
                    className="p-4 cursor-pointer flex items-center justify-between transform hover:scale-102"
                    onClick={() => handleDiseaseClick(disease.name)}
                  >
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-full ${
                        disease.confidence > 80 ? 'bg-green-500/20' :
                        disease.confidence > 60 ? 'bg-yellow-500/20' : 'bg-red-500/20'
                      }`}>
                        <ThumbsUp className={`w-5 h-5 ${
                          disease.confidence > 80 ? 'text-green-500' :
                          disease.confidence > 60 ? 'text-yellow-500' : 'text-red-500'
                        }`} />
                      </div>
                      <h3 className="text-lg font-medium">{disease.name}</h3>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                        disease.confidence > 80 ? 'bg-green-500/20 text-green-500' :
                        disease.confidence > 60 ? 'bg-yellow-500/20 text-yellow-500' : 'bg-red-500/20 text-red-500'
                      }`}>
                        {disease.confidence}% Match
                      </div>
                      <ChevronRight className={`w-5 h-5 transition-transform duration-300 ${
                        selectedDisease === disease.name ? 'rotate-90' : ''
                      }`} />
                    </div>
                  </div>
                  
                  {selectedDisease === disease.name && (
                    <div className="px-4 pb-4 animate-fadeIn">
                      <div className="border-t border-gray-200 dark:border-gray-600 pt-4 space-y-4">
                        <div className="space-y-2">
                          <div className="flex items-center space-x-2 text-blue-500">
                            <Shield className="w-5 h-5" />
                            <h4 className="font-medium">Prevention:</h4>
                          </div>
                          <ul className="pl-7 space-y-1 list-disc">
                            {disease.prevention?.map((point, index) => (
                              <li key={index}>{point.replace(/^- /, '').trim()}</li>
                            ))}
                          </ul>
                        </div>
                        <div className="space-y-2">
                          <div className="flex items-center space-x-2 text-green-500">
                            <Pill className="w-5 h-5" />
                            <h4 className="font-medium">Treatment:</h4>
                          </div>
                          <ul className="pl-7 space-y-1 list-disc">
                            {disease.cure?.map((point, index) => (
                              <li key={index}>{point.replace(/^- /, '').trim()}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;