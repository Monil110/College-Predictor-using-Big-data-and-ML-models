import React, { useState } from 'react'
import axios from 'axios'
import PredictionForm from './components/PredictionForm'
import ResultsTable from './components/ResultsTable'
import './index.css'

function App() {
  const [results, setResults] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  const handlePredict = async (data) => {
    setIsLoading(true);
    setResults(null);
    try {
      const { domain, ...payload } = data;
      // Cross-route seamlessly within the monolithic backend
      let endpoint = 'http://localhost:8000/predict';
      if (domain === 'NEET') {
        endpoint = 'http://localhost:8000/predict/neet';
      } else if (domain === 'KCET') {
        endpoint = 'http://localhost:8000/predict/kcet';
      }
      
      const response = await axios.post(endpoint, payload);
      setResults(response.data.data);
    } catch (error) {
      console.error("API Error:", error);
      alert("Failed to connect to the unified backend instance. Ensure it exists on 8000.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>PredictMe</h1>
        <p>Big Data & ML Powered Admission Intelligence</p>
      </div>
      
      <PredictionForm onPredict={handlePredict} isLoading={isLoading} />
      
      {isLoading && <div className="loading">Crunching 90M+ Historical Rows via PySpark & XGBoost/CatBoost...</div>}
      
      {!isLoading && results && <ResultsTable results={results} />}
    </div>
  )
}

export default App
