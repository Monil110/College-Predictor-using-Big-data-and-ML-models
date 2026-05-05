import React, { useState } from 'react'
import axios from 'axios'
import PredictionForm from './components/PredictionForm'
import ResultsTable from './components/ResultsTable'
import './index.css'

function App() {
  const [results, setResults] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  const BASE_URL = 'https://college-predictor-using-big-data-and-ml.onrender.com'

  const handlePredict = async (data) => {
    setIsLoading(true)
    setResults(null)

    try {
      const { domain, ...payload } = data

      let endpoint = `${BASE_URL}/predict`

      if (domain === 'NEET') {
        endpoint = `${BASE_URL}/predict/neet`
      }
      else if (domain === 'KCET') {
        endpoint = `${BASE_URL}/predict/kcet`
      }
      else if (domain === 'COMEDK') {
        endpoint = `${BASE_URL}/predict/comedk`
      }

      const response = await axios.post(endpoint, payload)
      setResults(response.data.data)

    } catch (error) {
      console.error("API Error:", error)
      alert("Failed to connect to deployed backend.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-container">
      <div className="header">
        <h1>PredictMe</h1>
        <p>Big Data & ML Powered Admission Intelligence</p>
      </div>

      <PredictionForm onPredict={handlePredict} isLoading={isLoading} />

      {isLoading && (
        <div className="loading">
          Crunching 90M+ Historical Rows via PySpark & XGBoost/CatBoost...
        </div>
      )}

      {!isLoading && results && <ResultsTable results={results} />}
    </div>
  )
}

export default App