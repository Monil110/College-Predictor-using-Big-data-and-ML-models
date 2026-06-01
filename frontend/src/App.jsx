import React, { useState } from 'react'
import axios from 'axios'
import PredictionForm from './components/PredictionForm'
import ResultsTable from './components/ResultsTable'
import './index.css'

// In development (localhost), try local backend first with a short timeout.
// In production (Vercel/any non-localhost origin), go straight to Render.
const IS_LOCAL = window.location.hostname === 'localhost' ||
                 window.location.hostname === '127.0.0.1'

const RENDER_URL = 'https://college-predictor-using-big-data-and-ml.onrender.com'

const BACKENDS = IS_LOCAL
  ? ['http://localhost:8000', RENDER_URL]
  : [RENDER_URL]

async function postWithFallback(path, payload) {
  let lastError = null
  for (const base of BACKENDS) {
    const isLocal = base.includes('localhost')
    try {
      const res = await axios.post(`${base}${path}`, payload, {
        timeout: isLocal ? 4000 : 90000,  // 4s for local probe, 90s for Render cold start
      })
      return res
    } catch (err) {
      lastError = err
      // If it's a 4xx from the server (validation error etc.), don't try next backend
      if (err.response && err.response.status >= 400 && err.response.status < 500) {
        throw err
      }
      // Otherwise (network error, timeout, 5xx) try next backend
    }
  }
  throw lastError
}

function App() {
  const [results, setResults] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [apiError, setApiError] = useState(null)
  const [lastDomain, setLastDomain] = useState(null)

  const handlePredict = async (data) => {
    setIsLoading(true)
    setResults(null)
    setApiError(null)

    try {
      const { domain, ...payload } = data
      setLastDomain(domain)

      let endpoint = '/predict'
      if (domain === 'NEET')   endpoint = '/predict/neet'
      else if (domain === 'KCET')   endpoint = '/predict/kcet'
      else if (domain === 'COMEDK') endpoint = '/predict/comedk'

      const response = await postWithFallback(endpoint, payload)
      const responseData = response.data

      if (responseData.source === 'error') {
        setApiError(responseData.error || 'Prediction failed. Please try again.')
        setResults(null)
      } else {
        setResults(responseData.data)
      }
    } catch (error) {
      console.error('API Error:', error)
      if (error.response?.data?.detail) {
        // FastAPI validation error
        const detail = error.response.data.detail
        if (Array.isArray(detail)) {
          setApiError(detail.map(d => d.msg).join('; '))
        } else {
          setApiError(String(detail))
        }
      } else if (error.code === 'ECONNABORTED') {
        setApiError('Request timed out. The server may be waking up — please try again in a moment.')
      } else {
        setApiError('Could not connect to the prediction server. Please check your connection and try again.')
      }
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-container">
      <div className="header">
        <h1>PredictMe</h1>
        <p>Big Data &amp; ML Powered Admission Intelligence</p>
        <p className="header-sub">Powered by PySpark · CatBoost · 90M+ Historical Records</p>
      </div>

      <PredictionForm onPredict={handlePredict} isLoading={isLoading} />

      {isLoading && (
        <div className="loading">
          <div className="loading-spinner" />
          <span>Crunching historical data via PySpark &amp; CatBoost...</span>
        </div>
      )}

      {!isLoading && apiError && (
        <div className="error-panel glass-panel">
          <span className="error-icon">⚠️</span>
          <span>{apiError}</span>
        </div>
      )}

      {!isLoading && results && !apiError && (
        <ResultsTable results={results} domain={lastDomain} />
      )}
    </div>
  )
}

export default App
