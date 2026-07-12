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
        timeout: isLocal ? 4000 : 90000,
      })
      return res
    } catch (err) {
      lastError = err
      if (err.response && err.response.status >= 400 && err.response.status < 500) {
        throw err
      }
    }
  }
  throw lastError
}

function App() {
  const [results, setResults] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [apiError, setApiError] = useState(null)
  const [lastDomain, setLastDomain] = useState(null)
  const [resultNote, setResultNote] = useState(null)

  const handlePredict = async (data) => {
    setIsLoading(true)
    setResults(null)
    setApiError(null)
    setResultNote(null)

    try {
      const { domain, ...payload } = data
      setLastDomain(domain)

      let endpoint = '/predict'
      if (domain === 'NEET_UG')     endpoint = '/predict/neet'
      else if (domain === 'NEET_PG') endpoint = '/predict/neet_pg'
      else if (domain === 'KCET')   endpoint = '/predict/kcet'
      else if (domain === 'COMEDK') endpoint = '/predict/comedk'

      const response = await postWithFallback(endpoint, payload)
      const responseData = response.data

      if (responseData.source === 'error') {
        setApiError(responseData.error || 'Prediction failed. Please try again.')
        setResults(null)
      } else {
        // All endpoints now return the same shape: { source, data: { Safe: [], Likely: [] } }
        setResults(responseData.data)
        setResultNote(responseData.note || null)
      }
    } catch (error) {
      console.error('API Error:', error)
      if (error.response?.data?.detail) {
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
    <div className="app-wrapper">
      {/* Animated background orbs */}
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />
      <div className="bg-orb bg-orb-3" />

      <div className="app-container">

        {/* ── Hero Header ── */}
        <header className="hero-header">
          <div className="logo-img-wrap">
            <img src="/logo.png" alt="AdmitSense Logo" className="logo-img" />
          </div>
          <h1 className="hero-title">
            Admit<span className="hero-accent">Sense</span>
          </h1>
          <p className="hero-tagline">AI-Powered College Admission Intelligence</p>
          <p className="hero-sub">
            Powered by PySpark · CatBoost · 90M+ Historical Records
          </p>

          {/* Stats row */}
          <div className="stats-row">
            <div className="stat-item">
              <span className="stat-number">5</span>
              <span className="stat-label">Exams Covered</span>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <span className="stat-number">90M+</span>
              <span className="stat-label">Data Points</span>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <span className="stat-number">ML</span>
              <span className="stat-label">Powered</span>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <span className="stat-number">2026</span>
              <span className="stat-label">Predictions</span>
            </div>
          </div>
        </header>

        {/* ── Form ── */}
        <PredictionForm onPredict={handlePredict} isLoading={isLoading} />

        {/* ── Loading ── */}
        {isLoading && (
          <div className="loading-card">
            <div className="loading-dots">
              <span /><span /><span />
            </div>
            <p className="loading-title">Analyzing your profile...</p>
            <p className="loading-sub">Crunching historical data via PySpark &amp; CatBoost</p>
          </div>
        )}

        {/* ── Error ── */}
        {!isLoading && apiError && (
          <div className="error-card">
            <div className="error-card-icon">⚠️</div>
            <div>
              <p className="error-card-title">Something went wrong</p>
              <p className="error-card-msg">{apiError}</p>
            </div>
          </div>
        )}

        {!isLoading && results && !apiError && resultNote && (
          <div className="notice-card">
            <span className="notice-card-icon">ℹ️</span>
            <span>{resultNote}</span>
          </div>
        )}

        {!isLoading && results && !apiError && (
          <ResultsTable results={results} domain={lastDomain} />
        )}

        {/* ── Footer ── */}
        <footer className="app-footer">
          <p>AdmitSense &copy; {new Date().getFullYear()} &nbsp;·&nbsp; Built with Big Data &amp; ML</p>
        </footer>
      </div>
    </div>
  )
}

export default App
