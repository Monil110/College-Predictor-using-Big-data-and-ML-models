import React, { useState } from 'react';

// ─── Rank validation ──────────────────────────────────────────────────────────
function validateRank(rawValue, label, max) {
  const str = String(rawValue).trim();
  if (!str) return `${label} is required.`;
  if (/^0+$/.test(str)) return `${label} cannot be zero. Please enter a valid rank.`;
  const parsed = parseInt(str, 10);
  if (isNaN(parsed) || !Number.isFinite(parsed)) return `${label} must be a valid number.`;
  if (parsed <= 0) return `${label} must be greater than 0.`;
  if (max && parsed > max) return `${label} seems too high (max ${max.toLocaleString()}). Please verify.`;
  return null;
}

const RANK_LIMITS = {
  JEE_ADVANCED: 200000,
  JEE_MAIN: 1200000,
  NEET_UG: 2000000,
  NEET_PG: 200000,
  KCET: 200000,
  COMEDK: 200000,
};

const DOMAINS = [
  { value: 'JEE',     label: 'JEE',      icon: '⚙️',  desc: 'IITs / NITs / IIITs' },
  { value: 'KCET',    label: 'KCET',     icon: '🏛️',  desc: 'Karnataka Engineering' },
  { value: 'COMEDK',  label: 'COMEDK',   icon: '🎓',  desc: 'Karnataka Private' },
  { value: 'NEET_UG', label: 'NEET UG',  icon: '🩺',  desc: 'MBBS / BDS Colleges' },
  { value: 'NEET_PG', label: 'NEET PG',  icon: '🏥',  desc: 'PG Medical Colleges' },
];

const PredictionForm = ({ onPredict, isLoading }) => {
  const [domain, setDomain] = useState('JEE');
  const [rankError, setRankError] = useState('');

  const [jeeData, setJeeData] = useState({
    user_rank: '', exam_type: 'JEE Advanced', category: 'GEN', quota: 'AI', pool: 'Gender-Neutral',
  });
  const [neetUgData, setNeetUgData] = useState({ user_rank: '', category: 'OPEN SEAT' });
  const [neetPgData, setNeetPgData] = useState({ user_rank: '', category: 'GM' });
  const [kcetData, setKcetData] = useState({
    user_rank: '', category: 'GM', base_category: 'GM', quota: 'General', region: 'General',
  });
  const [comedkData, setComedkData] = useState({ user_rank: '', category: 'GM' });

  const handleDomainChange = (val) => { setDomain(val); setRankError(''); };

  const mkChange = (setter, state) => (e) => {
    setter({ ...state, [e.target.name]: e.target.value });
    if (e.target.name === 'user_rank') setRankError('');
  };

  // JEE Advanced (IITs) only has an "All India" quota; JEE Main (NITs) never
  // has "All India" — it's Home/Other State. Reset quota when the exam mode
  // changes so the two fields can't drift into a combination the backend
  // (correctly) rejects.
  const handleExamTypeChange = (e) => {
    const exam_type = e.target.value;
    const quota = exam_type === 'JEE Advanced' ? 'AI' : 'OS';
    setJeeData({ ...jeeData, exam_type, quota });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    let rawRank, rankLabel, rankMax;
    if (domain === 'JEE') {
      rawRank = jeeData.user_rank; rankLabel = 'JEE Rank';
      rankMax = jeeData.exam_type === 'JEE Advanced' ? RANK_LIMITS.JEE_ADVANCED : RANK_LIMITS.JEE_MAIN;
    } else if (domain === 'NEET_UG') {
      rawRank = neetUgData.user_rank; rankLabel = 'NEET UG Rank'; rankMax = RANK_LIMITS.NEET_UG;
    } else if (domain === 'NEET_PG') {
      rawRank = neetPgData.user_rank; rankLabel = 'NEET PG Rank'; rankMax = RANK_LIMITS.NEET_PG;
    } else if (domain === 'KCET') {
      rawRank = kcetData.user_rank; rankLabel = 'KCET Rank'; rankMax = RANK_LIMITS.KCET;
    } else {
      rawRank = comedkData.user_rank; rankLabel = 'COMEDK Rank'; rankMax = RANK_LIMITS.COMEDK;
    }

    const err = validateRank(rawRank, rankLabel, rankMax);
    if (err) { setRankError(err); return; }
    setRankError('');
    const rank = parseInt(String(rawRank).trim(), 10);

    if (domain === 'JEE')
      onPredict({ ...jeeData, user_rank: rank, domain: 'JEE' });
    else if (domain === 'NEET_UG')
      onPredict({ candidate_rank: rank, category: neetUgData.category, domain: 'NEET_UG' });
    else if (domain === 'NEET_PG')
      onPredict({ candidate_rank: rank, category: neetPgData.category, domain: 'NEET_PG' });
    else if (domain === 'KCET')
      onPredict({ ...kcetData, user_rank: rank, domain: 'KCET' });
    else
      onPredict({ ...comedkData, user_rank: rank, domain: 'COMEDK' });
  };

  return (
    <div className="form-card">
      {/* Domain selector */}
      <div className="domain-selector">
        {DOMAINS.map(d => (
          <button
            key={d.value}
            type="button"
            className={`domain-btn ${domain === d.value ? 'domain-btn-active' : ''}`}
            onClick={() => handleDomainChange(d.value)}
          >
            <span className="domain-btn-icon">{d.icon}</span>
            <span className="domain-btn-label">{d.label}</span>
            <span className="domain-btn-desc">{d.desc}</span>
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} noValidate>
        <div className="fields-grid">

          {/* ── JEE ── */}
          {domain === 'JEE' && <>
            <div className="field-group">
              <label className="field-label">Your JEE Rank</label>
              <input
                type="text" inputMode="numeric" name="user_rank"
                value={jeeData.user_rank} onChange={mkChange(setJeeData, jeeData)}
                placeholder="e.g. 2500" className={`field-input ${rankError ? 'field-input-error' : ''}`}
              />
              {rankError && <span className="field-error-msg">⚠ {rankError}</span>}
            </div>
            <div className="field-group">
              <label className="field-label">Exam Mode</label>
              <select name="exam_type" value={jeeData.exam_type} onChange={handleExamTypeChange} className="field-input">
                <option value="JEE Advanced">JEE Advanced (IITs)</option>
                <option value="JEE Main">JEE Main (NITs &amp; IIITs)</option>
              </select>
            </div>
            <div className="field-group">
              <label className="field-label">Category</label>
              <select name="category" value={jeeData.category} onChange={mkChange(setJeeData, jeeData)} className="field-input">
                <option value="GEN">OPEN (General)</option>
                <option value="OBC-NCL">OBC-NCL</option>
                <option value="SC">SC</option>
                <option value="ST">ST</option>
                <option value="GEN-EWS">GEN-EWS</option>
              </select>
            </div>
            <div className="field-group">
              <label className="field-label">Quota</label>
              {jeeData.exam_type === 'JEE Advanced' ? (
                <select name="quota" value={jeeData.quota} onChange={mkChange(setJeeData, jeeData)} className="field-input">
                  <option value="AI">All India</option>
                </select>
              ) : (
                <select name="quota" value={jeeData.quota} onChange={mkChange(setJeeData, jeeData)} className="field-input">
                  <option value="OS">Other State</option>
                  <option value="HS">Home State</option>
                </select>
              )}
            </div>
            <div className="field-group">
              <label className="field-label">Gender Pool</label>
              <select name="pool" value={jeeData.pool} onChange={mkChange(setJeeData, jeeData)} className="field-input">
                <option value="Gender-Neutral">Gender-Neutral</option>
                <option value="Female-Only">Female-Only</option>
              </select>
            </div>
          </>}

          {/* ── NEET UG ── */}
          {domain === 'NEET_UG' && <>
            <div className="field-group">
              <label className="field-label">Your NEET UG Rank</label>
              <input
                type="text" inputMode="numeric" name="user_rank"
                value={neetUgData.user_rank} onChange={mkChange(setNeetUgData, neetUgData)}
                placeholder="e.g. 5000" className={`field-input ${rankError ? 'field-input-error' : ''}`}
              />
              {rankError && <span className="field-error-msg">⚠ {rankError}</span>}
            </div>
            <div className="field-group">
              <label className="field-label">Seat Category</label>
              <select name="category" value={neetUgData.category} onChange={mkChange(setNeetUgData, neetUgData)} className="field-input">
                <option value="OPEN SEAT">Open Seat (All India Open)</option>
                <option value="ALL INDIA">All India Quota</option>
                <option value="DEEMED/PAID">Deemed / Paid Seat</option>
                <option value="EMPLOYEES">Employees / ESIC Quota</option>
                <option value="DELHI">Delhi Regional</option>
                <option value="MUSLIM">Minority (Muslim)</option>
                <option value="NON-RESIDENT">NRI / Non-Resident</option>
              </select>
            </div>
          </>}

          {/* ── NEET PG ── */}
          {domain === 'NEET_PG' && <>
            <div className="field-group">
              <label className="field-label">Your NEET PG Rank</label>
              <input
                type="text" inputMode="numeric" name="user_rank"
                value={neetPgData.user_rank} onChange={mkChange(setNeetPgData, neetPgData)}
                placeholder="e.g. 1500" className={`field-input ${rankError ? 'field-input-error' : ''}`}
              />
              {rankError && <span className="field-error-msg">⚠ {rankError}</span>}
            </div>
            <div className="field-group">
              <label className="field-label">Seat Category</label>
              <select name="category" value={neetPgData.category} onChange={mkChange(setNeetPgData, neetPgData)} className="field-input">
                <optgroup label="General Merit">
                  <option value="GM">GM — General Merit</option>
                  <option value="GMH">GMH — General Merit (HK Region)</option>
                  <option value="GMP">GMP — General Merit (PH)</option>
                  <option value="GMPH">GMPH — General Merit (PH, HK)</option>
                  <option value="OPN">OPN — Open (Deemed/Paid)</option>
                </optgroup>
                <optgroup label="OBC Categories">
                  <option value="1G">1G — Category 1</option>
                  <option value="1H">1H — Category 1 (HK)</option>
                  <option value="2AG">2AG — Category 2A</option>
                  <option value="2AH">2AH — Category 2A (HK)</option>
                  <option value="2BG">2BG — Category 2B</option>
                  <option value="2BH">2BH — Category 2B (HK)</option>
                  <option value="3AG">3AG — Category 3A</option>
                  <option value="3AH">3AH — Category 3A (HK)</option>
                  <option value="3BG">3BG — Category 3B</option>
                  <option value="3BH">3BH — Category 3B (HK)</option>
                </optgroup>
                <optgroup label="SC / ST">
                  <option value="S1G">S1G — SC (Left Hand)</option>
                  <option value="S1H">S1H — SC (Left Hand, HK)</option>
                  <option value="S2G">S2G — SC (Right Hand)</option>
                  <option value="S2H">S2H — SC (Right Hand, HK)</option>
                  <option value="S3G">S3G — SC (Others)</option>
                  <option value="S3H">S3H — SC (Others, HK)</option>
                  <option value="STG">STG — ST</option>
                  <option value="STH">STH — ST (HK)</option>
                </optgroup>
                <optgroup label="Minority / Management / NRI">
                  <option value="MM">MM — Muslim Minority</option>
                  <option value="MMH">MMH — Muslim Minority (HK)</option>
                  <option value="MC">MC — Christian Minority</option>
                  <option value="MU">MU — Urdu Minority</option>
                  <option value="MA">MA — Anglo-Indian Minority</option>
                  <option value="MNG">MNG — Lingayat Minority</option>
                  <option value="ME">ME — Management / Paid</option>
                  <option value="MEH">MEH — Management / Paid (HK)</option>
                  <option value="NRI">NRI — Non-Resident Indian</option>
                </optgroup>
                <optgroup label="Postgraduate / Special">
                  <option value="PGM">PGM — PG Management</option>
                  <option value="PGMH">PGMH — PG Management (HK)</option>
                  <option value="P1G">P1G — PG Category 1</option>
                  <option value="P2AG">P2AG — PG Category 2A</option>
                  <option value="PSTG">PSTG — PG ST</option>
                  <option value="RC1">RC1 — Rural Candidate 1</option>
                  <option value="RC2">RC2 — Rural Candidate 2</option>
                  <option value="RC3">RC3 — Rural Candidate 3</option>
                </optgroup>
              </select>
            </div>
          </>}

          {/* ── KCET ── */}
          {domain === 'KCET' && <>
            <div className="field-group">
              <label className="field-label">Your KCET Rank</label>
              <input
                type="text" inputMode="numeric" name="user_rank"
                value={kcetData.user_rank} onChange={mkChange(setKcetData, kcetData)}
                placeholder="e.g. 25000" className={`field-input ${rankError ? 'field-input-error' : ''}`}
              />
              {rankError && <span className="field-error-msg">⚠ {rankError}</span>}
            </div>
            <div className="field-group">
              <label className="field-label">Rank Category</label>
              <select name="category" value={kcetData.category} onChange={mkChange(setKcetData, kcetData)} className="field-input">
                <option value="GM">GM</option>
                <option value="1G">1G</option>
                <option value="2AG">2AG</option>
                <option value="2BG">2BG</option>
                <option value="3AG">3AG</option>
                <option value="3BG">3BG</option>
                <option value="SCG">SCG</option>
                <option value="STG">STG</option>
                <option value="GMK">GMK (Kannada)</option>
                <option value="GMR">GMR (Rural)</option>
              </select>
            </div>
            <div className="field-group">
              <label className="field-label">Base Caste Category</label>
              <select name="base_category" value={kcetData.base_category} onChange={mkChange(setKcetData, kcetData)} className="field-input">
                <option value="GM">GM</option>
                <option value="1">1</option>
                <option value="2A">2A</option>
                <option value="2B">2B</option>
                <option value="3A">3A</option>
                <option value="3B">3B</option>
                <option value="SC">SC</option>
                <option value="ST">ST</option>
              </select>
            </div>
            <div className="field-group">
              <label className="field-label">Quota</label>
              <select name="quota" value={kcetData.quota} onChange={mkChange(setKcetData, kcetData)} className="field-input">
                <option value="General">General</option>
                <option value="Kannada">Kannada</option>
                <option value="Rural">Rural</option>
              </select>
            </div>
            <div className="field-group">
              <label className="field-label">Region</label>
              <select name="region" value={kcetData.region} onChange={mkChange(setKcetData, kcetData)} className="field-input">
                <option value="General">General Range</option>
                <option value="Hyderabad-Karnataka">Hyderabad-Karnataka (HK)</option>
              </select>
            </div>
          </>}

          {/* ── COMEDK ── */}
          {domain === 'COMEDK' && <>
            <div className="field-group">
              <label className="field-label">Your COMEDK Rank</label>
              <input
                type="text" inputMode="numeric" name="user_rank"
                value={comedkData.user_rank} onChange={mkChange(setComedkData, comedkData)}
                placeholder="e.g. 15000" className={`field-input ${rankError ? 'field-input-error' : ''}`}
              />
              {rankError && <span className="field-error-msg">⚠ {rankError}</span>}
            </div>
            <div className="field-group">
              <label className="field-label">Category</label>
              <select name="category" value={comedkData.category} onChange={mkChange(setComedkData, comedkData)} className="field-input">
                <option value="GM">General Merit (GM)</option>
                <option value="KKR">Kalyana Karnataka Region (KKR)</option>
              </select>
            </div>
          </>}

        </div>

        <button type="submit" className="predict-btn" disabled={isLoading}>
          {isLoading
            ? <><span className="btn-spinner" /> Analyzing...</>
            : <><span>🔍</span> Predict My Colleges</>
          }
        </button>
      </form>
    </div>
  );
};

export default PredictionForm;
