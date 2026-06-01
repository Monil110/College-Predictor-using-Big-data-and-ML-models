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
  NEET: 2000000,
  KCET: 200000,
  COMEDK: 200000,
};

const DOMAINS = [
  { value: 'JEE',    label: 'JEE',    icon: '⚙️',  desc: 'IITs / NITs / IIITs' },
  { value: 'KCET',   label: 'KCET',   icon: '🏛️',  desc: 'Karnataka Engineering' },
  { value: 'COMEDK', label: 'COMEDK', icon: '🎓',  desc: 'Karnataka Private' },
  { value: 'NEET',   label: 'NEET',   icon: '🩺',  desc: 'Medical Colleges' },
];

const PredictionForm = ({ onPredict, isLoading }) => {
  const [domain, setDomain] = useState('JEE');
  const [rankError, setRankError] = useState('');

  const [jeeData, setJeeData] = useState({
    user_rank: '', exam_type: 'JEE Advanced', category: 'GEN', quota: 'AI', pool: 'Gender-Neutral',
  });
  const [neetData, setNeetData] = useState({ user_rank: '', category: 'OPEN SEAT' });
  const [kcetData, setKcetData] = useState({
    user_rank: '', category: 'GM', base_category: 'GM', quota: 'General', region: 'General',
  });
  const [comedkData, setComedkData] = useState({ user_rank: '', category: 'GM' });

  const handleDomainChange = (val) => { setDomain(val); setRankError(''); };

  const mkChange = (setter, state) => (e) => {
    setter({ ...state, [e.target.name]: e.target.value });
    if (e.target.name === 'user_rank') setRankError('');
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    let rawRank, rankLabel, rankMax;
    if (domain === 'JEE') {
      rawRank = jeeData.user_rank; rankLabel = 'JEE Rank';
      rankMax = jeeData.exam_type === 'JEE Advanced' ? RANK_LIMITS.JEE_ADVANCED : RANK_LIMITS.JEE_MAIN;
    } else if (domain === 'NEET') {
      rawRank = neetData.user_rank; rankLabel = 'NEET Rank'; rankMax = RANK_LIMITS.NEET;
    } else if (domain === 'KCET') {
      rawRank = kcetData.user_rank; rankLabel = 'KCET Rank'; rankMax = RANK_LIMITS.KCET;
    } else {
      rawRank = comedkData.user_rank; rankLabel = 'COMEDK Rank'; rankMax = RANK_LIMITS.COMEDK;
    }

    const err = validateRank(rawRank, rankLabel, rankMax);
    if (err) { setRankError(err); return; }
    setRankError('');
    const rank = parseInt(String(rawRank).trim(), 10);

    if (domain === 'JEE')         onPredict({ ...jeeData,    user_rank: rank, domain: 'JEE' });
    else if (domain === 'KCET')   onPredict({ ...kcetData,   user_rank: rank, domain: 'KCET' });
    else if (domain === 'COMEDK') onPredict({ ...comedkData, user_rank: rank, domain: 'COMEDK' });
    else                          onPredict({ ...neetData, candidate_rank: rank, domain: 'NEET' });
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
              <select name="exam_type" value={jeeData.exam_type} onChange={mkChange(setJeeData, jeeData)} className="field-input">
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
              <select name="quota" value={jeeData.quota} onChange={mkChange(setJeeData, jeeData)} className="field-input">
                <option value="AI">All India</option>
                <option value="OS">Other State</option>
                <option value="HS">Home State</option>
              </select>
            </div>
            <div className="field-group">
              <label className="field-label">Gender Pool</label>
              <select name="pool" value={jeeData.pool} onChange={mkChange(setJeeData, jeeData)} className="field-input">
                <option value="Gender-Neutral">Gender-Neutral</option>
                <option value="Female-Only">Female-Only</option>
              </select>
            </div>
          </>}

          {/* ── NEET ── */}
          {domain === 'NEET' && <>
            <div className="field-group">
              <label className="field-label">Your NEET Rank</label>
              <input
                type="text" inputMode="numeric" name="user_rank"
                value={neetData.user_rank} onChange={mkChange(setNeetData, neetData)}
                placeholder="e.g. 5000" className={`field-input ${rankError ? 'field-input-error' : ''}`}
              />
              {rankError && <span className="field-error-msg">⚠ {rankError}</span>}
            </div>
            <div className="field-group">
              <label className="field-label">Seat Category</label>
              <select name="category" value={neetData.category} onChange={mkChange(setNeetData, neetData)} className="field-input">
                <option value="OPEN SEAT">OPEN SEAT</option>
                <option value="ALL INDIA">ALL INDIA</option>
                <option value="DEEMED/PAID">DEEMED / PAID</option>
                <option value="EMPLOYEES">EMPLOYEE QUOTA / ESIC</option>
                <option value="DELHI">DELHI REGIONAL</option>
                <option value="MUSLIM">MINORITY (MUSLIM)</option>
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
