import React, { useState } from 'react';

// ─── Rank validation ────────────────────────────────────────────────────────
// Rules:
//  • Raw string "00000" (all zeros) → error
//  • Leading zeros like "0001" → treated as 1 (parseInt strips them)
//  • 0 or negative → error
//  • Non-integer → error
function validateRank(rawValue, label, max) {
  const str = String(rawValue).trim();

  // All zeros check (e.g. "000", "00000")
  if (/^0+$/.test(str)) {
    return `${label} cannot be zero. Please enter a valid rank.`;
  }

  const parsed = parseInt(str, 10);

  if (isNaN(parsed) || !Number.isFinite(parsed)) {
    return `${label} must be a valid number.`;
  }
  if (parsed <= 0) {
    return `${label} must be greater than 0.`;
  }
  if (max && parsed > max) {
    return `${label} seems too high (max ${max.toLocaleString()}). Please verify.`;
  }
  return null; // valid
}

const RANK_LIMITS = {
  JEE_ADVANCED: 200000,
  JEE_MAIN: 1200000,
  NEET: 2000000,
  KCET: 200000,
  COMEDK: 200000,
};

// ─── Component ───────────────────────────────────────────────────────────────
const PredictionForm = ({ onPredict, isLoading }) => {
  const [domain, setDomain] = useState('JEE');
  const [rankError, setRankError] = useState('');

  const [formData, setFormData] = useState({
    user_rank: '',
    exam_type: 'JEE Advanced',
    category: 'GEN',
    quota: 'AI',
    pool: 'Gender-Neutral',
  });

  const [neetFormData, setNeetFormData] = useState({
    user_rank: '',
    category: 'OPEN SEAT',
  });

  const [kcetFormData, setKcetFormData] = useState({
    user_rank: '',
    category: 'GM',
    base_category: 'GM',
    quota: 'General',
    region: 'General',
  });

  const [comedkFormData, setComedkFormData] = useState({
    user_rank: '',
    category: 'GM',
  });

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleDomainChange = (e) => {
    setDomain(e.target.value);
    setRankError('');
  };

  const handleJeeChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    if (e.target.name === 'user_rank') setRankError('');
  };

  const handleNeetChange = (e) => {
    setNeetFormData({ ...neetFormData, [e.target.name]: e.target.value });
    if (e.target.name === 'user_rank') setRankError('');
  };

  const handleKcetChange = (e) => {
    setKcetFormData({ ...kcetFormData, [e.target.name]: e.target.value });
    if (e.target.name === 'user_rank') setRankError('');
  };

  const handleComedkChange = (e) => {
    setComedkFormData({ ...comedkFormData, [e.target.name]: e.target.value });
    if (e.target.name === 'user_rank') setRankError('');
  };

  // ── Submit ─────────────────────────────────────────────────────────────────
  const handleSubmit = (e) => {
    e.preventDefault();

    let rawRank, rankLabel, rankMax;

    if (domain === 'JEE') {
      rawRank = formData.user_rank;
      rankLabel = 'JEE Rank';
      rankMax = formData.exam_type === 'JEE Advanced'
        ? RANK_LIMITS.JEE_ADVANCED
        : RANK_LIMITS.JEE_MAIN;
    } else if (domain === 'NEET') {
      rawRank = neetFormData.user_rank;
      rankLabel = 'NEET Rank';
      rankMax = RANK_LIMITS.NEET;
    } else if (domain === 'KCET') {
      rawRank = kcetFormData.user_rank;
      rankLabel = 'KCET Rank';
      rankMax = RANK_LIMITS.KCET;
    } else {
      rawRank = comedkFormData.user_rank;
      rankLabel = 'COMEDK Rank';
      rankMax = RANK_LIMITS.COMEDK;
    }

    const error = validateRank(rawRank, rankLabel, rankMax);
    if (error) {
      setRankError(error);
      return;
    }

    setRankError('');
    const parsedRank = parseInt(String(rawRank).trim(), 10);

    if (domain === 'JEE') {
      onPredict({ ...formData, user_rank: parsedRank, domain: 'JEE' });
    } else if (domain === 'KCET') {
      onPredict({ ...kcetFormData, user_rank: parsedRank, domain: 'KCET' });
    } else if (domain === 'COMEDK') {
      onPredict({ ...comedkFormData, user_rank: parsedRank, domain: 'COMEDK' });
    } else {
      onPredict({
        ...neetFormData,
        candidate_rank: parsedRank,
        domain: 'NEET',
      });
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="glass-panel">
      <form onSubmit={handleSubmit} noValidate>

        {/* Domain Tabs */}
        <div className="domain-tabs">
          {[
            { value: 'JEE',    label: '🎓 Engineering (JEE)' },
            { value: 'KCET',   label: '🎓 Engineering (KCET)' },
            { value: 'COMEDK', label: '🎓 Engineering (COMEDK)' },
            { value: 'NEET',   label: '🩺 Medical (NEET)' },
          ].map(({ value, label }) => (
            <label
              key={value}
              className={`domain-tab ${domain === value ? 'active' : ''}`}
            >
              <input
                type="radio"
                value={value}
                checked={domain === value}
                onChange={handleDomainChange}
                style={{ display: 'none' }}
              />
              {label}
            </label>
          ))}
        </div>

        {/* Form Fields */}
        <div className="form-grid">
          {domain === 'JEE' && (
            <>
              <div className="form-group">
                <label>JEE Rank</label>
                <input
                  type="text"
                  inputMode="numeric"
                  name="user_rank"
                  value={formData.user_rank}
                  onChange={handleJeeChange}
                  placeholder="e.g. 2500"
                  className={rankError ? 'input-error' : ''}
                  required
                />
                {rankError && <span className="field-error">{rankError}</span>}
              </div>
              <div className="form-group">
                <label>Exam Mode</label>
                <select name="exam_type" value={formData.exam_type} onChange={handleJeeChange}>
                  <option value="JEE Advanced">JEE Advanced (IITs)</option>
                  <option value="JEE Main">JEE Main (NITs &amp; IIITs)</option>
                </select>
              </div>
              <div className="form-group">
                <label>Category</label>
                <select name="category" value={formData.category} onChange={handleJeeChange}>
                  <option value="GEN">OPEN (General)</option>
                  <option value="OBC-NCL">OBC-NCL</option>
                  <option value="SC">SC</option>
                  <option value="ST">ST</option>
                  <option value="GEN-EWS">GEN-EWS</option>
                </select>
              </div>
              <div className="form-group">
                <label>Quota</label>
                <select name="quota" value={formData.quota} onChange={handleJeeChange}>
                  <option value="AI">All India</option>
                  <option value="OS">Other State</option>
                  <option value="HS">Home State</option>
                </select>
              </div>
              <div className="form-group">
                <label>Gender Pool</label>
                <select name="pool" value={formData.pool} onChange={handleJeeChange}>
                  <option value="Gender-Neutral">Gender-Neutral</option>
                  <option value="Female-Only">Female-Only</option>
                </select>
              </div>
            </>
          )}

          {domain === 'NEET' && (
            <>
              <div className="form-group">
                <label>NEET Rank</label>
                <input
                  type="text"
                  inputMode="numeric"
                  name="user_rank"
                  value={neetFormData.user_rank}
                  onChange={handleNeetChange}
                  placeholder="e.g. 5000"
                  className={rankError ? 'input-error' : ''}
                  required
                />
                {rankError && <span className="field-error">{rankError}</span>}
              </div>
              <div className="form-group">
                <label>Allotted Category Constraint</label>
                <select name="category" value={neetFormData.category} onChange={handleNeetChange}>
                  <option value="OPEN SEAT">OPEN SEAT</option>
                  <option value="ALL INDIA">ALL INDIA</option>
                  <option value="DEEMED/PAID">DEEMED / PAID</option>
                  <option value="EMPLOYEES">EMPLOYEE QUOTA / ESIC</option>
                  <option value="DELHI">DELHI REGIONAL</option>
                  <option value="MUSLIM">MINORITY (MUSLIM)</option>
                </select>
              </div>
            </>
          )}

          {domain === 'KCET' && (
            <>
              <div className="form-group">
                <label>KCET Rank</label>
                <input
                  type="text"
                  inputMode="numeric"
                  name="user_rank"
                  value={kcetFormData.user_rank}
                  onChange={handleKcetChange}
                  placeholder="e.g. 25000"
                  className={rankError ? 'input-error' : ''}
                  required
                />
                {rankError && <span className="field-error">{rankError}</span>}
              </div>
              <div className="form-group">
                <label>Rank Category</label>
                <select name="category" value={kcetFormData.category} onChange={handleKcetChange}>
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
              <div className="form-group">
                <label>Base Caste Category</label>
                <select name="base_category" value={kcetFormData.base_category} onChange={handleKcetChange}>
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
              <div className="form-group">
                <label>Quota Extension</label>
                <select name="quota" value={kcetFormData.quota} onChange={handleKcetChange}>
                  <option value="General">General</option>
                  <option value="Kannada">Kannada</option>
                  <option value="Rural">Rural</option>
                </select>
              </div>
              <div className="form-group">
                <label>Geographic Region</label>
                <select name="region" value={kcetFormData.region} onChange={handleKcetChange}>
                  <option value="General">General Range</option>
                  <option value="Hyderabad-Karnataka">Hyderabad-Karnataka (HK)</option>
                </select>
              </div>
            </>
          )}

          {domain === 'COMEDK' && (
            <>
              <div className="form-group">
                <label>COMEDK Rank</label>
                <input
                  type="text"
                  inputMode="numeric"
                  name="user_rank"
                  value={comedkFormData.user_rank}
                  onChange={handleComedkChange}
                  placeholder="e.g. 15000"
                  className={rankError ? 'input-error' : ''}
                  required
                />
                {rankError && <span className="field-error">{rankError}</span>}
              </div>
              <div className="form-group">
                <label>Category</label>
                <select name="category" value={comedkFormData.category} onChange={handleComedkChange}>
                  <option value="GM">General Merit (GM)</option>
                  <option value="KKR">Kalyana Karnataka Region (KKR)</option>
                </select>
              </div>
            </>
          )}
        </div>

        <button type="submit" className="submit-btn" disabled={isLoading}>
          {isLoading ? 'Analyzing...' : '🔍 Predict Admission Tier'}
        </button>
      </form>
    </div>
  );
};

export default PredictionForm;
