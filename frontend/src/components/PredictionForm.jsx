import React, { useState } from 'react';

const PredictionForm = ({ onPredict, isLoading }) => {
  const [domain, setDomain] = useState('JEE');
  
  const [formData, setFormData] = useState({
    user_rank: 2500,
    exam_type: 'JEE Advanced',
    category: 'GEN',
    quota: 'AI',
    pool: 'Gender-Neutral'
  });

  const [neetFormData, setNeetFormData] = useState({
    user_rank: 5000,
    category: 'OPEN SEAT'
  });

  const [kcetFormData, setKcetFormData] = useState({
    user_rank: 25000,
    category: 'GM',
    base_category: 'GM',
    quota: 'General',
    region: 'General'
  });

  const [comedkFormData, setComedkFormData] = useState({
    user_rank: 15000,
    category: 'GM'
  });

  const handleDomainChange = (e) => {
    setDomain(e.target.value);
  };

  const handleJeeChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleNeetChange = (e) => {
    setNeetFormData({ ...neetFormData, [e.target.name]: e.target.value });
  };

  const handleKcetChange = (e) => {
    setKcetFormData({ ...kcetFormData, [e.target.name]: e.target.value });
  };

  const handleComedkChange = (e) => {
    setComedkFormData({ ...comedkFormData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (domain === 'JEE') {
      onPredict({ ...formData, user_rank: parseInt(formData.user_rank, 10), domain: 'JEE' });
    } else if (domain === 'KCET') {
      onPredict({ ...kcetFormData, user_rank: parseInt(kcetFormData.user_rank, 10), domain: 'KCET' });
    } else if (domain === 'COMEDK') {
      onPredict({ ...comedkFormData, user_rank: parseInt(comedkFormData.user_rank, 10), domain: 'COMEDK' });
    } else {
      // Maps the frontend visual `user_rank` logically to what `backend/neet/` expects: `candidate_rank`
      onPredict({ ...neetFormData, candidate_rank: parseInt(neetFormData.user_rank, 10), domain: 'NEET' });
    }
  };

  return (
    <div className="glass-panel">
      <form onSubmit={handleSubmit}>
        
        {/* Toggle Controls */}
        <div className="form-group" style={{ textAlign: "center", marginBottom: "25px", borderBottom: "1px solid rgba(255,255,255,0.1)", paddingBottom: "15px" }}>
            <label style={{ display: "inline-flex", alignItems: "center", marginRight: "30px", cursor: "pointer", fontSize: "1.1rem", fontWeight: "bold" }}>
              <input type="radio" value="JEE" checked={domain === 'JEE'} onChange={handleDomainChange} style={{marginRight: "8px", width: "18px", height: "18px"}} /> 
              Engineering (JEE)
            </label>
            <label style={{ display: "inline-flex", alignItems: "center", marginRight: "30px", cursor: "pointer", fontSize: "1.1rem", fontWeight: "bold" }}>
              <input type="radio" value="KCET" checked={domain === 'KCET'} onChange={handleDomainChange} style={{marginRight: "8px", width: "18px", height: "18px"}} /> 
              Engineering (KCET)
            </label>
            <label style={{ display: "inline-flex", alignItems: "center", marginRight: "30px", cursor: "pointer", fontSize: "1.1rem", fontWeight: "bold" }}>
              <input type="radio" value="COMEDK" checked={domain === 'COMEDK'} onChange={handleDomainChange} style={{marginRight: "8px", width: "18px", height: "18px"}} /> 
              Engineering (COMEDK)
            </label>
            <label style={{ display: "inline-flex", alignItems: "center", cursor: "pointer", fontSize: "1.1rem", fontWeight: "bold" }}>
              <input type="radio" value="NEET" checked={domain === 'NEET'} onChange={handleDomainChange} style={{marginRight: "8px", width: "18px", height: "18px"}} /> 
              Medical (NEET)
            </label>
        </div>

        <div className="form-grid">
          {domain === 'JEE' ? (
            <>
              <div className="form-group">
                <label>JEE Rank</label>
                <input type="number" name="user_rank" value={formData.user_rank} onChange={handleJeeChange} required />
              </div>
              <div className="form-group">
                <label>Exam Mode</label>
                <select name="exam_type" value={formData.exam_type} onChange={handleJeeChange}>
                  <option value="JEE Advanced">JEE Advanced (IITs)</option>
                  <option value="JEE Main">JEE Main (NITs & IIITs)</option>
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
          ) : domain === 'NEET' ? (
            <>
              <div className="form-group">
                <label>NEET Rank</label>
                <input type="number" name="user_rank" value={neetFormData.user_rank} onChange={handleNeetChange} required />
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
          ) : domain === 'KCET' ? (
            <>
              <div className="form-group">
                <label>KCET Rank</label>
                <input type="number" name="user_rank" value={kcetFormData.user_rank} onChange={handleKcetChange} required />
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
          ) : domain === 'COMEDK' ? (
            <>
              <div className="form-group">
                <label>COMEDK Rank</label>
                <input type="number" name="user_rank" value={comedkFormData.user_rank} onChange={handleComedkChange} required />
              </div>
              <div className="form-group">
                <label>Category</label>
                <select name="category" value={comedkFormData.category} onChange={handleComedkChange}>
                  <option value="GM">General Merit (GM)</option>
                  <option value="KKR">Kalyana Karnataka Region (KKR)</option>
                </select>
              </div>
            </>
          ) : null}
        </div>
        <button type="submit" className="submit-btn" disabled={isLoading}>
          {isLoading ? 'Booting ML Network...' : 'Predict Admission Tier'}
        </button>
      </form>
    </div>
  );
};

export default PredictionForm;
