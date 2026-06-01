import React, { useState, useMemo } from 'react';

// ─── PDF Export ───────────────────────────────────────────────────────────────
// Uses the browser's built-in print dialog with a print-specific stylesheet.
// No external library needed — works everywhere.
function exportToPDF(results, domain) {
  const allRows = [
    ...(results.Safe   || []).map(r => ({ ...r, tier: 'Safe' })),
    ...(results.Likely || []).map(r => ({ ...r, tier: 'Likely' })),
  ];

  const rows = allRows
    .map((item, i) => `
      <tr class="${item.tier.toLowerCase()}">
        <td>${i + 1}</td>
        <td>${item.institute || '—'}</td>
        <td>${item.program || item.course || '—'}</td>
        <td>${(item.predicted_cutoff || 0).toLocaleString()}</td>
        <td class="tier-cell ${item.tier.toLowerCase()}">${item.tier}</td>
      </tr>`)
    .join('');

  const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <title>PredictMe — ${domain} College Predictions</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 24px; color: #111; }
    h1 { font-size: 22px; margin-bottom: 4px; }
    p.sub { color: #555; font-size: 13px; margin-bottom: 20px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { background: #1e293b; color: #fff; padding: 10px 12px; text-align: left; }
    td { padding: 9px 12px; border-bottom: 1px solid #e2e8f0; }
    tr:nth-child(even) td { background: #f8fafc; }
    .tier-cell { font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .safe   .tier-cell { color: #059669; }
    .likely .tier-cell { color: #d97706; }
    .footer { margin-top: 20px; font-size: 11px; color: #888; }
    @media print { body { padding: 0; } }
  </style>
</head>
<body>
  <h1>PredictMe — ${domain} Admission Predictions</h1>
  <p class="sub">Generated on ${new Date().toLocaleString()} &nbsp;|&nbsp; Total: ${allRows.length} colleges</p>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Institute</th>
        <th>Program / Branch</th>
        <th>Predicted Cutoff Rank</th>
        <th>Tier</th>
      </tr>
    </thead>
    <tbody>${rows}</tbody>
  </table>
  <p class="footer">Powered by PredictMe · Big Data &amp; ML Admission Intelligence</p>
</body>
</html>`;

  const win = window.open('', '_blank');
  if (!win) {
    alert('Pop-up blocked. Please allow pop-ups for this site to export PDF.');
    return;
  }
  win.document.write(html);
  win.document.close();
  win.focus();
  setTimeout(() => {
    win.print();
  }, 400);
}

// ─── Sorting: ascending cutoff = most prestigious first ──────────────────────
function sortByPrestige(arr) {
  return [...arr].sort((a, b) => (a.predicted_cutoff || 0) - (b.predicted_cutoff || 0));
}

// ─── Component ────────────────────────────────────────────────────────────────
const ResultsTable = ({ results, domain }) => {
  const [search, setSearch] = useState('');
  const [activeTab, setActiveTab] = useState('All');

  if (!results) return null;

  const safeList   = sortByPrestige(results.Safe   || []);
  const likelyList = sortByPrestige(results.Likely || []);

  const totalCount = safeList.length + likelyList.length;

  const allEmpty = totalCount === 0;

  if (allEmpty) {
    return (
      <div className="glass-panel empty-state">
        <div className="empty-icon">🔍</div>
        <h3>No colleges found for this rank &amp; category combination.</h3>
        <p>Try adjusting your category, quota, or check if the rank is within a valid range.</p>
      </div>
    );
  }

  // Filter by tab
  const tabData = useMemo(() => {
    const base = activeTab === 'Safe'
      ? safeList
      : activeTab === 'Likely'
        ? likelyList
        : [...safeList.map(r => ({ ...r, tier: 'Safe' })),
           ...likelyList.map(r => ({ ...r, tier: 'Likely' }))];

    if (!search.trim()) return base;
    const q = search.toLowerCase();
    return base.filter(
      item =>
        (item.institute || '').toLowerCase().includes(q) ||
        (item.program   || '').toLowerCase().includes(q) ||
        (item.course    || '').toLowerCase().includes(q)
    );
  }, [activeTab, search, safeList, likelyList]);

  const renderRow = (item, idx, tierOverride) => {
    const tier = tierOverride || item.tier || 'Safe';
    return (
      <tr key={`${tier}-${idx}`}>
        <td className="rank-col">{idx + 1}</td>
        <td className="institute-col">{item.institute || '—'}</td>
        <td>{item.program || item.course || '—'}</td>
        <td className="cutoff-col">
          {(item.predicted_cutoff || 0).toLocaleString()}
        </td>
        {item.eligibility_prob !== undefined && (
          <td className="prob-col">
            <div className="prob-bar-wrap">
              <div
                className="prob-bar"
                style={{ width: `${Math.round(item.eligibility_prob * 100)}%` }}
              />
              <span>{Math.round(item.eligibility_prob * 100)}%</span>
            </div>
          </td>
        )}
        <td>
          <span className={`badge ${tier.toLowerCase()}`}>{tier}</span>
        </td>
      </tr>
    );
  };

  const hasProb = (safeList[0] || likelyList[0] || {}).eligibility_prob !== undefined;

  return (
    <div className="glass-panel results-wrapper">
      {/* Header row */}
      <div className="results-header">
        <div className="results-title">
          <span className="results-count">{totalCount}</span> colleges found
          {domain && <span className="results-domain"> · {domain}</span>}
        </div>
        <button
          className="pdf-btn"
          onClick={() => exportToPDF(results, domain || 'Exam')}
          title="Export results as PDF"
        >
          📄 Export PDF
        </button>
      </div>

      {/* Tier summary chips */}
      <div className="tier-summary">
        <span className="tier-chip safe-chip">
          ✅ Safe: {safeList.length}
        </span>
        <span className="tier-chip likely-chip">
          ⚡ Likely: {likelyList.length}
        </span>
      </div>

      {/* Search + Tab filter */}
      <div className="results-controls">
        <input
          type="text"
          className="search-input"
          placeholder="🔎 Search institute or branch..."
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <div className="tab-group">
          {['All', 'Safe', 'Likely'].map(tab => (
            <button
              key={tab}
              className={`tab-btn ${activeTab === tab ? 'tab-active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
              <span className="tab-count">
                {tab === 'All'    ? totalCount
                 : tab === 'Safe'  ? safeList.length
                 : likelyList.length}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="results-container">
        {tabData.length === 0 ? (
          <div className="no-match">No results match your search.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Institute</th>
                <th>Program / Branch</th>
                <th>Predicted Cutoff</th>
                {hasProb && <th>Eligibility</th>}
                <th>Tier</th>
              </tr>
            </thead>
            <tbody>
              {tabData.map((item, idx) =>
                renderRow(item, idx, activeTab !== 'All' ? activeTab : item.tier)
              )}
            </tbody>
          </table>
        )}
      </div>

      <p className="results-note">
        ℹ️ Colleges are sorted by predicted cutoff rank (ascending) — lower cutoff = more prestigious / competitive.
        Safe = high admission probability · Likely = moderate probability.
      </p>
    </div>
  );
};

export default ResultsTable;
