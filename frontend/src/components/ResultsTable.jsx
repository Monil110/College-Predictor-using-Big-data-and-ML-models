import React, { useState, useMemo } from 'react';

const APP_NAME = 'AdmitSense';

// ─── Column config by domain ─────────────────────────────────────────────────
function getCutoffLabel(domain) {
  if (domain === 'NEET_PG') return 'Match %';
  return 'Predicted Cutoff';
}

function formatCutoff(item, domain) {
  if (domain === 'NEET_PG') {
    // predicted_cutoff was stored as Math.round(probability * 100) in App.jsx
    return `${item.predicted_cutoff}%`;
  }
  return (item.predicted_cutoff || 0).toLocaleString();
}

// ─── PDF Export ───────────────────────────────────────────────────────────────
function exportToPDF(results, domain) {
  const allRows = [
    ...(results.Safe   || []).map(r => ({ ...r, tier: 'Safe' })),
    ...(results.Likely || []).map(r => ({ ...r, tier: 'Likely' })),
  ];

  const rows = allRows.map((item, i) => `
    <tr>
      <td>${i + 1}</td>
      <td><strong>${item.institute || '—'}</strong></td>
      <td>${item.program || item.course || '—'}</td>
      <td style="font-weight:600;color:#0369a1">${typeof item.predicted_cutoff === 'number' ? item.predicted_cutoff.toLocaleString() : (item.predicted_cutoff || '—')}</td>
      <td><span class="tier-badge ${item.tier.toLowerCase()}">${item.tier}</span></td>
    </tr>`).join('');

  const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <title>${APP_NAME} — ${domain} Predictions</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; padding: 32px; color: #1e293b; background: #fff; }
    .header { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
    .header h1 { font-size: 24px; font-weight: 800; color: #0f172a; }
    .header .accent { color: #0ea5e9; }
    .meta { color: #64748b; font-size: 13px; margin-bottom: 24px; }
    .summary { display: flex; gap: 20px; margin-bottom: 20px; }
    .chip { padding: 5px 14px; border-radius: 20px; font-size: 12px; font-weight: 700; }
    .chip-safe   { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .chip-likely { background: #fef9c3; color: #a16207; border: 1px solid #fde047; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    thead tr { background: #0f172a; }
    th { color: #e2e8f0; padding: 11px 14px; text-align: left; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
    td { padding: 10px 14px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
    tr:nth-child(even) td { background: #f8fafc; }
    tr:hover td { background: #f0f9ff; }
    .tier-badge { padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 700; text-transform: uppercase; }
    .tier-badge.safe   { background: #dcfce7; color: #15803d; }
    .tier-badge.likely { background: #fef9c3; color: #a16207; }
    .footer { margin-top: 24px; font-size: 11px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 12px; }
    @media print { body { padding: 16px; } }
  </style>
</head>
<body>
  <div class="header">
    <h1>🎯 <span class="accent">Admit</span>Sense</h1>
  </div>
  <p class="meta">${domain} Admission Predictions &nbsp;·&nbsp; Generated ${new Date().toLocaleString()} &nbsp;·&nbsp; ${allRows.length} colleges found</p>
  <div class="summary">
    <span class="chip chip-safe">✅ Safe: ${(results.Safe || []).length}</span>
    <span class="chip chip-likely">⚡ Likely: ${(results.Likely || []).length}</span>
  </div>
  <table>
    <thead><tr><th>#</th><th>Institute</th><th>Program / Branch</th><th>Predicted Cutoff</th><th>Tier</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
  <p class="footer">${APP_NAME} · AI-Powered Admission Intelligence · Big Data &amp; ML</p>
</body>
</html>`;

  const win = window.open('', '_blank');
  if (!win) { alert('Pop-up blocked. Please allow pop-ups to export PDF.'); return; }
  win.document.write(html);
  win.document.close();
  win.focus();
  setTimeout(() => win.print(), 500);
}

// ─── Sort helpers ─────────────────────────────────────────────────────────────
// Rank-based exams (JEE / NEET UG / KCET / COMEDK):
//   ascending cutoff — most lenient (highest cutoff rank) first
// NEET PG:
//   descending probability % — highest match confidence first
//   (backend already sorts this way; we preserve it by NOT re-sorting)
function sortResults(arr, domain) {
  if (domain === 'NEET_PG') {
    // Backend sends descending by prob — preserve that order
    return [...arr];
  }
  // All rank-based exams: ascending cutoff (most lenient = safest first)
  return [...arr].sort((a, b) => (b.predicted_cutoff || 0) - (a.predicted_cutoff || 0));
}

// ─── Component ────────────────────────────────────────────────────────────────
const ResultsTable = ({ results, domain }) => {
  const [search, setSearch] = useState('');
  const [activeTab, setActiveTab] = useState('All');

  if (!results) return null;

  const safeList   = sortResults(results.Safe   || [], domain);
  const likelyList = sortResults(results.Likely || [], domain);
  const totalCount = safeList.length + likelyList.length;

  const cutoffLabel = getCutoffLabel(domain);
  const isNeetPg    = domain === 'NEET_PG';
  // For NEET PG we show "Match %" not a rank, so tweak the footnote
  const domainLabel = domain === 'NEET_UG' ? 'NEET UG'
                    : domain === 'NEET_PG' ? 'NEET PG'
                    : domain || 'Exam';

  if (totalCount === 0) {
    return (
      <div className="results-card empty-card">
        <div className="empty-icon-wrap">🔍</div>
        <h3 className="empty-title">No colleges found</h3>
        <p className="empty-sub">Try adjusting your category, quota, or verify your rank is within a valid range.</p>
      </div>
    );
  }

  const tabData = useMemo(() => {
    const base = activeTab === 'Safe'
      ? safeList.map(r => ({ ...r, tier: 'Safe' }))
      : activeTab === 'Likely'
        ? likelyList.map(r => ({ ...r, tier: 'Likely' }))
        : [
            ...safeList.map(r => ({ ...r, tier: 'Safe' })),
            ...likelyList.map(r => ({ ...r, tier: 'Likely' })),
          ];

    if (!search.trim()) return base;
    const q = search.toLowerCase();
    return base.filter(item =>
      (item.institute || '').toLowerCase().includes(q) ||
      (item.program   || '').toLowerCase().includes(q) ||
      (item.course    || '').toLowerCase().includes(q)
    );
  }, [activeTab, search, safeList, likelyList]);

  const hasProb = false; // eligibility_prob removed — rank-based filtering is the gate

  return (
    <div className="results-card">

      {/* ── Top bar ── */}
      <div className="results-topbar">
        <div className="results-headline">
          <span className="results-big-num">{totalCount}</span>
          <div>
            <p className="results-headline-title">Colleges Found</p>
            {domain && <p className="results-headline-sub">{domainLabel} · 2026 Predictions</p>}
          </div>
        </div>
        <button className="export-btn" onClick={() => exportToPDF(results, domainLabel)}>
          <span>📄</span> Export PDF
        </button>
      </div>

      {/* ── Tier chips ── */}
      <div className="tier-chips-row">
        <div className="tier-chip-item safe-chip-item">
          <span className="tier-chip-dot safe-dot" />
          <span className="tier-chip-label">Safe</span>
          <span className="tier-chip-count">{safeList.length}</span>
        </div>
        <div className="tier-chip-item likely-chip-item">
          <span className="tier-chip-dot likely-dot" />
          <span className="tier-chip-label">Likely</span>
          <span className="tier-chip-count">{likelyList.length}</span>
        </div>
        <p className="tier-legend">
          {isNeetPg ? 'Higher match % = stronger prediction' : 'Higher cutoff rank = more lenient (safest) college'}
        </p>
      </div>

      {/* ── Controls ── */}
      <div className="results-controls-bar">
        <div className="search-wrap">
          <span className="search-icon">🔎</span>
          <input
            type="text"
            className="search-field"
            placeholder="Search institute or branch..."
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
          {search && (
            <button className="search-clear" onClick={() => setSearch('')}>✕</button>
          )}
        </div>
        <div className="filter-tabs">
          {[
            { key: 'All',    count: totalCount },
            { key: 'Safe',   count: safeList.length },
            { key: 'Likely', count: likelyList.length },
          ].map(({ key, count }) => (
            <button
              key={key}
              className={`filter-tab ${activeTab === key ? 'filter-tab-active' : ''}`}
              onClick={() => setActiveTab(key)}
            >
              {key} <span className="filter-tab-count">{count}</span>
            </button>
          ))}
        </div>
      </div>

      {/* ── Table ── */}
      <div className="table-scroll">
        {tabData.length === 0 ? (
          <div className="no-match-msg">No results match "{search}"</div>
        ) : (
          <table className="results-table">
            <thead>
              <tr>
                <th className="col-num">#</th>
                <th>Institute</th>
                <th>Program / Branch</th>
                <th className="col-cutoff">{cutoffLabel}</th>
                {hasProb && <th className="col-prob">Eligibility</th>}
                <th className="col-tier">Tier</th>
              </tr>
            </thead>
            <tbody>
              {tabData.map((item, idx) => {
                const tier = item.tier || 'Safe';
                return (
                  <tr key={`${tier}-${idx}`} className={`result-row result-row-${tier.toLowerCase()}`}>
                    <td className="col-num td-num">{idx + 1}</td>
                    <td className="td-institute">{item.institute || '—'}</td>
                    <td className="td-program">{item.program || item.course || '—'}</td>
                    <td className="td-cutoff col-cutoff">
                      {formatCutoff(item, domain)}
                    </td>
                    {hasProb && (
                      <td className="col-prob">
                        {item.eligibility_prob !== undefined ? (
                          <div className="prob-wrap">
                            <div className="prob-track">
                              <div
                                className="prob-fill"
                                style={{ width: `${Math.round(item.eligibility_prob * 100)}%` }}
                              />
                            </div>
                            <span className="prob-pct">{Math.round(item.eligibility_prob * 100)}%</span>
                          </div>
                        ) : '—'}
                      </td>
                    )}
                    <td className="col-tier">
                      <span className={`tier-badge tier-${tier.toLowerCase()}`}>{tier}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      <p className="results-footnote">
        {isNeetPg
          ? 'ℹ️ Results show top predicted college + course allocations based on your rank and category. Match % = model confidence. Only Karnataka NEET PG data was used for training.'
          : 'ℹ️ Results sorted by predicted cutoff — most lenient (safest) colleges appear first. Safe = cutoff comfortably above your rank · Likely = cutoff close to your rank. All listed colleges are ones where your rank qualifies for admission.'
        }
      </p>
    </div>
  );
};

export default ResultsTable;
