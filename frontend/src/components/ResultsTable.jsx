import React from 'react';

const ResultsTable = ({ results }) => {
  if (!results) return null;

  const renderRows = (data, type) => {
    if (!data) return null;
    return data.map((item, idx) => (
      <tr key={`${type}-${idx}`}>
        <td>{item.institute}</td>
        <td>{item.program || item.course}</td>
        <td>{item.predicted_cutoff}</td>
        <td>
          <span className={`badge ${type.toLowerCase()}`}>
            {type}
          </span>
        </td>
      </tr>
    ));
  };

  const allEmpty = (results.Safe?.length || 0) === 0 && 
                   (results.Likely?.length || 0) === 0 && 
                   (results.Ambitious?.length || 0) === 0;

  if (allEmpty) {
    return (
      <div className="glass-panel" style={{ textAlign: 'center', padding: '40px' }}>
        <h3>No colleges found within a reasonable margin for this rank.</h3>
      </div>
    );
  }

  return (
    <div className="glass-panel results-container">
      <table>
        <thead>
          <tr>
            <th>Institute</th>
            <th>Program / Branch</th>
            <th>Predicted Cutoff</th>
            <th>Status Tier</th>
          </tr>
        </thead>
        <tbody>
          {renderRows(results.Safe, 'Safe')}
          {renderRows(results.Likely, 'Likely')}
          {renderRows(results.Ambitious, 'Ambitious')}
        </tbody>
      </table>
    </div>
  );
};

export default ResultsTable;
