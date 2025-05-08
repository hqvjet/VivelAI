import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import './SentimentResult.css';

export default function SentimentResult({ result, visible }) {
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (visible) {
      setTimeout(() => setShow(true), 100);
    } else {
      setShow(false);
    }
  }, [visible]);

  if (!result) return null;

  const { Sentiment } = result;
  const emoji =
    Sentiment === 'Positive' ? '😊' :
    Sentiment === 'Neutral'  ? '😐' : '😡';

  return (
    <div className={`result-card ${show ? 'show' : ''}`}>
      <div className="result-header">
        <span className="result-emoji">{emoji}</span>
        <span className="result-label">{Sentiment}</span>
      </div>
      {/*<div className="result-bars">
        {Object.entries(scores).map(([key, val]) => (
          <div className="bar-wrapper" key={key}>
            <span className="bar-label">
              {key.charAt(0).toUpperCase() + key.slice(1)}
            </span>
            <div className="bar-track">
              <div
                className={`bar-fill ${key}`}
                style={{ width: `${show ? val * 100 : 0}%` }}
              />
            </div>
            <span className="bar-percent">
              {(val * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>*/}
    </div>
  );
}

SentimentResult.propTypes = {
  result: PropTypes.shape({
    sentiment: PropTypes.string.isRequired,
    scores:    PropTypes.object.isRequired,
  }),
  visible: PropTypes.bool,
};

SentimentResult.defaultProps = {
  visible: false,
};

