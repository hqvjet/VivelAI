// src/pages/Demo.js
import React, { useState, useRef } from 'react';
import { analyzeSentiment } from '../services/api';
import EmojiPicker from '../components/EmojiPicker';
import SentimentResult from '../components/SentimentResult';
import './Demo.css';

export default function Demo() {
  const [title, setTitle] = useState('');
  const [comment, setComment] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [approach, setApproach] = useState('app1');

  const taRef = useRef(null);

  // Insert emoji at cursor position
  const insertEmoji = emoji => {
    const ta = taRef.current;
    if (!ta) return;
    const { selectionStart: s, selectionEnd: e } = ta;
    const updated = comment.slice(0, s) + emoji + comment.slice(e);
    setComment(updated);
    setTimeout(() => {
      ta.focus();
      ta.setSelectionRange(s + emoji.length, s + emoji.length);
    }, 0);
  };

  // Chọn model API dựa trên người dùng chọn
  const selectModel = (selectedModel) => {
    setApproach(selectedModel);
  };

  const handleSubmit = async ev => {
    ev.preventDefault();
    if (!comment.trim()) return;
    setLoading(true);
    setShowResult(false);
    try {
      const res = await analyzeSentiment({ title, comment, approach });
      setResult(res);
      setTimeout(() => setShowResult(true), 100);
    } catch {
      alert('Error analyzing sentiment.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="demo">
      <div className="demo-container">
        <h2 className="demo-title">Demo Sentiment AI</h2>
        
        <div className="model-select">
          <label>Select Approach:</label>
          <select value={approach} onChange={(e) => selectModel(e.target.value)}>
            <option value="app1">Approach 1 - Title & Content Combination</option>
            <option value="app2">Approach 2 - Emoji Description</option>
            <option value="app3">Approach 3 - Emoji Embedding</option>
          </select>
        </div>

        <form onSubmit={handleSubmit} className="demo-form">
          {approach === 'app1' && (
           <div className="field-group">
            <label htmlFor="title" className="field-label">
              Title
            </label>
            <input
              id="title"
              type="text"
              value={title}
              onChange={e => setTitle(e.target.value)}
              className="field-input"
              placeholder="Enter title"
              required
            />
          </div>
          )}

          <div className="field-group">
            <label htmlFor="comment" className="field-label">
              Comment
            </label>
            <textarea
              id="comment"
              ref={taRef}
              value={comment}
              onChange={e => setComment(e.target.value)}
              className="field-input field-textarea"
              placeholder="Enter your text here"
              required
            />
          </div>

          <EmojiPicker onSelect={insertEmoji} />

          <button
            type="submit"
            className="btn-analyze"
            disabled={loading}
          >
            {loading ? 'Analyzing…' : 'Analyze'}
          </button>
        </form>

        <SentimentResult
          result={result}
          visible={showResult}
        />
      </div>
    </section>
  );
}

