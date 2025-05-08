// src/pages/Demo.js
import React, { useState, useRef } from 'react';
import { analyzeSentiment } from '../services/api';
import EmojiPicker from '../components/EmojiPicker';
import SentimentResult from '../components/SentimentResult';
import './Demo.css';

export default function Demo() {
  const [title, setTitle]           = useState('');
  const [content, setContent]       = useState('');
  const [loading, setLoading]       = useState(false);
  const [result, setResult]         = useState(null);
  const [showResult, setShowResult] = useState(false);

  // refs cho 2 field
  const titleRef = useRef(null);
  const taRef    = useRef(null);

  // chèn emoji vào nơi đang focus
  const insertEmoji = emoji => {
    const active = document.activeElement;
    // Nếu đang focus ở input title
    if (active === titleRef.current) {
      const { selectionStart: s, selectionEnd: e } = active;
      const updated = title.slice(0, s) + emoji + title.slice(e);
      setTitle(updated);
      // reset caret sau khi React cập nhật
      setTimeout(() => {
        titleRef.current.focus();
        titleRef.current.setSelectionRange(s + emoji.length, s + emoji.length);
      }, 0);

    // Nếu đang focus ở textarea content
    } else if (active === taRef.current) {
      const { selectionStart: s, selectionEnd: e } = active;
      const updated = content.slice(0, s) + emoji + content.slice(e);
      setContent(updated);
      setTimeout(() => {
        taRef.current.focus();
        taRef.current.setSelectionRange(s + emoji.length, s + emoji.length);
      }, 0);

    // Nếu không focus vào đâu (mặc định chèn vào content cuối)
    } else {
      setContent(prev => prev + emoji);
    }
  };

  const handleSubmit = async ev => {
    ev.preventDefault();
    if (!content.trim()) return;
    setLoading(true);
    setShowResult(false);
    try {
      const res = await analyzeSentiment({ title, content });
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
        <form onSubmit={handleSubmit} className="demo-form">
          <div className="field-group">
            <label htmlFor="title" className="field-label">
              Title <span className="optional">(optional)</span>
            </label>
            <input
              id="title"
              ref={titleRef}
              type="text"
              value={title}
              onChange={e => setTitle(e.target.value)}
              className="field-input"
              placeholder="Enter title"
            />
          </div>

          <div className="field-group">
            <label htmlFor="content" className="field-label">
              Content
            </label>
            <textarea
              id="content"
              ref={taRef}
              value={content}
              onChange={e => setContent(e.target.value)}
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

