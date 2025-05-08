import React from 'react';
import PropTypes from 'prop-types';
import './EmojiPicker.css';

const EMOJIS = ['😡','😍','😐','🙂'];

export default function EmojiPicker({ onSelect }) {
  return (
    <div className="emoji-picker">
      {EMOJIS.map(emoji => (
        <button
          key={emoji}
          type="button"
          className="emoji-btn"
          onClick={() => onSelect(emoji)}
        >
          {emoji}
        </button>
      ))}
    </div>
  );
}

EmojiPicker.propTypes = {
  onSelect: PropTypes.func.isRequired,
};

