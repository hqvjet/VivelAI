import React, { useState } from 'react';
import PropTypes from 'prop-types';
import './EmojiPicker.css';

const icons_mapping = {
  pos: [
        "😀", "😃", "😄", "😁", "😆", "😅", "😂", "🤣", "😊", "😇",
        "🙂", "😉", "😌", "😍", "🥰", "😘", "😗", "😙", "😚", "🤗",
        "🤩", "🤭", "🤠", "🥳", "💖", "💓", "💞", "💕", "💗", "💘",
        "💝", "🌟", "✨", "💫", "🎉", "🎊", "👏", "🙌", "👍", "💪",
        "🆗", "💖", "❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍"  
  ],
  neu: [
        "😐", "😑", "😶", "🙃", "🧐", "🤨", "😏", "😒", "😬", "🤔",
        "🤷", "😕", "😟", "🤝", "👌", "✌️",
        "🤞", "🤙", "💆", "🙆",
        "👐", "🙌", "🤲", "🤝", "👋", "🤚", "🖖", "✋", "🤏"  
  ],
  neg: [
        "😞", "😠", "😡", "🤬", "😭", "😢", "😿", "🙀", "💔", "😔",
        "😖", "😣", "😤", "😩", "😫", "🥵", "🥶", "🤒", "🤕", "🤧",
        "🥴", "😵", "🤯", "😰", "😨", "😧", "😦", "😬", "😿",
        "🙄", "💀", "☠️", "👿", "😈", "😒", "😓", "😑", "😞", "💢",
        "🤡", "👎", "🙅", "🚫", "❌", "🛑", "🤦"  
  ]
};

export default function EmojiPicker({ onSelect }) {
  const [open, setOpen] = useState({
    pos: false,
    neu: false,
    neg: false
  });

  const toggleCategory = (category) => {
    setOpen(prev => ({ ...prev, [category]: !prev[category] }));
  };

  return (
    <div className="emoji-picker">
      {Object.keys(icons_mapping).map((category) => (
        <div key={category} className="emoji-category">
          <button
            className="category-header"
            onClick={() => toggleCategory(category)}
          >
            {category === 'pos' ? 'Positive' : category === 'neu' ? 'Neutral' : 'Negative'}
          </button>
          <div
            className={`emoji-row ${open[category] ? 'open' : ''}`}
          >
            {icons_mapping[category].map(emoji => (
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
        </div>
      ))}
    </div>
  );
}

EmojiPicker.propTypes = {
  onSelect: PropTypes.func.isRequired,
};

