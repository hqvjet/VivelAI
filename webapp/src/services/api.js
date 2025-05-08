import axios from 'axios';

// Thay URL bằng endpoint thật của bạn
const API_BASE = 'https://your-domain.com/api';

export async function analyzeSentiment({ title, content, emoji }) {
  const resp = await axios.post(`${API_BASE}/sentiment`, {
    title,
    content,
    emoji,
  });
  return resp.data; 
  // mong trả về { sentiment: 'Positive', scores: { positive:0.7, neutral:0.2, negative:0.1 } }
}
