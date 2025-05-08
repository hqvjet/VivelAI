import axios from 'axios';

// Thay URL bằng endpoint thật của bạn
const API_BASE = 'http://localhost:8000/api/v1';

export async function analyzeSentiment({title, comment, approach}) {
    let resp = {};
    if (approach === 'app3') {
        resp = await axios.post(`${API_BASE}/E2V-PhoBERT/predict`, {
            comment,
        });
    }
    else if (approach === 'app2') {
        resp = await axios.post(`${API_BASE}/VED-PhoBERT/predict`, {
            comment,
        });
    }
    // else {
    //     resp = await axios.post(`${API_BASE}/app1/predict`, {
    //         title,
    //         comment
    //     });
    // }
    return resp.data; 
}
