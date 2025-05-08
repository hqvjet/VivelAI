import React from 'react';
import './Info.css';

const papers = [
  {
    title: 'E2v-PhoBERT: Enhanced Vietnamese Sentiment Analysis with Emoji Integration',
    authors: 'Dai Tho Dang, Quoc Viet Hoang, Nguyen Xuan Thao Mai, Ngoc Thanh Nguyen',
    conference: 'ACIIDS 2025',
    proceeding: 'Communications in Computer and Information Science, vol. 2493 (Springer)',
    indexed: 'Scopus'
  },
  {
    title: 'Sentiment Analysis of Hotel Customer Reviews',
    authors: 'Nguyen Xuan Thao Mai, Pham Song Nguyen Tran, Cong Phap Huynh, Dai Tho Dang',
    conference: 'ICIIT 2025',
    proceeding: '–',
    indexed: 'Scopus'
  },
  {
    title: 'VED_PhoBERT: Integrating Emoji Descriptions for Improved Vietnamese Sentiment Detection',
    authors: 'Cong Phap Huynh, Quoc Viet Hoang, Nguyen Xuan Thao Mai, Pham Song Nguyen Tran',
    conference: 'CITA 2025',
    proceeding: '–',
    indexed: 'Scopus'
  }
];

export default function Info() {
  return (
    <section className="info">
      <h1 className="info__title">Info</h1>

      <div className="info__motivation">
        <h2>Motivation</h2>
        <p>
          In the global digital-transformation era, online reviews have become a crucial data source
          reflecting customer emotions, satisfaction, and expectations towards service industries. 
          Beyond individual opinions, they guide consumer behavior, shape brand reputation and 
          influence service-development strategies. Notably, 82% of customers research products 
          online before buying in-store.
        </p>
        <p>
          In Vietnam, the service sector drives economic growth: tourism saw over 17.5 million
          international visitors in 2024, aviation led Southeast Asia with a 17.4% CAGR (2016–2021),
          and e-commerce exceeded $20 billion in 2023. However, despite abundant review data,
          academic research on Vietnamese sentiment analysis in service domains remains limited.
          This demo therefore focuses on improving the quality of Vietnamese customer-review
          sentiment analysis across tourism, aviation and e-commerce.
        </p>
      </div>

      <div className="info__papers">
        <h2>Publications</h2>
        <div className="paper-grid">
          {papers.map((p, index) => (
            <div key={index} className="paper-card">
              <h3 className="paper-card__title">{p.title}</h3>
              <div className="paper-card__authors">{p.authors}</div>
              <div className="paper-card__conf">{p.conference}</div>
              <div className="paper-card__proc">{p.proceeding}</div>
              {p.indexed && (
                <div className="paper-card__idx">
                  Indexed: {p.indexed}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

