import React from 'react';
import './About.css';

const members = [
  {
    name: 'Hoàng Quốc Việt',
    id: '21IT119',
    motivation:
      'As our team lead, I combine a passion for linguistics with a drive to push the boundaries of NLP research and deliver real-world AI solutions.',
    photo: 'https://via.placeholder.com/150'
  },
  {
    name: 'Mai Nguyễn Xuân Thảo',
    id: '21DA111',
    motivation:
      'I specialize in computational semantics and am motivated by crafting intuitive AI tools that truly understand human emotions in service industries.',
    photo: 'https://via.placeholder.com/150'
  },
  {
    name: 'Phạm Trần Song Nguyên',
    id: '21DA222',
    motivation:
      'With a strong machine-learning background, I focus on building robust sentiment models that drive insights for hospitality, aviation, and beyond.',
    photo: 'https://via.placeholder.com/150'
  }
];

export default function About() {
  return (
    <section className="about">
      <h1 className="about__title">About Us</h1>
      <p className="about__lead">
        <strong>VKU Cerevex</strong> is a passionate NLP team dedicated to bringing
        advanced sentiment-analysis solutions to Vietnam’s service industries.
        We specialize in state-of-the-art natural language processing and
        are driven by a relentless ambition to innovate.
      </p>

      <div className="about__team">
        {members.map((m) => (
          <div key={m.id} className="member-card">
            <div
              className="member-photo"
              style={{ backgroundImage: `url(${m.photo})` }}
            />
            <h3 className="member-name">{m.name}</h3>
            <div className="member-id">{m.id}</div>
            <p className="member-motivation">{m.motivation}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

