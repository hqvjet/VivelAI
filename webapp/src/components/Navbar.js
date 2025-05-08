
import React from 'react';
import { NavLink } from 'react-router-dom';
import './Navbar.css';

export default function Navbar() {
  return (
    <header className="navbar">
      <div className="navbar__logo">
        AI SENTIMENT ANALYSIS - VKU CEREVEX
      </div>
      <nav className="navbar__links">
        <NavLink
          to="/about"
          className={({ isActive }) =>
            isActive ? 'navbar__link navbar__link--active' : 'navbar__link'
          }
        >
          About Us
        </NavLink>
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            isActive ? 'navbar__link navbar__link--active' : 'navbar__link'
          }
        >
          Demo
        </NavLink>
        <NavLink
          to="/info"
          className={({ isActive }) =>
            isActive ? 'navbar__link navbar__link--active' : 'navbar__link'
          }
        >
          Info
        </NavLink>
      </nav>
    </header>
  );
}

