import React from 'react';
import './nav-bar.css'; // Import CSS file for styling
import logo from '../images/logo.png';

function NavBar() {
  return (
    <nav className='navbar'>
      <img src={logo} alt='Rad Sult' className='navbar-logo' />
    </nav>
  );
}

export default NavBar;
