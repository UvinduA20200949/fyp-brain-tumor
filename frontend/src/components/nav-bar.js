import React from 'react';
import './nav-bar.css'; // Import CSS file for styling
import logo from '../images/logo.png';

function NavBar() {
  return (
    <nav className='navbar'>
      <img src={logo} alt='Tumor Inspect' className='navbar-logo' />
    </nav>
  );
}

export default NavBar;
