import React from 'react';
import './result.css';

function Result({ category, description }) {
  return (
    <div className='result-container'>
      <h3 className='result-category'>Category: {category}</h3>
      <p className='result-description'>{description}</p>
    </div>
  );
}

export default Result;
