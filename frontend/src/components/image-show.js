import React, { useState, useRef } from 'react';
import defaultImage from '../images/upload.jpg'; // Import the default image
import './image-show.css'; // Import CSS file for styling

function ImageShow({ segmentedImage }) {
  return (
    <div className='image-show'>
      <h2 className='image-heading'>Segmented Image</h2>
      {/* Display default or uploaded image */}
      <img
        src={segmentedImage ? segmentedImage : defaultImage}
        alt={segmentedImage ? 'Uploaded' : 'Default'}
        className='segmented-image'
      />
    </div>
  );
}

export default ImageShow;
