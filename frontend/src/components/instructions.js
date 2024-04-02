import React from 'react';
import './instructions.css';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';

function Instructions() {
  return (
    <div
      style={{
        // paddingLeft: '20px',
        backgroundColor: '#f0f0f0',
        borderRadius: '10px',
      }}
    >
      <div className='instructions'>
        <h3 className='instruction-heading'>Instructions</h3>
        <p className='instruction-text'>
          This application allows you to upload an image for tumor detection.
          Follow these steps to use the application:
        </p>

        <List sx={{ listStyleType: 'disc' }} className='instruction-list'>
          <ListItem sx={{ display: 'list-item' }}>
            Click on the "Upload Image" button.
          </ListItem>
          <ListItem sx={{ display: 'list-item', paddingLeft: '20px' }}>
            Select an image file from your device.
          </ListItem>
          <ListItem sx={{ display: 'list-item' }}>
            The uploaded image will be analyzed and the results will be
            displayed on the screen.
          </ListItem>
          <ListItem sx={{ display: 'list-item' }}>
            Click on the "Change Image" button to upload a different image.
          </ListItem>
        </List>
      </div>
    </div>
  );
}

export default Instructions;
