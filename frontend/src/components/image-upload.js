import React, { useState, useRef } from 'react';
import defaultImage from '../images/upload.jpg'; // Import the default image
import './image-uploader.css'; // Import CSS file for styling
import { getPredictions } from '../api/api';

function ImageUploader({ setImageState, setData }) {
  // State to store the uploaded image
  const [image, setImage] = useState(null);
  // Ref to reference the file input element
  const fileInputRef = useRef(null);

  // Function to handle file input change
  const handleFileInputChange = async (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = async () => {
        setImage(reader.result);
        setImageState(reader.result);
        try {
          const result = await getPredictions({
            image: reader.result,
            name: 'John Doe',
            age: '18',
          });
          setData(result);
        } catch (error) {
          console.error('Error while getting predictions:', error);
          // Handle error (e.g., show error message to user)
        }
      };
      reader.readAsDataURL(file);
    }
  };
  // Function to trigger file input click
  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  // Function to clear the uploaded image
  const clearImage = () => {
    setImage(null);
    setImageState(null);
    setData(null);
  };

  return (
    <div className='image-uploader'>
      <h2 className='upload-heading'>Upload Image</h2>
      {/* Hidden input field for uploading image */}
      <input
        type='file'
        accept='image/*'
        onChange={handleFileInputChange}
        ref={fileInputRef}
        className='file-input'
      />
      {/* Display default or uploaded image */}
      <img
        src={image ? image : defaultImage}
        alt={image ? 'Uploaded' : 'Default'}
        className='uploaded-image'
        onClick={handleUploadClick} // Trigger file input click when clicked
      />
      {/* Button to clear the uploaded image */}
      {image && (
        <div>
          <button onClick={clearImage} className='clear-button'>
            Change Image
          </button>
        </div>
      )}
    </div>
  );
}

export default ImageUploader;
