import logo from './logo.svg';
import './App.css';
import NavBar from './components/nav-bar';
import ImageUploader from './components/image-upload';
import React, { useEffect, useState } from 'react';
import Stack from '@mui/material/Stack';
import Divider from '@mui/material/Divider';
import Instructions from './components/instructions';
import ImageShow from './components/image-show';
import Result from './components/result';
import tempimage from './images/after.jpg';
import { CircularProgress } from '@mui/material';

function App() {
  const [imageState, setImageState] = useState(null);
  const [segmentedImageState, setSegmentedImageState] = useState(null);
  const [category, setCategory] = useState('Loading');
  const [description, setDescription] = useState('Loading');
  const [data, setData] = useState(null);

  useEffect(() => {
    // Update category and description based on data
    if (data) {
      const base64String = data.segmentedImage;
      const imageUrl = `data:image/jpeg;base64,${base64String}`;

      setSegmentedImageState(imageUrl);
      // console.log(segmentedImageState);
      setCategory(data.prediction);
      setDescription(data.description);
    } else {
      setSegmentedImageState(null);
      setCategory('Loading');
      setDescription('Loading');
    }
  }, [data]);

  return (
    <div className='App'>
      <NavBar />
      <Stack
        direction={'row'}
        justifyContent={'space-evenly'}
        style={{ marginTop: '30px', marginBottom: '30px' }}
      >
        {imageState === null && <Instructions />}
        <ImageUploader setImageState={setImageState} setData={setData} />
        {imageState && segmentedImageState === null && <CircularProgress />}
        {imageState && segmentedImageState && (
          <ImageShow segmentedImage={segmentedImageState} />
        )}
      </Stack>
      {imageState && <Divider>RESULT</Divider>}
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          paddingTop: '20px',
        }}
      >
        {imageState && <Result category={category} description={description} />}
      </div>
    </div>
  );
}

export default App;
