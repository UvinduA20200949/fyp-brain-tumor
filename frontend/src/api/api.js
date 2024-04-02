import axios from 'axios';

const BASE_URL = 'http://127.0.0.1:5000';

export const getPredictions = async ({ name, age, image }) => {
  try {
    image = image.split(',')[1];
    const response = await axios.post(`${BASE_URL}/get-prediction`, {
      name,
      age,
      image,
    });
    // console.log(response);
    // console.log(response.data);
    return response.data;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
};
