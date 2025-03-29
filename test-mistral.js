require('dotenv').config();
const axios = require('axios');

// Mistral API configuration
const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY;
const MISTRAL_API_URL = 'https://api.mistral.ai/v1/chat/completions';

async function testMistralAPI() {
  try {
    console.log('Testing Mistral API connection...');
    
    // Simple test prompt
    const response = await axios.post(
      MISTRAL_API_URL,
      {
        model: "mistral-large-latest",
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: "Hello, can you confirm this connection is working?" }
        ],
        temperature: 0.7,
        max_tokens: 100
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${MISTRAL_API_KEY}`
        }
      }
    );
    
    console.log('Mistral API Response:');
    console.log(response.data.choices[0].message.content);
    console.log('\nConnection successful!');
  } catch (error) {
    console.error('Error testing Mistral API:');
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    } else if (error.request) {
      // The request was made but no response was received
      console.error('No response received:', error.request);
    } else {
      // Something happened in setting up the request that triggered an Error
      console.error('Error message:', error.message);
    }
    console.error('\nPlease check your API key and connection.');
  }
}

testMistralAPI();
