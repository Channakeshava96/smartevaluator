# Smart Evaluator

A web application that uses OCR and AI to evaluate student answers against teacher answers.

## Deploying to Vercel

### Prerequisites

- A Vercel account
- Google Cloud Vision API credentials
- Gemini API key
- Mistral API key

### Environment Variables

Set the following environment variables in your Vercel project settings:

1. `GEMINI_API_KEY` - Your Google Gemini API key
2. `VISION_API_KEY` - Your Google Vision API key (optional if using service account)
3. `MISTRAL_API_KEY` - Your Mistral API key
4. `GOOGLE_APPLICATION_CREDENTIALS_JSON` - The entire JSON content of your Google Cloud service account key file

### Important Notes for Vercel Deployment

1. **File Uploads**: The application has been modified to use memory storage instead of disk storage for file uploads, as Vercel's serverless functions don't support persistent file storage.

2. **Google Cloud Credentials**: Instead of using a file path for Google Cloud credentials, you need to provide the entire JSON content as an environment variable.

3. **Function Resources**: The Vercel configuration includes settings for memory (1024 MB) and execution duration (60 seconds) to accommodate the AI processing requirements.

4. **Troubleshooting**:
   - If you encounter errors, check the Vercel logs for detailed information
   - Ensure all environment variables are correctly set
   - For Vision API issues, verify your Google Cloud credentials

## Local Development

1. Clone the repository
2. Install dependencies: `npm install`
3. Create a `.env` file with the required environment variables:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   VISION_API_KEY=your_vision_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-file.json
   ```
4. Run the application: `npm start`
