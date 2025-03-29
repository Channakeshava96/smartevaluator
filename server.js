const express = require('express');
const multer = require('multer');
const vision = require('@google-cloud/vision');
const path = require('path');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const axios = require('axios');
const cheerio = require('cheerio');
require('dotenv').config();  // To load environment variables from .env file
const fs = require('fs');
const { spawn } = require('child_process');

// API keys
const VISION_API_KEY = process.env.VISION_API_KEY;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY;
const MISTRAL_API_URL = 'https://api.mistral.ai/v1/chat/completions';

// Initialize the Google Vision client
const client = new vision.ImageAnnotatorClient({
    keyFilename: process.env.GOOGLE_APPLICATION_CREDENTIALS,  // Use path to service account JSON file
  });

// Initialize the Google Generative AI client
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

const app = express();

// Middleware to parse JSON bodies
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
 
// Store the extracted data
const extractedData = {
  student: [],
  teacher: []
};

// Serve the frontend HTML file
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Serve the openingpage.html file when specifically requested
app.get('/openingpage.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'openingpage.html'));
});

const upload = multer({ dest: 'uploads/' });

// Function to extract question numbers and answers from OCR text
function extractQuestionsAndAnswers(textAnnotations) {
  if (!textAnnotations || textAnnotations.length === 0) {
    return { success: false, message: 'No text detected' };
  }

  // Get full text and blocks with their bounding boxes
  const fullText = textAnnotations[0].description;
  const blocks = textAnnotations.slice(1); // Skip the first element which contains the entire text
  
  // Log the raw text to the terminal
  console.log('\n==== RAW EXTRACTED TEXT ====');
  console.log(fullText);
  console.log('============================\n');
  
  // Store detected questions and answers
  const questionsAndAnswers = [];
  
  // FACTOR 1: Enhanced pattern recognition for question numbers
  // This regex is designed to match question numbers at the START of a line
  // It handles various formats: "1", "1.", "1)", "(1)", "Q1", etc.
  // The ^ anchor ensures we only match at the beginning of a line
  const questionNumberRegex = /^(?:\s*(?:Q\.?|Question)?\s*)?(\d+)[\.\)\s\:\-]|\s*\((\d+)\)|\s*(\d+)[a-z]?[\.\)]/i;
  
  // FACTOR 2: Analyze spatial layout if available
  let leftMarginPositions = [];
  let leftMarginThreshold = null;
  let hasLayoutInfo = false;
  
  // Try to determine the left margin for question numbers based on bounding boxes
  if (blocks.length > 0 && blocks[0].boundingPoly && blocks[0].boundingPoly.vertices) {
    hasLayoutInfo = true;
    
    // Collect all left margin positions
    blocks.forEach(block => {
      if (block.boundingPoly && block.boundingPoly.vertices && block.boundingPoly.vertices.length > 0) {
        const leftX = block.boundingPoly.vertices[0].x;
        leftMarginPositions.push(leftX);
      }
    });
    
    // Sort positions and find the most common left margin
    if (leftMarginPositions.length > 0) {
      // Group positions that are within 10 pixels of each other
      const groupedPositions = {};
      leftMarginPositions.forEach(pos => {
        const roundedPos = Math.round(pos / 10) * 10; // Round to nearest 10 pixels
        groupedPositions[roundedPos] = (groupedPositions[roundedPos] || 0) + 1;
      });
      
      // Find the most common left margin position
      let mostCommonPosition = 0;
      let highestCount = 0;
      
      for (const [position, count] of Object.entries(groupedPositions)) {
        if (count > highestCount) {
          highestCount = count;
          mostCommonPosition = parseInt(position);
        }
      }
      
      // Set the left margin threshold with a small tolerance
      leftMarginThreshold = mostCommonPosition + 15; // 15 pixels tolerance
      console.log(`Detected common left margin around: ${mostCommonPosition}px`);
    }
  }
  
  // FACTOR 3: Split text by newlines and analyze line by line
  const lines = fullText.split('\n');
  let currentQuestion = null;
  let currentAnswer = [];
  let emptyLineCount = 0;
  let previousLineWasEmpty = false;
  let lineIndentations = [];
  let previousLineIndentation = 0;
  let averageAnswerIndentation = 0;
  let answerIndentationCount = 0;
  
  // First pass: collect line indentation information
  if (hasLayoutInfo) {
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      
      // Find the block that corresponds to this line
      const blockForLine = blocks.find(b => b.description.trim() === line);
      if (blockForLine && blockForLine.boundingPoly && blockForLine.boundingPoly.vertices) {
        const leftX = blockForLine.boundingPoly.vertices[0].x;
        lineIndentations[i] = leftX;
      }
    }
  }
  
  // Second pass: identify question boundaries using multiple factors
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    // FACTOR 4: Empty line detection
    if (!line) {
      emptyLineCount++;
      previousLineWasEmpty = true;
      continue;
    }
    
    // Calculate a confidence score for this line being a new question
    let newQuestionConfidence = 0;
    
    // FACTOR 5: Check if this line starts with a question number pattern
    const match = line.match(questionNumberRegex);
    if (match) {
      newQuestionConfidence += 50; // Strong indicator
    }
    
    // FACTOR 6: Check if this line follows an empty line
    if (previousLineWasEmpty) {
      newQuestionConfidence += 30; // Good indicator
    }
    
    // FACTOR 7: Check if this line starts at the left margin
    let currentIndentation = 0;
    let isAtLeftMargin = false;
    
    if (hasLayoutInfo && lineIndentations[i]) {
      currentIndentation = lineIndentations[i];
      isAtLeftMargin = currentIndentation <= leftMarginThreshold;
      
      if (isAtLeftMargin) {
        newQuestionConfidence += 20; // Supporting indicator
      }
    }
    
    // FACTOR 8: Check if this line contains a standalone number at the beginning
    // This helps with formats like "1 Answer text" without punctuation
    if (!match && /^\s*\d+\s+[A-Z]/.test(line)) {
      newQuestionConfidence += 40; // Strong indicator for number followed by capital letter
    }
    
    // FACTOR 9: Check for digits that are NOT question numbers
    // If a digit appears in the middle of a line and not at the beginning, it's likely part of the answer
    if (!match && /\s\d+\s/.test(line) && !/^\s*\d+/.test(line)) {
      newQuestionConfidence -= 20; // Negative indicator - likely not a question number
    }
    
    // FACTOR 10: Check for continuation of previous answer
    // If this line has similar indentation to previous answer lines, it's likely a continuation
    if (currentQuestion !== null && hasLayoutInfo && lineIndentations[i] && averageAnswerIndentation > 0) {
      // Check if indentation is similar to average answer indentation (within 15 pixels)
      const isAnswerIndentation = Math.abs(lineIndentations[i] - averageAnswerIndentation) < 15;
      
      if (isAnswerIndentation && !isAtLeftMargin) {
        newQuestionConfidence -= 30; // Strong negative indicator - likely continuation of answer
      }
    }
    
    // FACTOR 11: Check for sentence continuation
    // If the line starts with lowercase and previous line doesn't end with period, it's likely a continuation
    if (currentQuestion !== null && currentAnswer.length > 0) {
      const previousLine = currentAnswer[currentAnswer.length - 1];
      const endsWithPunctuation = /[.!?]$/.test(previousLine);
      const startsWithLowercase = /^[a-z]/.test(line);
      
      if (!endsWithPunctuation && startsWithLowercase) {
        newQuestionConfidence -= 25; // Negative indicator - likely continuation of previous sentence
      }
    }
    
    // Determine if this is likely a new question based on combined factors
    const isLikelyNewQuestion = newQuestionConfidence >= 40; // Threshold for confidence
    
    if (isLikelyNewQuestion) {
      // If we were already collecting an answer, save it
      if (currentQuestion !== null && currentAnswer.length > 0) {
        questionsAndAnswers.push({
          questionNumber: currentQuestion,
          answer: currentAnswer.join(' '),
          confidence: newQuestionConfidence
        });
      }
      
      // Start a new question
      if (match) {
        // Get the question number from the regex match
        currentQuestion = match[1] || match[2] || match[3];
      } else {
        // For other patterns like "1 Answer text", extract the number
        const numMatch = line.match(/^\s*(\d+)/);
        if (numMatch) {
          currentQuestion = numMatch[1];
        }
      }
      
      // Start collecting the answer
      // If the line contains more than just the question number, add it to the answer
      const answerText = line.replace(questionNumberRegex, '').trim();
      if (answerText) {
        currentAnswer = [answerText];
      } else {
        currentAnswer = [];
      }
      
      // Reset indentation tracking for this answer
      averageAnswerIndentation = currentIndentation;
      answerIndentationCount = 1;
    } else {
      // This is a continuation of the current answer
      if (currentQuestion !== null) {
        currentAnswer.push(line);
        
        // Update average answer indentation
        if (hasLayoutInfo && lineIndentations[i]) {
          averageAnswerIndentation = (averageAnswerIndentation * answerIndentationCount + lineIndentations[i]) / (answerIndentationCount + 1);
          answerIndentationCount++;
        }
      }
    }
    
    previousLineWasEmpty = false;
  }
  
  // Don't forget to add the last question/answer if we have one
  if (currentQuestion !== null && currentAnswer.length > 0) {
    questionsAndAnswers.push({
      questionNumber: currentQuestion,
      answer: currentAnswer.join(' ')
    });
  }
  
  // Sort by question number (numeric sort)
  const processedQA = questionsAndAnswers.sort((a, b) => {
    return parseInt(a.questionNumber) - parseInt(b.questionNumber);
  });
  
  // Log the structured output
  console.log('\n==== STRUCTURED OUTPUT ====');
  processedQA.forEach(qa => {
    console.log(`Q${qa.questionNumber}: ${qa.answer.substring(0, 50)}${qa.answer.length > 50 ? '...' : ''}`);
  });
  console.log('=======================================\n');
  
  return {
    success: true,
    questionsAndAnswers: processedQA
  };
}

app.post('/upload', upload.single('image'), async (req, res) => {
  try {
    const filePath = path.join(__dirname, req.file.path);
    const uploadType = req.body.type || 'student'; // Default to student if not specified

    // Perform document text detection using Google Vision API
    // This gives more detailed information about text layout compared to simple text detection
    const [result] = await client.documentTextDetection(filePath);
    
    // Process the result to extract questions and answers
    const processedResult = extractQuestionsAndAnswers(result.textAnnotations);
    
    // Store the extracted data in the appropriate variable
    if (processedResult.success && processedResult.questionsAndAnswers) {
      extractedData[uploadType] = processedResult.questionsAndAnswers;
      
      // Log the stored data
      console.log(`\n==== STORED ${uploadType.toUpperCase()} DATA ====`);
      console.log(JSON.stringify(extractedData[uploadType], null, 2));
      console.log('=======================================\n');
    }
    
    res.json(processedResult);
  } catch (error) {
    console.error(error);
    res.status(500).send('Error processing the image');
  }
});

// Endpoint to get the stored data
app.get('/api/data', (req, res) => {
  res.json(extractedData);
});

// Endpoint to compare student and teacher answers
app.post('/api/compare', (req, res) => {
  try {
    // Check if we have both student and teacher data
    if (extractedData.student.length === 0 || extractedData.teacher.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'Both student and teacher answers must be uploaded before comparison'
      });
    }

    // Get evaluation type from request body
    const evaluationType = req.body.evaluationType || 'concept-based';
    const contextType = req.body.contextType || 'case-study';
    
    console.log(`Evaluation type: ${evaluationType}, Context type: ${contextType}`);

    if (evaluationType === 'concept-based') {
      // Use the existing Python script for concept-based evaluation
      // Get absolute path to the Python script
      const pythonScriptPath = path.join(__dirname, 'miniproject2.py');
      
      // Log for debugging
      console.log(`Executing Python script at: ${pythonScriptPath}`);
      console.log('Data being sent to Python script:', JSON.stringify(extractedData, null, 2));

      // Spawn the Python process with the full path
      const pythonProcess = spawn('python', [pythonScriptPath]);
      
      // Send the data to the Python script
      pythonProcess.stdin.write(JSON.stringify(extractedData));
      pythonProcess.stdin.end();
      
      let resultData = '';
      let errorData = '';
      
      // Collect data from the Python script
      pythonProcess.stdout.on('data', (data) => {
        console.log(`Python stdout: ${data}`);
        resultData += data.toString();
      });
      
      // Handle errors
      pythonProcess.stderr.on('data', (data) => {
        console.error(`Python stderr: ${data}`);
        errorData += data.toString();
      });
      
      // When the process is complete
      pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        
        if (code !== 0) {
          console.error('Python script error:', errorData);
          return res.status(500).json({
            success: false,
            message: 'Error during similarity comparison',
            error: errorData
          });
        }
        
        try {
          // Parse the JSON result from the Python script
          const comparisonResults = JSON.parse(resultData);
          
          // Return the results
          res.json({
            success: true,
            results: comparisonResults
          });
        } catch (error) {
          console.error('Error parsing Python script output:', error);
          console.error('Raw output:', resultData);
          res.status(500).json({
            success: false,
            message: 'Error parsing comparison results',
            error: error.toString(),
            rawOutput: resultData
          });
        }
      });
    } else if (evaluationType === 'context-based') {
      // Use Mistral API for context-based evaluation
      evaluateWithMistral(extractedData, contextType)
        .then(results => {
          res.json({
            success: true,
            results: results
          });
        })
        .catch(error => {
          console.error('Error in Mistral evaluation:', error);
          res.status(500).json({
            success: false,
            message: 'Error during Mistral evaluation',
            error: error.toString()
          });
        });
    } else {
      res.status(400).json({
        success: false,
        message: 'Invalid evaluation type'
      });
    }
  } catch (error) {
    console.error('Error in comparison endpoint:', error);
    res.status(500).json({
      success: false,
      message: 'Server error during comparison',
      error: error.toString()
    });
  }
});

// New endpoint to get AI feedback on student answers
app.post('/api/feedback', async (req, res) => {
  try {
    const { questionNumber, studentAnswer, teacherAnswer, similarityScore, evaluationType, contextType } = req.body;
    
    if (!studentAnswer || !teacherAnswer) {
      return res.status(400).json({ 
        success: false, 
        message: 'Both student and teacher answers are required' 
      });
    }

    console.log(`Getting AI feedback for question ${questionNumber}`);
    console.log(`Student answer length: ${studentAnswer.length}`);
    console.log(`Teacher answer length: ${teacherAnswer.length}`);
    console.log(`Similarity score: ${similarityScore}`);
    console.log(`Evaluation type: ${evaluationType}`);
    console.log(`Context type: ${contextType}`);

    // Generate AI feedback
    const feedback = await getAIFeedback(questionNumber, studentAnswer, teacherAnswer, similarityScore, evaluationType, contextType);
    
    // Find learning resources through web scraping
    const resources = await findLearningResources(feedback.topicsToImprove);
    
    // Return combined feedback and resources
    res.json({
      success: true,
      feedback,
      resources
    });
  } catch (error) {
    console.error('Error generating feedback:', error);
    res.status(500).json({
      success: false,
      message: 'Error generating feedback',
      error: error.toString()
    });
  }
});

// Function to get AI feedback using Gemini API
async function getAIFeedback(questionNumber, studentAnswer, teacherAnswer, similarityScore, evaluationType, contextType) {
  try {
    console.log("Getting AI feedback for question:", questionNumber);
    console.log("Student answer length:", studentAnswer.length);
    console.log("Teacher answer length:", teacherAnswer.length);
    console.log("Similarity score:", similarityScore);
    console.log("Evaluation type:", evaluationType);
    console.log("Context type:", contextType);
    
    // Initialize the Gemini API with safety settings turned down
    const model = genAI.getGenerativeModel({ 
      model: "gemini-1.5-pro",
      generationConfig: {
        temperature: 0.7,
        topP: 0.9,
        topK: 40,
        maxOutputTokens: 2048,
      },
      safetySettings: [
        {
          category: "HARM_CATEGORY_HARASSMENT",
          threshold: "BLOCK_ONLY_HIGH"
        },
        {
          category: "HARM_CATEGORY_HATE_SPEECH",
          threshold: "BLOCK_ONLY_HIGH"
        },
        {
          category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
          threshold: "BLOCK_ONLY_HIGH"
        },
        {
          category: "HARM_CATEGORY_DANGEROUS_CONTENT",
          threshold: "BLOCK_ONLY_HIGH"
        }
      ]
    });
    
    // First, analyze the subject area to ensure we have accurate classification
    const subjectPrompt = `
      You are an expert educational assessment AI specializing in academic subject classification.
      
      Analyze the following student answer and teacher's reference answer to determine the SPECIFIC academic subject, sub-topic, and educational level.
      
      Student Answer:
      "${studentAnswer}"
      
      Teacher's Reference Answer:
      "${teacherAnswer}"
      
      Based ONLY on the content of these answers:
      1. Identify the MAIN academic subject (e.g., Mathematics, Physics, Biology, History)
      2. Identify the SPECIFIC sub-topic within that subject (e.g., Calculus - Integration, Organic Chemistry - Alkene Reactions)
      3. Determine the likely educational level (e.g., High School, Undergraduate, Graduate)
      
      IMPORTANT: Be as precise and specific as possible. Analyze the terminology, concepts, and knowledge domains present in both answers.
      DO NOT provide generic classifications if the content clearly belongs to a specific subject area.
      
      Format your response as a structured JSON object with these keys:
      "mainSubject" (string)
      "specificTopic" (string)
      "educationalLevel" (string)
      
      Be as precise and detailed as possible. Avoid generic classifications.
    `;
    
    // Get subject analysis first
    let subjectData = {
      mainSubject: "General academics",
      specificTopic: "General knowledge",
      educationalLevel: "Unknown"
    };
    
    try {
      const subjectResult = await model.generateContent(subjectPrompt);
      const subjectResponse = subjectResult.response;
      const subjectText = subjectResponse.text();
      
      console.log("Subject analysis response:", subjectText.substring(0, 300) + "...");
      
      // Try to parse the subject data with improved error handling
      try {
        // Direct parsing if it's already well-formatted JSON
        const parsedData = JSON.parse(subjectText.trim());
        if (parsedData.mainSubject && parsedData.specificTopic) {
          // Validate the subject data is not too generic
          if (parsedData.mainSubject !== "General academics" && 
              parsedData.mainSubject !== "General" && 
              parsedData.specificTopic !== "General knowledge" &&
              parsedData.specificTopic !== "General") {
            subjectData = parsedData;
            console.log("Successfully parsed subject data:", JSON.stringify(subjectData));
          } else {
            console.log("Parsed subject data was too generic, attempting deeper analysis");
            // Additional analysis could be added here
          }
        }
      } catch (e) {
        // Look for JSON blocks with improved regex
        const jsonMatch = subjectText.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/) || 
                         subjectText.match(/{[\s\S]*?}/) || 
                         subjectText.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const jsonStr = jsonMatch[1] || jsonMatch[0];
          try {
            const parsedData = JSON.parse(jsonStr.trim());
            if (parsedData.mainSubject && parsedData.specificTopic) {
              // Validate the subject data is not too generic
              if (parsedData.mainSubject !== "General academics" && 
                  parsedData.mainSubject !== "General" && 
                  parsedData.specificTopic !== "General knowledge" &&
                  parsedData.specificTopic !== "General") {
                subjectData = parsedData;
                console.log("Successfully parsed subject data from markdown block:", JSON.stringify(subjectData));
              } else {
                console.log("Parsed subject data from markdown was too generic");
              }
            }
          } catch (innerError) {
            console.error("Error parsing JSON from subject response:", innerError);
          }
        } else {
          // Try regex extraction as a fallback with improved patterns
          const mainSubjectMatch = subjectText.match(/["']?mainSubject["']?\s*:\s*["'](.+?)["']/i) || 
                                  subjectText.match(/main\s*subject\s*:\s*(.+?)(?:\n|$)/i) ||
                                  subjectText.match(/subject\s*:\s*(.+?)(?:\n|$)/i);
          const specificTopicMatch = subjectText.match(/["']?specificTopic["']?\s*:\s*["'](.+?)["']/i) || 
                                    subjectText.match(/specific\s*topic\s*:\s*(.+?)(?:\n|$)/i) ||
                                    subjectText.match(/sub-topic\s*:\s*(.+?)(?:\n|$)/i);
          const educationalLevelMatch = subjectText.match(/["']?educationalLevel["']?\s*:\s*["'](.+?)["']/i) || 
                                       subjectText.match(/educational\s*level\s*:\s*(.+?)(?:\n|$)/i) ||
                                       subjectText.match(/level\s*:\s*(.+?)(?:\n|$)/i);
          
          if (mainSubjectMatch && mainSubjectMatch[1]) {
            const subject = mainSubjectMatch[1].trim();
            if (subject !== "General academics" && subject !== "General") {
              subjectData.mainSubject = subject;
            }
          }
          if (specificTopicMatch && specificTopicMatch[1]) {
            const topic = specificTopicMatch[1].trim();
            if (topic !== "General knowledge" && topic !== "General") {
              subjectData.specificTopic = topic;
            }
          }
          if (educationalLevelMatch && educationalLevelMatch[1]) {
            subjectData.educationalLevel = educationalLevelMatch[1].trim();
          }
          
          console.log("Extracted subject data via regex:", JSON.stringify(subjectData));
        }
      }
      
      // If we still have generic subjects, try to extract from the full text
      if (subjectData.mainSubject === "General academics" || subjectData.specificTopic === "General knowledge") {
        // Look for subject mentions in the text
        const subjectMatches = subjectText.match(/(?:subject|topic|field|discipline|area)\s+(?:is|appears to be|seems to be)\s+([^,.]+)/gi);
        if (subjectMatches && subjectMatches.length > 0) {
          const bestMatch = subjectMatches[0].replace(/(?:subject|topic|field|discipline|area)\s+(?:is|appears to be|seems to be)\s+/i, '').trim();
          if (bestMatch && bestMatch.length > 3) {
            subjectData.mainSubject = bestMatch;
            console.log("Extracted subject from text patterns:", bestMatch);
          }
        }
      }
    } catch (subjectError) {
      console.error("Error in subject area analysis:", subjectError);
    }
    
    // Combine subject and topic for a comprehensive subject area
    const detectedSubjectArea = subjectData.specificTopic && subjectData.specificTopic !== "General knowledge" 
      ? `${subjectData.mainSubject} - ${subjectData.specificTopic}`
      : subjectData.mainSubject;
    
    console.log("Detected subject area:", detectedSubjectArea);
    
    // Choose the appropriate prompt based on evaluation type and context type
    let prompt;
    
    if (evaluationType === 'context-based') {
      if (contextType === 'letter-writing') {
        prompt = `
          You are an expert educational assessment AI specializing in English language and letter writing evaluation.
          
          Your task is to analyze a student's letter writing answer compared to a teacher's reference answer.
          
          Question Number: ${questionNumber}
          
          Student Answer:
          "${studentAnswer}"
          
          Teacher's Reference Answer:
          "${teacherAnswer}"
          
          Similarity Score: ${similarityScore}%
          
          Provide a detailed analysis of the student's letter writing skills using the following categories:
          
          1. Introduction (10 points): Evaluate the greeting, opening paragraph, and purpose statement
          2. Content & Arguments (40 points): Evaluate the main body, ideas, supporting details, and persuasiveness
          3. Structure & Organization (20 points): Evaluate paragraph structure, logical flow, and coherence
          4. Language & Grammar (15 points): Evaluate vocabulary, sentence structure, grammar, and punctuation
          5. Conclusion (15 points): Evaluate the closing paragraph, summary, and sign-off
          
          For each category, provide:
          - Specific strengths
          - Areas for improvement
          - A score out of the available points
          
          Provide the following analysis:
          
          1. Strengths: List what aspects of the student's letter are good (2-3 points)
          2. Areas for Improvement: Identify specific weaknesses or mistakes in the student's letter (2-3 points)
          3. Missing Concepts: List key letter writing elements from the teacher's answer that are missing in the student's answer (2-3 points)
          4. Topics to Study: List 3-5 SPECIFIC letter writing topics the student should review to improve
          5. Subject Area: "English - Letter Writing"
          
          Format your response as a structured JSON object with these keys:
          "strengths" (array of strings)
          "areasForImprovement" (array of strings)
          "missingConcepts" (array of strings)
          "topicsToImprove" (array of strings)
          "subjectArea" (string)
        `;
      } else if (contextType === 'case-study') {
        prompt = `
          You are an expert educational assessment AI specializing in case study analysis.
          
          Your task is to analyze a student's case study answer compared to a teacher's reference answer.
          
          Question Number: ${questionNumber}
          
          Student Answer:
          "${studentAnswer}"
          
          Teacher's Reference Answer:
          "${teacherAnswer}"
          
          Similarity Score: ${similarityScore}%
          
          IMPORTANT: Provide a detailed analysis specific to case study analysis concepts and methodology.
          
          Provide the following analysis:
          
          1. Strengths: List what aspects of the student's case analysis are good (2-3 points)
          2. Areas for Improvement: Identify specific weaknesses or mistakes in the student's analysis (2-3 points)
          3. Missing Concepts: List key case study analysis elements from the teacher's answer that are missing in the student's answer (2-3 points)
          4. Topics to Study: List 3-5 SPECIFIC case study analysis topics the student should review to improve
          5. Subject Area: Confirm the specific academic subject this case study relates to
          
          Format your response as a structured JSON object with these keys:
          "strengths" (array of strings)
          "areasForImprovement" (array of strings)
          "missingConcepts" (array of strings)
          "topicsToImprove" (array of strings)
          "subjectArea" (string)
        `;
      } else if (contextType === 'character-perspective') {
        prompt = `
          You are an expert educational assessment AI specializing in character perspective and descriptive writing.
          
          Your task is to analyze a student's character perspective/descriptive answer compared to a teacher's reference answer.
          
          Question Number: ${questionNumber}
          
          Student Answer:
          "${studentAnswer}"
          
          Teacher's Reference Answer:
          "${teacherAnswer}"
          
          Similarity Score: ${similarityScore}%
          
          IMPORTANT: Provide a detailed analysis specific to character perspective, descriptive writing, and literary analysis.
          
          Provide the following analysis:
          
          1. Strengths: List what aspects of the student's character perspective/description are good (2-3 points)
          2. Areas for Improvement: Identify specific weaknesses or mistakes in the student's writing (2-3 points)
          3. Missing Concepts: List key descriptive/perspective elements from the teacher's answer that are missing in the student's answer (2-3 points)
          4. Topics to Study: List 3-5 SPECIFIC descriptive writing or character analysis topics the student should review to improve
          5. Subject Area: "Literature - Character Analysis and Descriptive Writing"
          
          Format your response as a structured JSON object with these keys:
          "strengths" (array of strings)
          "areasForImprovement" (array of strings)
          "missingConcepts" (array of strings)
          "topicsToImprove" (array of strings)
          "subjectArea" (string)
        `;
      } else {
        // Default context-based prompt
        prompt = `
          You are an expert educational assessment AI specializing in ${detectedSubjectArea} at the ${subjectData.educationalLevel} level. 
          Your task is to analyze a student's answer compared to a teacher's reference answer for a question in ${detectedSubjectArea}.
          
          Question Number: ${questionNumber}
          
          Student Answer:
          "${studentAnswer}"
          
          Teacher's Reference Answer:
          "${teacherAnswer}"
          
          Similarity Score: ${similarityScore}%
          
          IMPORTANT: Provide a detailed analysis specific to ${detectedSubjectArea} concepts and terminology.
          
          Provide the following analysis:
          
          1. Strengths: List what aspects of the student's answer are good (2-3 points), focusing on correct ${detectedSubjectArea} concepts they mentioned
          2. Areas for Improvement: Identify specific weaknesses or mistakes in the student's answer (2-3 points), using proper ${detectedSubjectArea} terminology
          3. Missing Concepts: List key ${detectedSubjectArea} concepts from the teacher's answer that are missing in the student's answer (2-3 points)
          4. Topics to Study: List 3-5 VERY SPECIFIC ${detectedSubjectArea} topics the student should review to improve (be precise and detailed)
          5. Subject Area: Confirm the specific academic subject and sub-topic this question relates to (should be ${detectedSubjectArea} unless you strongly disagree)
          
          Format your response as a structured JSON object with these keys:
          "strengths" (array of strings)
          "areasForImprovement" (array of strings)
          "missingConcepts" (array of strings)
          "topicsToImprove" (array of strings)
          "subjectArea" (string)
        `;
      }
    } else {
      // Default concept-based prompt
      prompt = `
        You are an expert educational assessment AI specializing in ${detectedSubjectArea} at the ${subjectData.educationalLevel} level. 
        Your task is to analyze a student's answer compared to a teacher's reference answer for a question in ${detectedSubjectArea}.
        
        Question Number: ${questionNumber}
        
        Student Answer:
        "${studentAnswer}"
        
        Teacher's Reference Answer:
        "${teacherAnswer}"
        
        Similarity Score: ${similarityScore}%
        
        IMPORTANT: Provide a detailed analysis specific to ${detectedSubjectArea} concepts and terminology.
        
        Provide the following analysis:
        
        1. Strengths: List what aspects of the student's answer are good (2-3 points), focusing on correct ${detectedSubjectArea} concepts they mentioned
        2. Areas for Improvement: Identify specific weaknesses or mistakes in the student's answer (2-3 points), using proper ${detectedSubjectArea} terminology
        3. Missing Concepts: List key ${detectedSubjectArea} concepts from the teacher's answer that are missing in the student's answer (2-3 points)
        4. Topics to Study: List 3-5 VERY SPECIFIC ${detectedSubjectArea} topics the student should review to improve (be precise and detailed)
        5. Subject Area: Confirm the specific academic subject and sub-topic this question relates to (should be ${detectedSubjectArea} unless you strongly disagree)
        
        Format your response as a structured JSON object with these keys:
        "strengths" (array of strings)
        "areasForImprovement" (array of strings)
        "missingConcepts" (array of strings)
        "topicsToImprove" (array of strings)
        "subjectArea" (string)
      `;
    }
    
    // Generate content using Gemini
    const result = await model.generateContent(prompt);
    const response = result.response;
    const text = response.text();
    
    console.log("Gemini API raw response:", text.substring(0, 500) + "...");
    
    // Extract JSON from the response
    let feedbackData;
    
    try {
      // First try: Direct parsing if it's already well-formatted JSON
      feedbackData = JSON.parse(text.trim());
      console.log("Successfully parsed JSON directly");
    } catch (e) {
      console.log("First JSON parse attempt failed, trying to extract JSON from text");
      
      try {
        // Second try: Look for JSON blocks (```json ... ```)
        const jsonMatch = text.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/) || text.match(/{[\s\S]*}/);
        if (jsonMatch) {
          const jsonStr = jsonMatch[1] || jsonMatch[0];
          feedbackData = JSON.parse(jsonStr.trim());
          console.log("Successfully parsed JSON from markdown block");
        } else {
          // Third try: Manual extraction
          feedbackData = extractFeedbackFromText(text);
          console.log("Manually extracted feedback from text");
        }
      } catch (innerError) {
        console.error("Error parsing JSON from response:", innerError);
        // Fallback to manual extraction
        feedbackData = extractFeedbackFromText(text);
        console.log("Fallback to manual extraction after error");
      }
    }
    
    // Validate that we have meaningful feedback
    const isGenericFeedback = checkIfGenericFeedback(feedbackData);
    
    // If we got generic feedback, try one more time with more explicit instructions
    if (isGenericFeedback) {
      console.log("WARNING: Detected generic feedback. Retrying with more specific instructions...");
      
      const retryPrompt = `
        You are an expert educational assessment AI specializing in ${detectedSubjectArea} at the ${subjectData.educationalLevel} level.
        
        You are analyzing a student's answer for Question #${questionNumber} in the field of ${detectedSubjectArea}.
        
        Student Answer:
        "${studentAnswer}"
        
        Teacher's Reference Answer:
        "${teacherAnswer}"
        
        Similarity Score: ${similarityScore}%
        
        CRITICALLY IMPORTANT: You MUST provide SPECIFIC, DETAILED feedback using proper ${detectedSubjectArea} terminology and concepts. DO NOT provide generic feedback that could apply to any answer.
        
        1. Strengths: List SPECIFIC aspects of the student's answer that are good, citing EXACT concepts or points they mentioned correctly using proper ${detectedSubjectArea} terminology
        2. Areas for Improvement: Identify SPECIFIC weaknesses or mistakes in the student's answer, with EXACT references to what they wrote, using proper ${detectedSubjectArea} terminology
        3. Missing Concepts: List SPECIFIC key concepts from the teacher's answer that are missing in the student's answer, citing EXACT terms or ideas from the teacher's answer, using proper ${detectedSubjectArea} terminology
        4. Topics to Study: List 3-5 VERY SPECIFIC ${detectedSubjectArea} topics the student should review to improve, based on the EXACT content of both answers
        5. Subject Area: Confirm the specific academic subject and sub-topic this question relates to (should be ${detectedSubjectArea} unless you strongly disagree).
        
        Format your response as a structured JSON object with these keys:
        "strengths" (array of strings)
        "areasForImprovement" (array of strings)
        "missingConcepts" (array of strings)
        "topicsToImprove" (array of strings)
        "subjectArea" (string)
      `;
      
      try {
        const retryResult = await model.generateContent(retryPrompt);
        const retryResponse = retryResult.response;
        const retryText = retryResponse.text();
        
        console.log("Retry Gemini API raw response:", retryText.substring(0, 500) + "...");
        
        // Try to parse the retry response
        try {
          const retryData = JSON.parse(retryText.trim());
          // Only update the feedback data if we got a valid response
          if (retryData.strengths && retryData.areasForImprovement && retryData.missingConcepts && retryData.topicsToImprove) {
            feedbackData = retryData;
            console.log("Successfully parsed retry JSON directly");
          }
        } catch (e) {
          const jsonMatch = retryText.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/) || retryText.match(/{[\s\S]*}/);
          if (jsonMatch) {
            const jsonStr = jsonMatch[1] || jsonMatch[0];
            try {
              const retryData = JSON.parse(jsonStr.trim());
              // Only update the feedback data if we got a valid response
              if (retryData.strengths && retryData.areasForImprovement && retryData.missingConcepts && retryData.topicsToImprove) {
                feedbackData = retryData;
                console.log("Successfully parsed retry JSON from markdown block");
              }
            } catch (innerError) {
              console.error("Error parsing JSON from retry response:", innerError);
            }
          }
        }
      } catch (retryError) {
        console.error("Error in retry attempt:", retryError);
      }
    }
    
    // If subject area wasn't provided or is generic, use our detected subject area
    if (!feedbackData.subjectArea || 
        feedbackData.subjectArea === "General academics" || 
        feedbackData.subjectArea === "General" ||
        feedbackData.subjectArea.toLowerCase().includes("general")) {
      feedbackData.subjectArea = detectedSubjectArea;
      console.log("Using detected subject area:", detectedSubjectArea);
    }
    
    // If the subject area still looks generic, try to extract it from the feedback content
    if (feedbackData.subjectArea === "General academics" || 
        feedbackData.subjectArea === "General" ||
        feedbackData.subjectArea.toLowerCase().includes("general")) {
      
      // Look for subject-specific terminology in the feedback
      const allFeedbackText = [
        ...(Array.isArray(feedbackData.strengths) ? feedbackData.strengths : []),
        ...(Array.isArray(feedbackData.areasForImprovement) ? feedbackData.areasForImprovement : []),
        ...(Array.isArray(feedbackData.missingConcepts) ? feedbackData.missingConcepts : []),
        ...(Array.isArray(feedbackData.topicsToImprove) ? feedbackData.topicsToImprove : [])
      ].join(' ');
      
      // Common subject area indicators
      const subjectIndicators = {
        "Mathematics": ["equation", "formula", "calculation", "theorem", "proof", "function", "variable"],
        "Physics": ["force", "energy", "motion", "particle", "quantum", "relativity", "momentum"],
        "Chemistry": ["reaction", "compound", "molecule", "element", "bond", "acid", "base"],
        "Biology": ["cell", "organism", "species", "evolution", "gene", "protein", "ecosystem"],
        "History": ["century", "period", "war", "civilization", "empire", "revolution", "dynasty"],
        "Literature": ["author", "novel", "poem", "character", "theme", "narrative", "literary"],
        "Computer Science": ["algorithm", "code", "programming", "data structure", "function", "variable", "class"]
      };
      
      // Check for subject indicators
      let bestSubject = "";
      let maxCount = 0;
      
      for (const [subject, indicators] of Object.entries(subjectIndicators)) {
        let count = 0;
        for (const indicator of indicators) {
          const regex = new RegExp(`\\b${indicator}\\b`, 'gi');
          const matches = allFeedbackText.match(regex);
          if (matches) {
            count += matches.length;
          }
        }
        
        if (count > maxCount) {
          maxCount = count;
          bestSubject = subject;
        }
      }
      
      if (bestSubject && maxCount >= 3) {
        feedbackData.subjectArea = bestSubject;
        console.log("Inferred subject area from terminology:", bestSubject);
      }
    }
    
    // Ensure all required fields exist with proper format
    const formattedFeedback = {
      strengths: Array.isArray(feedbackData.strengths) ? feedbackData.strengths : 
                [feedbackData.strengths || "Good attempt at answering the question"],
      areasForImprovement: Array.isArray(feedbackData.areasForImprovement) ? feedbackData.areasForImprovement : 
                          [feedbackData.areasForImprovement || "Work on being more specific in your answers"],
      missingConcepts: Array.isArray(feedbackData.missingConcepts) ? feedbackData.missingConcepts : 
                      [feedbackData.missingConcepts || "Some key concepts from the reference answer are missing"],
      topicsToImprove: Array.isArray(feedbackData.topicsToImprove) ? feedbackData.topicsToImprove : 
                      [feedbackData.topicsToImprove || "Core subject fundamentals"],
      subjectArea: feedbackData.subjectArea,
      educationalLevel: subjectData.educationalLevel || "Unknown"
    };
    
    console.log("Final formatted feedback:", JSON.stringify(formattedFeedback, null, 2));
    return formattedFeedback;
  } catch (error) {
    console.error('Error calling Gemini API:', error);
    // Return default feedback if API call fails
    return {
      strengths: ["Attempted to answer the question"],
      areasForImprovement: ["Review the reference answer for guidance"],
      missingConcepts: ["Could not analyze missing concepts due to an error"],
      topicsToImprove: ["General subject knowledge", "Foundational concepts"],
      subjectArea: "General academics",
      educationalLevel: "Unknown"
    };
  }
}

// Function to check if feedback is generic/non-specific
function checkIfGenericFeedback(feedback) {
  if (!feedback) return true;
  
  // List of generic phrases that might indicate non-specific feedback
  const genericPhrases = [
    "good attempt", "well done", "nice try", "good understanding", 
    "review the", "study more", "practice more", "be more specific",
    "pay attention", "focus on", "general understanding", "basic concepts",
    "core concepts", "fundamental principles", "key points"
  ];
  
  // Count how many generic phrases appear in the feedback
  let genericCount = 0;
  let totalPhrases = 0;
  
  // Check strengths
  if (Array.isArray(feedback.strengths)) {
    totalPhrases += feedback.strengths.length;
    feedback.strengths.forEach(strength => {
      if (genericPhrases.some(phrase => strength.toLowerCase().includes(phrase))) {
        genericCount++;
      }
    });
  }
  
  // Check areas for improvement
  if (Array.isArray(feedback.areasForImprovement)) {
    totalPhrases += feedback.areasForImprovement.length;
    feedback.areasForImprovement.forEach(area => {
      if (genericPhrases.some(phrase => area.toLowerCase().includes(phrase))) {
        genericCount++;
      }
    });
  }
  
  // Check missing concepts
  if (Array.isArray(feedback.missingConcepts)) {
    totalPhrases += feedback.missingConcepts.length;
    feedback.missingConcepts.forEach(concept => {
      if (genericPhrases.some(phrase => concept.toLowerCase().includes(phrase))) {
        genericCount++;
      }
    });
  }
  
  // Check topics to improve
  if (Array.isArray(feedback.topicsToImprove)) {
    totalPhrases += feedback.topicsToImprove.length;
    feedback.topicsToImprove.forEach(topic => {
      if (genericPhrases.some(phrase => topic.toLowerCase().includes(phrase))) {
        genericCount++;
      }
    });
  }
  
  // If more than 50% of phrases are generic, consider it generic feedback
  return totalPhrases > 0 && (genericCount / totalPhrases) > 0.5;
}

// Function to extract feedback from text if JSON parsing fails
function extractFeedbackFromText(text) {
  const feedback = {
    strengths: [],
    areasForImprovement: [],
    missingConcepts: [],
    topicsToImprove: [],
    subjectArea: "General academics"
  };

  // Define a flexible regex function to handle optional sections
  function matchSection(sectionName) {
    const regex = new RegExp(`${sectionName}:?\\s*\\n?([\\s\\S]*?)(?=\\n[A-Z][a-z]|$)`, 'i');
    const match = text.match(regex);
    return match ? extractTopics(match[1]) : [];
  }

  // Extract each section
  feedback.strengths = matchSection("Strengths");
  feedback.areasForImprovement = matchSection("Areas for Improvement");
  feedback.missingConcepts = matchSection("Missing Concepts");
  feedback.topicsToImprove = matchSection("Topics to Study");

  // Extract subject area separately
  const subjectMatch = text.match(/Subject Area:?\s*\n?([\s\S]*)/i);
  if (subjectMatch && subjectMatch[1].trim()) {
    feedback.subjectArea = subjectMatch[1].trim();
  }

  return feedback;
}

// Function to clean and extract topics from text
function extractTopics(text) {
  if (!text || text.trim().length === 0) return [];

  return text
    .split('\n')
    .map(line => line.trim().replace(/^\d+\.\s*|-\s*|\*\s*/, '')) // Remove bullets/numbers
    .filter(line => line.length > 0);
}

// Function to find learning resources
async function findLearningResources(topics) {
  if (!topics || topics.length === 0) {
    return [{ topic: "General Learning", links: getFallbackResources() }];
  }

  const resources = [];
  const topicsToSearch = topics.slice(0, 3);

  for (const topic of topicsToSearch) {
    let topicResources = [];

    try {
      console.log(`Searching resources for: ${topic}`);

      // Attempt Khan Academy first
      topicResources = await searchKhanAcademy(topic);

      // If no results, try specialized sites
      if (topicResources.length === 0) {
        topicResources = await searchSpecializedSites(topic);
      }

      // If still empty, use fallback resources
      if (topicResources.length === 0) {
        topicResources = getFallbackResources(topic);
      }

    } catch (error) {
      console.error(`Error searching resources for ${topic}:`, error);
      topicResources = getFallbackResources(topic);
    }

    resources.push({
      topic,
      links: topicResources.slice(0, 3)
    });
  }

  console.log("Final Resources Found:", JSON.stringify(resources, null, 2));
  return resources;
}

// Function to search Khan Academy
async function searchKhanAcademy(topic) {
  try {
    const searchQuery = encodeURIComponent(`${topic}`);
    const searchUrl = `https://www.khanacademy.org/search?page_search_query=${searchQuery}`;
    
    console.log(`Fetching from Khan Academy URL: ${searchUrl}`);
    
    const response = await axios.get(searchUrl, {
      timeout: 10000,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
      }
    });
    
    const $ = cheerio.load(response.data);
    
    const links = [];
    $('.result-container a, .link_1uvuyao-o_O-computing_1nblrap a').each((i, element) => {
      const url = $(element).attr('href');
      const title = $(element).text().trim();
      
      if (url && title && title.length > 5 && !links.some(link => link.url === url)) {
        const fullUrl = url.startsWith('/') ? `https://www.khanacademy.org${url}` : url;
        links.push({
          title: title,
          url: fullUrl
        });
      }
    });
    
    return links;
  } catch (error) {
    console.error(`Error searching Khan Academy for topic ${topic}:`, error);
    return [];
  }
}

// Function to search specialized educational sites based on topic
async function searchSpecializedSites(topic) {
  const links = [];
  
  // Determine appropriate specialized sites based on topic keywords
  let specializedSites = [];
  
  if (/math|algebra|calculus|geometry|trigonometry/i.test(topic)) {
    specializedSites = [
      { name: "Khan Academy Math", url: `https://www.khanacademy.org/math/search?search_again=1&item_types=all&page_search_query=${encodeURIComponent(topic)}` },
      { name: "Paul's Online Math Notes", url: "https://tutorial.math.lamar.edu/" },
      { name: "Wolfram MathWorld", url: "https://mathworld.wolfram.com/" }
    ];
  } else if (/physics|mechanics|thermodynamics|electricity|magnetism|quantum/i.test(topic)) {
    specializedSites = [
      { name: "Physics Classroom", url: "https://www.physicsclassroom.com/" },
      { name: "HyperPhysics", url: "http://hyperphysics.phy-astr.gsu.edu/hbase/index.html" },
      { name: "MIT OpenCourseWare Physics", url: "https://ocw.mit.edu/search/?q=physics" }
    ];
  } else if (/chemistry|molecule|atom|reaction|element|compound|organic|inorganic/i.test(topic)) {
    specializedSites = [
      { name: "Chemistry LibreTexts", url: "https://chem.libretexts.org/" },
      { name: "Royal Society of Chemistry", url: "https://edu.rsc.org/" },
      { name: "ChemGuide", url: "https://www.chemguide.co.uk/" }
    ];
  } else if (/biology|cell|organism|gene|dna|evolution|ecosystem|protein|enzyme/i.test(topic)) {
    specializedSites = [
      { name: "Biology Online", url: "https://www.biology-online.org/" },
      { name: "Nature Education", url: "https://www.nature.com/scitable" },
      { name: "Khan Academy Biology", url: "https://www.khanacademy.org/science/biology" }
    ];
  } else if (/history|civilization|war|revolution|century/i.test(topic)) {
    specializedSites = [
      { name: "History.com", url: "https://www.history.com/" },
      { name: "BBC History", url: "https://www.bbc.co.uk/history" },
      { name: "Khan Academy World History", url: "https://www.khanacademy.org/humanities/world-history" }
    ];
  } else if (/english|literature|grammar|writing|essay/i.test(topic)) {
    specializedSites = [
      { name: "Purdue OWL", url: "https://owl.purdue.edu/owl/purdue_owl.html" },
      { name: "LitCharts", url: "https://www.litcharts.com/" },
      { name: "English Grammar", url: "https://www.englishgrammar.org/" }
    ];
  } else if (/computer|algorithm|code|program|software/i.test(topic)) {
    specializedSites = [
      { name: "W3Schools", url: "https://www.w3schools.com/" },
      { name: "GeeksforGeeks", url: `https://www.geeksforgeeks.org/search/?q=${encodeURIComponent(topic)}` },
      { name: "MDN Web Docs", url: "https://developer.mozilla.org/en-US/" }
    ];
  }
  
  // If we have specialized sites, add them to links
  if (specializedSites.length > 0) {
    specializedSites.forEach(site => {
      links.push({
        title: site.name,
        url: site.url
      });
    });
  }
  
  return links;
}

// Function to get fallback educational resources based on topic
function getFallbackResources(topic) {
  // Default resources if all else fails
  let fallbackLinks = [
    { title: "Khan Academy", url: "https://www.khanacademy.org/" },
    { title: "Coursera", url: "https://www.coursera.org/" }
  ];
  
  // Provide topic-specific fallbacks
  if (/math|algebra|calculus|geometry|trigonometry/i.test(topic)) {
    fallbackLinks = [
      { title: "Khan Academy Math", url: "https://www.khanacademy.org/math" },
      { title: "Math is Fun", url: "https://www.mathsisfun.com/" }
    ];
  } else if (/physics|mechanics|thermodynamics|electricity|magnetism/i.test(topic)) {
    fallbackLinks = [
      { title: "Khan Academy Physics", url: "https://www.khanacademy.org/science/physics" },
      { title: "Physics Classroom", url: "https://www.physicsclassroom.com/" }
    ];
  } else if (/chemistry|molecule|atom|reaction|element|compound/i.test(topic)) {
    fallbackLinks = [
      { title: "Khan Academy Chemistry", url: "https://www.khanacademy.org/science/chemistry" },
      { title: "Chemistry LibreTexts", url: "https://chem.libretexts.org/" }
    ];
  } else if (/biology|cell|organism|gene|dna|evolution/i.test(topic)) {
    fallbackLinks = [
      { title: "Khan Academy Biology", url: "https://www.khanacademy.org/science/biology" },
      { title: "Biology Online", url: "https://www.biology-online.org/" }
    ];
  } else if (/history|civilization|war|revolution|century/i.test(topic)) {
    fallbackLinks = [
      { title: "Khan Academy History", url: "https://www.khanacademy.org/humanities/world-history" },
      { title: "History.com", url: "https://www.history.com/" }
    ];
  } else if (/english|literature|grammar|writing|essay/i.test(topic)) {
    fallbackLinks = [
      { title: "Purdue OWL", url: "https://owl.purdue.edu/owl/purdue_owl.html" },
      { title: "Grammarly Blog", url: "https://www.grammarly.com/blog/" }
    ];
  } else if (/computer|algorithm|code|program|software/i.test(topic)) {
    fallbackLinks = [
      { title: "W3Schools", url: "https://www.w3schools.com/" },
      { title: "GeeksforGeeks", url: "https://www.geeksforgeeks.org/" }
    ];
  }
  
  return fallbackLinks;
}

// Function to evaluate answers using Mistral API
async function evaluateWithMistral(data, contextType) {
  try {
    const results = [];
    const studentAnswers = data.student;
    const teacherAnswers = data.teacher;
    
    console.log(`Starting Mistral evaluation with context type: ${contextType}`);
    
    // Process each student answer
    for (const studentItem of studentAnswers) {
      const questionNumber = studentItem.questionNumber;
      const studentAnswer = studentItem.answer || "";
      
      // Find matching teacher answer
      let teacherAnswer = "";
      for (const teacherItem of teacherAnswers) {
        if (teacherItem.questionNumber === questionNumber) {
          teacherAnswer = teacherItem.answer || "";
          break;
        }
      }
      
      if (!studentAnswer || !teacherAnswer) {
        // Skip if either answer is missing
        console.log(`Question ${questionNumber}: Missing student or teacher answer`);
        results.push({
          questionNumber,
          studentAnswer,
          teacherAnswer,
          similarityScore: 0
        });
        continue;
      }
      
      // Get the appropriate prompt based on context type
      const prompt = getMistralPrompt(contextType, studentAnswer, teacherAnswer, questionNumber);
      
      // Log the prompt for debugging
      console.log(`\n----------- PROMPT FOR QUESTION ${questionNumber} -----------`);
      console.log(prompt);
      console.log(`----------- END PROMPT -----------\n`);
      
      // Call Mistral API
      console.log(`Calling Mistral API for question ${questionNumber}...`);
      const response = await axios.post(
        MISTRAL_API_URL,
        {
          model: "mistral-large-latest",
          messages: [
            { role: "system", content: "You are an expert educational evaluator specialized in analyzing and comparing answers." },
            { role: "user", content: prompt }
          ],
          temperature: 0.7,
          max_tokens: 1024
        },
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${MISTRAL_API_KEY}`
          }
        }
      );
      
      // Extract similarity score and feedback from response
      const assistantResponse = response.data.choices[0].message.content;
      
      // Log the response for debugging
      console.log(`\n----------- MISTRAL RESPONSE FOR QUESTION ${questionNumber} -----------`);
      console.log(assistantResponse);
      console.log(`----------- END RESPONSE -----------\n`);
      
      const { similarityScore, feedback } = extractMistralFeedback(assistantResponse);
      console.log(`Extracted similarity score: ${similarityScore}`);
      
      // Add to results
      results.push({
        questionNumber,
        studentAnswer,
        teacherAnswer,
        similarityScore,
        feedback
      });
    }
    
    return results;
  } catch (error) {
    console.error('Error in Mistral evaluation:', error);
    throw error;
  }
}

// Function to extract similarity score and feedback from Mistral response
function extractMistralFeedback(response) {
  try {
    // Extract similarity score with improved regex pattern
    // This will match formats like "SIMILARITY_SCORE: 75" or "SIMILARITY_SCORE: 75.5" or even just numbers
    const scoreMatch = response.match(/SIMILARITY_SCORE:\s*(\d+(?:\.\d+)?)/i) || 
                      response.match(/similarity score:?\s*(\d+(?:\.\d+)?)/i) ||
                      response.match(/score:?\s*(\d+(?:\.\d+)?)/i);
    
    // Log the regex match for debugging
    console.log('Score regex match:', scoreMatch);
    
    let similarityScore = 0;
    if (scoreMatch && scoreMatch[1]) {
      similarityScore = parseFloat(scoreMatch[1]);
      // Ensure score is within 0-100 range
      similarityScore = Math.max(0, Math.min(100, similarityScore));
    } else {
      // If no score pattern found, scan for any number that might be a percentage
      const numberMatches = response.match(/\b(\d{1,3}(?:\.\d+)?)\s*(?:\/100|percent|%)/g);
      if (numberMatches && numberMatches.length > 0) {
        // Extract the first number that looks like a score
        const firstScore = numberMatches[0].match(/(\d+(?:\.\d+)?)/);
        if (firstScore && firstScore[1]) {
          similarityScore = parseFloat(firstScore[1]);
          similarityScore = Math.max(0, Math.min(100, similarityScore));
        }
      }
    }
    
    console.log(`Final extracted similarity score: ${similarityScore}`);
    
    // Extract evaluation feedback
    const evaluationMatch = response.match(/EVALUATION:([\s\S]+)/i) || 
                           response.match(/evaluation:([\s\S]+)/i);
    
    let feedback = "";
    if (evaluationMatch && evaluationMatch[1]) {
      feedback = evaluationMatch[1].trim();
    } else {
      // If no EVALUATION label found, use everything after the score
      if (scoreMatch) {
        const scoreIndex = response.indexOf(scoreMatch[0]) + scoreMatch[0].length;
        feedback = response.substring(scoreIndex).trim();
      } else {
        // If no structure found, use the whole response as feedback
        feedback = response.trim();
      }
    }
    
    return {
      similarityScore,
      feedback
    };
  } catch (error) {
    console.error('Error extracting Mistral feedback:', error);
    return {
      similarityScore: 0,
      feedback: "Error extracting feedback from AI response."
    };
  }
}

// Function to get the appropriate prompt for Mistral based on context type
function getMistralPrompt(contextType, studentAnswer, teacherAnswer, questionNumber) {
  switch (contextType) {
    case 'case-study':
      return `
You are evaluating a case study answer for Question ${questionNumber}.

Teacher's Answer (Marking Scheme):
${teacherAnswer}

Student's Answer:
${studentAnswer}

Please analyze the student's answer against the teacher's marking scheme. Don't simply compare for similarity, but evaluate how well the student has addressed the key points in the case study.

Provide a detailed evaluation with the following:
1. A similarity score from 0-100 based on how well the student addressed the key points
2. Key strengths in the student's analysis
3. Areas where the student missed important points
4. Suggestions for improvement

Format your response as follows:
SIMILARITY_SCORE: [score]
EVALUATION: [your detailed evaluation]
`;

    case 'character-perspective':
      return `
You are evaluating a character perspective/descriptive answer for Question ${questionNumber}.

Teacher's Answer (Reference):
${teacherAnswer}

Student's Answer:
${studentAnswer}

Please analyze the student's descriptive answer against the teacher's reference. Consider creativity, understanding of the character/topic, language use, and how well they've addressed the key elements.

Provide a detailed evaluation with the following:
1. A similarity score from 0-100 based on overall quality and alignment with key elements
2. Strengths in the student's writing
3. Areas for improvement
4. Specific suggestions to enhance the answer

Format your response as follows:
SIMILARITY_SCORE: [score]
EVALUATION: [your detailed evaluation]
`;

    case 'letter-writing':
      return `
You are an expert evaluator of letter writing assignments in educational settings.

Teacher's Reference Letter (Format & Content Guide):
${teacherAnswer}

Student's Letter:
${studentAnswer}

EVALUATION INSTRUCTIONS:
Evaluate the student's letter against the teacher's reference letter using the following comprehensive marking scheme (total 100 points):

1. INTRODUCTION (10 points):
   - Clear purpose statement and context setting
   - Appropriate opening that engages the reader
   - Sets the tone for the rest of the letter

2. CONTENT & ARGUMENTS (40 points):
   - Main purpose/subject of the letter clearly addressed
   - All key points from teacher's reference included
   - Supporting details and examples where appropriate
   - Persuasive or informative elements as required by the letter type
   - Logical development of ideas
   - Depth and quality of content

3. STRUCTURE & ORGANIZATION (20 points):
   - Proper letter format (sender's address, date, recipient's address)
   - Appropriate salutation and complimentary close
   - Clear paragraphing with one main idea per paragraph
   - Logical sequence and flow between paragraphs
   - Coherent overall structure
   - Signature and name

4. LANGUAGE & GRAMMAR (15 points):
   - Appropriate tone and register for letter type
   - Grammar and spelling accuracy
   - Vocabulary and expression
   - Sentence structure and variety
   - Conciseness and clarity

5. CONCLUSION (15 points):
   - Effective summary of main points
   - Clear call to action or next steps (if appropriate)
   - Proper closing statement
   - Leaves reader with clear understanding of writer's purpose
   - Appropriate final impression

Calculate a final similarity score from 0-100 based on the total points earned across all categories.

IMPORTANT: Even if the student's letter differs in specific wording but captures the essence and purpose of the reference letter, it should receive a high score. Focus on whether the student's letter would achieve the same communicative purpose as the reference letter.

Format your response as follows:
SIMILARITY_SCORE: [numerical score between 0-100]
EVALUATION: 
[Provide a detailed breakdown of points earned in each category]
[Include specific strengths of the student's letter]
[Mention areas for improvement]
[Give constructive suggestions]
`;
    default:
      return `
You are evaluating a student's answer for Question ${questionNumber}.

Teacher's Answer (Reference):
${teacherAnswer}

Student's Answer:
${studentAnswer}

Please analyze the student's answer against the teacher's reference answer. Consider accuracy, completeness, and understanding of the topic.

Provide a detailed evaluation with the following:
1. A similarity score from 0-100
2. Key strengths in the student's answer
3. Areas for improvement
4. Specific suggestions to enhance the answer

Format your response as follows:
SIMILARITY_SCORE: [score]
EVALUATION: [your detailed evaluation]
`;
  }
}

// Serve static files like CSS, JS, etc.
app.use(express.static(path.join(__dirname, 'public')));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`Using Gemini model: gemini-1.5-pro`);
  console.log(`API Key: ${process.env.GEMINI_API_KEY ? 'Configured' : 'Missing'}`);
}).on('error', (err) => {
  console.error('Server failed to start:', err.message);
  if (err.code === 'EADDRINUSE') {
    console.error(`Port ${PORT} is already in use. Try closing other applications or using a different port.`);
  }
});
