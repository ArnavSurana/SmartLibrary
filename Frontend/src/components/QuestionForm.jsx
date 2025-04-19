import React, { useState } from 'react';
import { 
  TextField, 
  Button, 
  Box, 
  Typography, 
  Paper, 
  CircularProgress,
  Divider
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import apiService from '../services/apiService';
import authService from '../appwrite/appwrite';

const QuestionForm = ({ bookName, userId, onAnswerReceived }) => {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    
    setLoading(true);
    setError('');
    
    try {
      // Get answer from AI backend
      const responseAnswer = await apiService.askQuestion(userId, bookName, question);
      setAnswer(responseAnswer);
      
      // Save Q&A pair to Appwrite
      console.log('Saving Q&A pair:', { bookName, question, responseAnswer });
      const saveResponse = await authService.saveQAPair(bookName, question, responseAnswer);
      console.log('Save Q&A pair response:', saveResponse);
      
      if (onAnswerReceived) {
        onAnswerReceived({
          question,
          answer: responseAnswer,
          timestamp: new Date().toISOString()
        });
      }
      } catch (err) {
      // Improved user-friendly error message for no relevant content
      if (err.response && err.response.status === 404) {
        setError('No relevant content found for your question. Please try a different query or upload the book.');
      } else {
        setError(`Failed to get answer: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Ask a Question
      </Typography>
      <Box component="form" onSubmit={handleSubmit}>
        <TextField
          fullWidth
          label="Enter your question"
          variant="outlined"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
          sx={{ mb: 2 }}
        />
        <Button
          type="submit"
          variant="contained"
          endIcon={<SendIcon />}
          disabled={!question.trim() || loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Ask'}
        </Button>
      </Box>

      {error && (
        <Typography color="error" sx={{ mt: 2 }}>
          {error}
        </Typography>
      )}

      {answer && (
        <>
          <Divider sx={{ my: 3 }} />
          <Box>
            <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
              Question:
            </Typography>
            <Typography variant="body1" sx={{ mb: 2 }}>
              {question}
            </Typography>
            
            <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
              Answer:
            </Typography>
            <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
              {answer}
            </Typography>
          </Box>
        </>
      )}
    </Paper>
  );
};

export default QuestionForm;