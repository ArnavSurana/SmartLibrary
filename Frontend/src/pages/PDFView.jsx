import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  Paper, 
  Grid, 
  Button,
  CircularProgress,
  Alert,
  Divider
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import Layout from '../components/Layout';
import PDFViewer from '../components/PDFViewer.jsx';
import QuestionForm from '../components/QuestionForm';
import QAHistoryList from '../components/QAHistoryList';
import authService from '../appwrite/appwrite';
import { useSelector } from 'react-redux';

const PDFView = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [pdf, setPdf] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [qaHistory, setQaHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('qaHistory_' + id);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [qaLoading, setQaLoading] = useState(true);
  const [qaError, setQaError] = useState('');
  const userData = useSelector((state) => state.auth.userData);
  console.log('Current userData.$id:', userData ? userData.$id : 'No user data');

  useEffect(() => {
    const fetchPDFDetails = async () => {
      try {
        // Fetch the specific PDF document
        const userPDFs = await authService.getUserPDFs(userData.$id);
        const pdfDoc = userPDFs.find(p => p.$id === id);
        
        if (pdfDoc) {
          setPdf(pdfDoc);
          
        // Fetch QA history for this PDF
        setQaLoading(true);
        setQaError('');
        try {
          const questions = await authService.getPDFQuestions(pdfDoc.name, userData.$id);
          console.log('Fetched QA history:', questions);
          setQaHistory(questions);
          try {
            localStorage.setItem('qaHistory_' + id, JSON.stringify(questions));
          } catch {}
        } catch (err) {
          console.error('Error fetching QA history:', err);
          setQaError('Failed to load chat history.');
        } finally {
          setQaLoading(false);
        }
      } else {
        setError('PDF not found or you do not have access to it.');
        navigate('/dashboard');
      }
    } catch (err) {
      console.error('Error fetching PDF details:', err);
      setError('Failed to load document details.');
    } finally {
      setLoading(false);
    }
  };

    if (userData) {
      fetchPDFDetails();
    }
  }, [id, userData, navigate]);

  const handleNewQA = (qaItem) => {
    setQaHistory(prev => {
      const newHistory = [qaItem, ...prev];
      try {
        localStorage.setItem('qaHistory_' + id, JSON.stringify(newHistory));
      } catch {}
      return newHistory;
    });
  };

  if (loading) {
    return (
      <Layout>
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout>
        <Alert severity="error">{error}</Alert>
        <Button 
          sx={{ mt: 2 }}
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate('/dashboard')}
        >
          Back to Dashboard
        </Button>
      </Layout>
    );
  }

  return (
    <Layout>
      <Button 
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate('/dashboard')}
        sx={{ mb: 3 }}
      >
        Back to Dashboard
      </Button>

      <Typography variant="h4" component="h1" gutterBottom>
        {pdf.name}
      </Typography>
      
      <Grid container spacing={3}>
        {/* PDF Viewer */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Document Preview
            </Typography>
            <PDFViewer fileId={pdf.fileId} fileName={pdf.name} />
          </Paper>
        </Grid>
        
        {/* Question Form and History */}
      <Grid item xs={12} md={6}>
        <QuestionForm 
          bookName={pdf.name} 
          userId={userData.$id}
          onAnswerReceived={handleNewQA}
        />
        
        <Divider sx={{ my: 3 }} />
        
        {qaLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
            <CircularProgress size={24} />
          </Box>
        ) : qaError ? (
          <Alert severity="error">{qaError}</Alert>
        ) : (
          <QAHistoryList qaList={qaHistory} />
        )}
      </Grid>
      </Grid>
    </Layout>
  );
};

export default PDFView;