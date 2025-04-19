import React, { useState } from 'react';
import { 
  Button, 
  Box, 
  Typography, 
  Paper, 
  LinearProgress, 
  Alert,
  IconButton
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CloseIcon from '@mui/icons-material/Close';
import authService from '../appwrite/appwrite';
import apiService from '../services/apiService';
import { useSelector } from 'react-redux';

const PDFUploader = ({ onUploadComplete }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const userData = useSelector((state) => state.auth.userData);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setError('');
    } else {
      setFile(null);
      setError('Please select a valid PDF file');
    }
  };

  const simulateProgress = () => {
    let currentProgress = 0;
    const interval = setInterval(() => {
      currentProgress += Math.random() * 10;
      if (currentProgress > 95) {
        clearInterval(interval);
        currentProgress = 95; // Wait for actual completion
      }
      setProgress(Math.min(currentProgress, 95));
    }, 300);
    return interval;
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setProgress(0);
    setError('');
    setSuccess('');

    const progressInterval = simulateProgress();

    try {
      // 1. Upload to Appwrite Storage
      const uploadResult = await authService.uploadPDF(file);
      
      // 2. Upload to AI backend for indexing
      await apiService.uploadPDF(file, userData.$id, file.name);
      console.log('PDF uploaded to AI backend:', userData.$id, file.name);
      
      setProgress(100);
      setSuccess(`Successfully uploaded ${file.name}`);
      
      // Notify parent component
      if (onUploadComplete) {
        onUploadComplete();
      }
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
    } finally {
      clearInterval(progressInterval);
      setUploading(false);
      // Reset file after upload attempt
      setFile(null);
    }
  };

  const clearFile = () => {
    setFile(null);
  };

  return (
    <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Upload PDF
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
      )}
      
      {success && (
        <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>
      )}

      <Box sx={{ mb: 2 }}>
        <input
          accept="application/pdf"
          style={{ display: 'none' }}
          id="pdf-file-input"
          type="file"
          onChange={handleFileChange}
          disabled={uploading}
        />
        <label htmlFor="pdf-file-input">
          <Button
            variant="outlined"
            component="span"
            startIcon={<CloudUploadIcon />}
            disabled={uploading}
          >
            Select PDF File
          </Button>
        </label>
      </Box>

      {file && (
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Typography variant="body2" sx={{ mr: 1, flexGrow: 1 }}>
            {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
          </Typography>
          <IconButton size="small" onClick={clearFile} disabled={uploading}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      )}

      {uploading && (
        <Box sx={{ width: '100%', mb: 2 }}>
          <LinearProgress variant="determinate" value={progress} />
          <Typography variant="body2" sx={{ mt: 1 }}>
            {Math.round(progress)}% Complete
          </Typography>
        </Box>
      )}

      <Button
        variant="contained"
        color="primary"
        onClick={handleUpload}
        disabled={!file || uploading}
      >
        {uploading ? 'Uploading...' : 'Upload PDF'}
      </Button>
    </Paper>
  );
};

export default PDFUploader;
