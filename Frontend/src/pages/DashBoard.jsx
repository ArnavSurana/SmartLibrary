import React, { useState, useEffect } from 'react';
import { 
  Typography, 
  Grid, 
  Card, 
  CardContent, 
  CardActions, 
  Button, 
  Box,
  CardMedia,
  CircularProgress,
  Alert,
  Divider
} from '@mui/material';
import { Link } from 'react-router-dom';
import VisibilityIcon from '@mui/icons-material/Visibility';
import DeleteIcon from '@mui/icons-material/Delete';
import Layout from '../components/Layout';
import PDFUploader from '../components/PDFUploader';
import authService from '../appwrite/appwrite';
import { useSelector } from 'react-redux';

const Dashboard = () => {
  const [pdfs, setPdfs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const userData = useSelector((state) => state.auth.userData);

  const fetchPDFs = async () => {
    if (!userData) return;
    
    setLoading(true);
    try {
      const userPDFs = await authService.getUserPDFs(userData.$id);
      setPdfs(userPDFs);
    } catch (err) {
      console.error('Error fetching PDFs:', err);
      setError('Failed to load your documents. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPDFs();
  }, [userData]);

  const handleDelete = async (fileId, documentId, bookName) => {
    try {
      const userId = userData ? userData.$id : null;
      if (!userId) {
        setError('User not authenticated.');
        return;
      }
      await authService.deletePDF(fileId, documentId, userId, bookName);
      setPdfs(pdfs.filter(pdf => pdf.$id !== documentId));
    } catch (err) {
      console.error('Error deleting PDF:', err);
      setError('Failed to delete the document. Please try again.');
    }
  };

  return (
    <Layout>
      <Typography variant="h4" component="h1" gutterBottom>
        Your Library
      </Typography>
      
      <PDFUploader onUploadComplete={fetchPDFs} />
      
      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}
      
      <Divider sx={{ my: 3 }} />
      
      <Typography variant="h5" component="h2" gutterBottom>
        Your Documents
      </Typography>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      ) : pdfs.length === 0 ? (
        <Alert severity="info" sx={{ my: 2 }}>
          You haven't uploaded any documents yet. Use the uploader above to add your first PDF.
        </Alert>
      ) : (
        <Grid container spacing={3} sx={{ mt: 1 }} columns={{ xs: 12, sm: 12, md: 12 }}>
          {pdfs.map((pdf) => (
            <Grid key={pdf.$id} item xs={12} sm={6} md={4}>
              <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <CardMedia
                  component="img"
                  height="140"
                  image={`https://ui-avatars.com/api/?name=${encodeURIComponent(pdf.name)}&background=random&color=fff&bold=true`}
                  alt={pdf.name}
                  sx={{ objectFit: 'contain', bgcolor: 'grey.100', p: 1 }}
                />
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h6" component="h3" noWrap title={pdf.name}>
                    {pdf.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {new Date(pdf.uploadDate).toLocaleDateString()}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {(pdf.size / 1024 / 1024).toFixed(2)} MB
                  </Typography>
                </CardContent>
                <CardActions>
                  <Button 
                    size="small" 
                    component={Link} 
                    to={`/pdf/${pdf.$id}`}
                    startIcon={<VisibilityIcon />}
                  >
                    View & Ask
                  </Button>
                  <Button 
                    size="small" 
                    color="error" 
                    startIcon={<DeleteIcon />}
                    onClick={() => handleDelete(pdf.fileId, pdf.$id, pdf.name)}
                  >
                    Delete
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Layout>
  );
};

export default Dashboard;
