import React from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  Container, 
  Paper 
} from '@mui/material';
import Grid from '@mui/material/Grid';
import { Link } from 'react-router-dom';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import LiveHelpIcon from '@mui/icons-material/LiveHelp';
import { useSelector } from 'react-redux';

const FeatureCard = ({ icon, title, description }) => (
  <Paper elevation={3} sx={{ p: 3, display: 'flex', flexDirection: 'column' }}>
    <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
      {icon}
    </Box>
    <Typography variant="h6" component="h3" gutterBottom align="center">
      {title}
    </Typography>
    <Typography variant="body2" color="text.secondary" align="center">
      {description}
    </Typography>
  </Paper>
);

const Home = () => {
  const isLoggedIn = useSelector((state) => state.auth.status);

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Hero Section */}
      <Box 
        sx={{ 
          bgcolor: 'primary.main', 
          color: 'white', 
          py: 8,
          textAlign: 'center'
        }}
      >
        <Container maxWidth="md">
          <Typography variant="h2" component="h1" gutterBottom>
            Smart Library
          </Typography>
          <Typography variant="h5" component="h2" gutterBottom>
            Your personal AI-powered PDF library
          </Typography>
          <Typography variant="body1" paragraph sx={{ mb: 4 }}>
            Upload your PDFs and ask questions to get intelligent answers based on the content.
          </Typography>
          <Box sx={{ mt: 4 }}>
            {isLoggedIn ? (
              <Button 
                component={Link} 
                to="/dashboard" 
                variant="contained" 
                color="secondary" 
                size="large"
                sx={{ px: 4, py: 1.5 }}
              >
                Go to Dashboard
              </Button>
            ) : (
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
                <Button 
                  component={Link} 
                  to="/login" 
                  variant="contained" 
                  color="secondary" 
                  size="large"
                >
                  Log In
                </Button>
                <Button 
                  component={Link} 
                  to="/register" 
                  variant="outlined" 
                  color="secondary" 
                  size="large"
                >
                  Register
                </Button>
              </Box>
            )}
          </Box>
        </Container>
      </Box>

      {/* Features Section */}
      <Container maxWidth="lg" sx={{ my: 8 }}>
        <Typography variant="h4" component="h2" align="center" gutterBottom sx={{ mb: 6 }}>
          Features
        </Typography>
        <Grid container spacing={4}>
        <Grid item xs={12} md={4}>
            <FeatureCard 
              icon={<CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main' }} />}
              title="Upload PDFs"
              description="Easily upload and manage your PDF documents in a secure cloud storage."
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <FeatureCard 
              icon={<MenuBookIcon sx={{ fontSize: 60, color: 'primary.main' }} />}
              title="Intelligent Indexing"
              description="Our AI system analyzes and indexes your documents for quick and accurate retrieval."
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <FeatureCard 
              icon={<MenuBookIcon sx={{ fontSize: 60, color: 'primary.main' }} />}
              title="Read Intelligently"
              description="Easily upload and manage your PDF documents in a secure cloud storage"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <FeatureCard 
              icon={<LiveHelpIcon sx={{ fontSize: 60, color: 'primary.main' }} />}
              title="Ask Questions"
              description="Our AI system analyzes and indexes your documents for quick and accurate retrieval.."
            />
          </Grid>
        </Grid>
      </Container>

      {/* CTA Section */}
      <Box sx={{ bgcolor: 'grey.100', py: 8 }}>
        <Container maxWidth="md" sx={{ textAlign: 'center' }}>
          <Typography variant="h4" component="h2" gutterBottom>
            Ready to get started?
          </Typography>
          <Typography variant="body1" paragraph sx={{ mb: 4 }}>
            Join Smart Library today and transform the way you interact with your documents.
          </Typography>
          {!isLoggedIn && (
            <Button 
              component={Link} 
              to="/register" 
              variant="contained" 
              color="primary" 
              size="large"
              sx={{ px: 4, py: 1.5 }}
            >
              Create Account
            </Button>
          )}
        </Container>
      </Box>
    </Box>
  );
};

export default Home;
