import React, { useState } from 'react';
import { Box, Typography, Button, CircularProgress, Alert } from '@mui/material';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import authService from '../appwrite/appwrite';

// Import worker file as a URL asset so Vite bundles it correctly
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

const PDFViewer = ({ fileId, fileName }) => {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [scale, setScale] = useState(0.8); // Changed default scale to 0.9
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const pdfUrl = authService.getPDFViewURL(fileId);

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
    setLoading(false);
  };

  const changePage = (offset) => {
    setPageNumber(prev => {
      const next = prev + offset;
      return Math.max(1, Math.min(next, numPages));
    });
  };
  const previousPage = () => changePage(-1);
  const nextPage = () => changePage(1);
  const zoomIn = () => setScale(s => Math.min(s + 0.2, 2.5));
  const zoomOut = () => setScale(s => Math.max(s - 0.2, 0.5));

  const onLoadError = (err) => {
    console.error('Error loading PDF:', err);
    setError('Error loading PDF. Please try again later.');
    setLoading(false);
  };

  return (
    <Box sx={{ width: '100%' }}>
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      <Box sx={{
        border: '1px solid #e0e0e0',
        borderRadius: 1,
        overflow: 'auto',
        height: '600px',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'flex-start',
        p: 2
      }}>
        <Document
          file={pdfUrl}
          onLoadSuccess={onDocumentLoadSuccess}
          onLoadError={onLoadError}
          loading={<CircularProgress />}
        >
          <Box sx={{ maxWidth: '800px', width: '100%', display: 'flex', justifyContent: 'center' }}>
            <Page
              pageNumber={pageNumber}
              scale={scale}
              renderTextLayer={true}
              renderAnnotationLayer={true}
            />
          </Box>
        </Document>
      </Box>

      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2, gap: 1 }}>
        <Button onClick={zoomOut} disabled={scale <= 0.5} startIcon={<ZoomOutIcon />}>Zoom Out</Button>
        <Button onClick={previousPage} disabled={pageNumber <= 1} startIcon={<NavigateBeforeIcon />}>Prev</Button>
        <Typography sx={{ display: 'flex', alignItems: 'center', mx: 1 }}>
          {pageNumber} / {numPages || '?'}
        </Typography>
        <Button onClick={nextPage} disabled={pageNumber >= numPages} endIcon={<NavigateNextIcon />}>Next</Button>
        <Button onClick={zoomIn} disabled={scale >= 2.5} startIcon={<ZoomInIcon />}>Zoom In</Button>
      </Box>
    </Box>
  );
};

export default PDFViewer;
