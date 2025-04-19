import React from 'react';
import { 
  Typography, 
  Box, 
  Paper, 
  List, 
  ListItem, 
  ListItemText, 
  Accordion, 
  AccordionSummary, 
  AccordionDetails,
  Divider
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import QuestionAnswerIcon from '@mui/icons-material/QuestionAnswer';

const QAHistoryList = ({ qaList }) => {
  // Sort by timestamp (newest first)
  const sortedQAList = [...qaList].sort((a, b) => 
    new Date(b.timestamp) - new Date(a.timestamp)
  );

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <QuestionAnswerIcon sx={{ mr: 1 }} />
        <Typography variant="h6">
          Question History
        </Typography>
      </Box>

      {sortedQAList.length === 0 ? (
        <Typography variant="body2" color="text.secondary">
          No questions have been asked yet. Try asking a question about the document.
        </Typography>
      ) : (
        <List sx={{ width: '100%' }}>
          {sortedQAList.map((qa, index) => (
            <React.Fragment key={qa.$id || index}>
              <ListItem sx={{ px: 0, py: 1 }} alignItems="flex-start">
                <Accordion sx={{ width: '100%' }}>
                  <AccordionSummary
                    expandIcon={<ExpandMoreIcon />}
                    aria-controls={`panel-${index}-content`}
                    id={`panel-${index}-header`}
                  >
                    <ListItemText
                      primary={qa.question}
                      secondary={new Date(qa.timestamp).toLocaleString()}
                      primaryTypographyProps={{
                        noWrap: true,
                        fontWeight: 500,
                        sx: { pr: 5 }
                      }}
                    />
                  </AccordionSummary>
                  <AccordionDetails>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Question:
                    </Typography>
                    <Typography variant="body2" paragraph>
                      {qa.question}
                    </Typography>
                    
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Answer:
                    </Typography>
                    <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                      {qa.answer}
                    </Typography>
                  </AccordionDetails>
                </Accordion>
              </ListItem>
              {index < sortedQAList.length - 1 && <Divider />}
            </React.Fragment>
          ))}
        </List>
      )}
    </Paper>
  );
};

export default QAHistoryList;