/**
 * Smart Library utility helper functions
 */

/**
 * Format a date to a readable string
 * @param {string|Date} date - Date to format
 * @param {boolean} includeTime - Whether to include time
 * @returns {string} Formatted date string
 */
export const formatDate = (date, includeTime = false) => {
    if (!date) return '';
    
    const dateObj = new Date(date);
    
    // Check if date is valid
    if (isNaN(dateObj.getTime())) {
      return '';
    }
    
    const options = {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      ...(includeTime && {
        hour: '2-digit',
        minute: '2-digit'
      })
    };
    
    return dateObj.toLocaleDateString(undefined, options);
  };
  
  /**
   * Format file size to human-readable string
   * @param {number} bytes - Size in bytes
   * @param {number} decimals - Number of decimal places
   * @returns {string} Formatted size with unit
   */
  export const formatFileSize = (bytes, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
  
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + sizes[i];
  };
  
  /**
   * Validate if a file is a valid PDF
   * @param {File} file - File to validate
   * @returns {boolean} Whether the file is a valid PDF
   */
  export const isValidPDF = (file) => {
    // Check file type
    if (file && file.type === 'application/pdf') {
      // Check file size (max 20MB)
      const maxSize = 20 * 1024 * 1024; // 20MB in bytes
      return file.size <= maxSize;
    }
    return false;
  };
  
  /**
   * Truncate text to a specific length with ellipsis
   * @param {string} text - Text to truncate
   * @param {number} maxLength - Maximum length
   * @returns {string} Truncated text
   */
  export const truncateText = (text, maxLength = 100) => {
    if (!text) return '';
    
    if (text.length <= maxLength) {
      return text;
    }
    
    return text.substring(0, maxLength) + '...';
  };
  
  /**
   * Debounce a function call
   * @param {Function} func - Function to debounce
   * @param {number} wait - Wait time in milliseconds
   * @returns {Function} Debounced function
   */
  export const debounce = (func, wait = 300) => {
    let timeout;
    
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  };
  
  /**
   * Extract text from a PDF file name (remove extension and path)
   * @param {string} filename - PDF file name
   * @returns {string} Cleaned file name
   */
  export const cleanPDFName = (filename) => {
    if (!filename) return '';
    
    // Remove path if present
    let name = filename.split('/').pop().split('\\').pop();
    
    // Remove extension
    name = name.replace(/\.pdf$/i, '');
    
    // Replace underscores and hyphens with spaces
    name = name.replace(/[_-]/g, ' ');
    
    return name;
  };
  
  /**
   * Generate a random string ID
   * @param {number} length - Length of the ID
   * @returns {string} Random ID
   */
  export const generateId = (length = 8) => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    
    return result;
  };
  
  /**
   * Check if a string contains a keyword
   * @param {string} text - Text to search in
   * @param {string} keyword - Keyword to search for
   * @returns {boolean} Whether the text contains the keyword
   */
  export const containsKeyword = (text, keyword) => {
    if (!text || !keyword) return false;
    
    return text.toLowerCase().includes(keyword.toLowerCase());
  };
  
  /**
   * Group QA items by date
   * @param {Array} qaItems - Array of QA items
   * @returns {Object} Grouped QA items by date
   */
  export const groupQAByDate = (qaItems) => {
    if (!qaItems || !qaItems.length) return {};
    
    return qaItems.reduce((groups, item) => {
      const date = new Date(item.timestamp).toLocaleDateString();
      
      if (!groups[date]) {
        groups[date] = [];
      }
      
      groups[date].push(item);
      return groups;
    }, {});
  };
  
  /**
   * Create a downloadable link for text content
   * @param {string} content - Content to download
   * @param {string} filename - File name
   * @param {string} contentType - Content type
   */
  export const downloadTextAsFile = (content, filename, contentType = 'text/plain') => {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    
    URL.revokeObjectURL(url);
  };
  
  /**
   * Extract meaningful keywords from text
   * @param {string} text - Text to extract keywords from
   * @param {number} maxCount - Maximum number of keywords
   * @returns {Array} Array of keywords
   */
  export const extractKeywords = (text, maxCount = 5) => {
    if (!text) return [];
    
    // Common English stop words to filter out
    const stopWords = new Set([
      'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with',
      'by', 'about', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
      'has', 'had', 'do', 'does', 'did', 'of', 'this', 'that', 'these', 'those'
    ]);
    
    // Clean and split text into words
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, '') // Remove punctuation
      .split(/\s+/) // Split by whitespace
      .filter(word => word.length > 2 && !stopWords.has(word)); // Filter out stop words and short words
    
    // Count word frequency
    const wordCounts = words.reduce((counts, word) => {
      counts[word] = (counts[word] || 0) + 1;
      return counts;
    }, {});
    
    // Sort by frequency and get top N
    const sortedWords = Object.entries(wordCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, maxCount)
      .map(entry => entry[0]);
    
    return sortedWords;
  };