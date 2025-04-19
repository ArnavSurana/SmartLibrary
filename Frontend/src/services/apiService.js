import axios from 'axios';

class APIService {
  constructor() {
    // ← relative baseURL so that '/books/' and '/questions/' go through Vite’s proxy
    this.client = axios.create({
      baseURL:  '/',  
      headers:  { 'Content-Type': 'application/json' },
    });
  }

  // 1) Upload a PDF
  async uploadPDF(file, userId) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', userId);

    try {
      // ← use the trailing slash to hit your FastAPI route exactly
      const response = await this.client.post('/books/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    } catch (error) {
      // if you want to see FastAPI’s error detail, you can console.log(error.response.data)
      console.error('Error uploading PDF to backend:', error.response?.data || error);
      throw error;
    }
  }

  // 2) List books
  async listBooks(userId) {
    try {
      const response = await this.client.get(`/books/${userId}/`);
      return response.data.books;
    } catch (error) {
      console.error('Error listing books:', error.response?.data || error);
      throw error;
    }
  }

  // 3) Ask a question
  async askQuestion(userId, bookName, query) {
    try {
      const response = await this.client.post('/questions/', {
        user_id:   userId,
        book_name: bookName,
        query:     query,
      });
      console.log('Response from backend:', response.data);
      return response.data.answer;
    } catch (error) {
      console.error('Error asking question:', error.response?.data || error);
      throw error;
    }
  }
}

export default new APIService();
