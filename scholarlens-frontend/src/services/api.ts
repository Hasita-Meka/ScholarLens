import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient: AxiosInstance = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

// API Methods
export const ApiService = {
  // Papers endpoints
  getPapers: (page: number = 1, limit: number = 20) =>
    apiClient.get(`/papers?page=${page}&limit=${limit}`),
  
  getPaperById: (id: string) =>
    apiClient.get(`/papers/${id}`),
  
  // Search endpoints
  semanticSearch: (query: string, limit: number = 10) =>
    apiClient.post('/search/semantic', { query, limit }),
  
  keywordSearch: (query: string, limit: number = 10) =>
    apiClient.get(`/search/papers?query=${query}&limit=${limit}`),
  
  autoComplete: (query: string) =>
    apiClient.post('/search/autocomplete', { query }),
  
  // Graph endpoints
  getAuthorNetwork: (authorId: string) =>
    apiClient.get(`/graph/authors/${authorId}/network`),
  
  getCollaborationGraph: () =>
    apiClient.get('/graph/collaborations'),
  
  // Analytics endpoints
  getStatistics: () =>
    apiClient.get('/analytics/statistics'),
  
  // Health check
  healthCheck: () =>
    apiClient.get('/health'),
};

export default ApiService;
