import { useState } from 'react';
import ApiService from '../services/api';

export const useSearch = () => {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = async (query: string, type: 'semantic' | 'keyword' = 'semantic') => {
    try {
      setLoading(true);
      const response = type === 'semantic'
        ? await ApiService.semanticSearch(query)
        : await ApiService.keywordSearch(query);
      setResults(response.data);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Search failed');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return { results, loading, error, search };
};
