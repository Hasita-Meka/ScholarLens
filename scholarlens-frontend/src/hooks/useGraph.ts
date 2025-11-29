import { useEffect, useState } from 'react';
import ApiService from '../services/api';

export const useGraph = () => {
  const [graphData, setGraphData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchGraph = async () => {
      try {
        setLoading(true);
        const response = await ApiService.getCollaborationGraph();
        setGraphData(response.data);
        setError(null);
      } catch (err: any) {
        setError(err.message || 'Failed to load graph');
      } finally {
        setLoading(false);
      }
    };

    fetchGraph();
  }, []);

  return { graphData, loading, error };
};
