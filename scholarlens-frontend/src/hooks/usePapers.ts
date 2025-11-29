import { useEffect, useState } from 'react';
import ApiService from '../services/api';

interface Paper {
  paper_id: number;
  title: string;
  authors: string[];
  abstract: string;
  published_date: string;
  arxiv_id: string;
  categories: string[];
}

export const usePapers = (page: number = 1, limit: number = 20) => {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    const fetchPapers = async () => {
      try {
        setLoading(true);
        const response = await ApiService.getPapers(page, limit);
        setPapers(response.data.papers);
        setTotal(response.data.total);
        setError(null);
      } catch (err: any) {
        setError(err.message || 'Failed to fetch papers');
      } finally {
        setLoading(false);
      }
    };

    fetchPapers();
  }, [page, limit]);

  return { papers, loading, error, total };
};
