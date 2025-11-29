// src/hooks/useAnalytics.ts
import { useQuery } from '@tanstack/react-query';
import httpClient from '@services/HttpClient';

interface AnalyticsData {
  totalPapers: number;
  totalAuthors: number;
  averageCitations: number;
  categoriesCount: Record<string, number>;
}

export const useAnalytics = () => {
  return useQuery({
    queryKey: ['analytics'],
    queryFn: async () => {
      const response = await httpClient.get<AnalyticsData>('/analytics');
      return response.data;
    },
  });
};

export default useAnalytics;