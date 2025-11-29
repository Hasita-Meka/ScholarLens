// src/utils/constants.ts
export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
export const API_TIMEOUT = parseInt(process.env.REACT_APP_API_TIMEOUT || '30000', 10);

export const PAPER_CATEGORIES = {
  cs: 'Computer Science',
  bio: 'Biology',
  phys: 'Physics',
  math: 'Mathematics',
  chem: 'Chemistry',
};

export const SORT_OPTIONS = [
  { label: 'Relevance', value: 'relevance' },
  { label: 'Date (Newest)', value: 'date_desc' },
  { label: 'Date (Oldest)', value: 'date_asc' },
  { label: 'Citations (High)', value: 'citations_desc' },
];

export const PAGINATION = {
  PAGE_SIZE: 20,
  MAX_RESULTS: 1000,
};

export default {
  API_BASE_URL,
  API_TIMEOUT,
  PAPER_CATEGORIES,
  SORT_OPTIONS,
  PAGINATION,
};