// src/utils/formatters.ts
import dayjs from 'dayjs';

export const formatDate = (date: string | Date, format = 'MMM DD, YYYY') => {
  return dayjs(date).format(format);
};

export const formatNumber = (num: number) => {
  return new Intl.NumberFormat().format(num);
};

export const formatCitations = (citations: number) => {
  if (citations >= 1000000) return (citations / 1000000).toFixed(1) + 'M';
  if (citations >= 1000) return (citations / 1000).toFixed(1) + 'K';
  return citations.toString();
};

export const truncateText = (text: string, length = 100) => {
  return text.length > length ? text.substring(0, length) + '...' : text;
};

export default {
  formatDate,
  formatNumber,
  formatCitations,
  truncateText,
};