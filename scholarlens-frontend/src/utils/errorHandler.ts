// src/utils/errorHandler.ts
import { message } from 'antd';

export const handleError = (error: any) => {
  if (error.response) {
    const status = error.response.status;
    const data = error.response.data;
    
    switch (status) {
      case 400:
        message.error(data.message || 'Bad request');
        break;
      case 401:
        message.error('Unauthorized');
        break;
      case 404:
        message.error('Not found');
        break;
      case 500:
        message.error('Server error');
        break;
      default:
        message.error(data.message || 'An error occurred');
    }
  } else if (error.request) {
    message.error('No response from server');
  } else {
    message.error(error.message || 'An error occurred');
  }
};

export const formatErrorMessage = (error: any): string => {
  if (error.response?.data?.message) {
    return error.response.data.message;
  }
  return error.message || 'An error occurred';
};

export default handleError;