// src/components/UI/Loading.tsx
import React from 'react';
import { Spin, Card } from 'antd';

interface LoadingProps {
  message?: string;
  fullPage?: boolean;
}

export const Loading: React.FC<LoadingProps> = ({ message = 'Loading...', fullPage = false }) => {
  if (fullPage) {
    return (
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '100vh',
          backgroundColor: '#f5f5f5',
        }}
      >
        <Spin size="large" tip={message} />
      </div>
    );
  }

  return (
    <Card style={{ textAlign: 'center', padding: '50px' }}>
      <Spin size="large" tip={message} />
    </Card>
  );
};

export default Loading;