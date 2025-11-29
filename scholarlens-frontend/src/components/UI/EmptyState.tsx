// src/components/UI/EmptyState.tsx
import React from 'react';
import { Empty, Button } from 'antd';

interface EmptyStateProps {
  title?: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No Data',
  description = 'There is no data to display',
  action,
}) => {
  return (
    <div style={{ textAlign: 'center', padding: '50px 20px' }}>
      <Empty
        description={
          <div>
            <h2>{title}</h2>
            <p>{description}</p>
          </div>
        }
      />
      {action && (
        <Button type="primary" onClick={action.onClick} style={{ marginTop: '20px' }}>
          {action.label}
        </Button>
      )}
    </div>
  );
};

export default EmptyState;