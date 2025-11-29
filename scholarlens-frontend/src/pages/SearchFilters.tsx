// src/components/Search/SearchFilters.tsx
import React from 'react';
import { Card, Button, Space } from 'antd';

interface SearchFiltersProps {
  onApply: (filters: any) => void;
}

export const SearchFilters: React.FC<SearchFiltersProps> = ({ onApply }) => {
  return (
    <Card title="Filters">
      <Space direction="vertical" style={{ width: '100%' }}>
        <div>
          <p><strong>Category</strong></p>
          {/* Add filter options here */}
        </div>
        <Button type="primary" block onClick={() => onApply({})}>
          Apply Filters
        </Button>
      </Space>
    </Card>
  );
};

export default SearchFilters;