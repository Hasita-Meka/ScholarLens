// src/components/Search/SearchFilters.tsx
import React, { useState } from 'react';
import { Card, DatePicker, Select, Button, Space, Collapse } from 'antd';
import dayjs from 'dayjs';

interface FilterOptions {
  dateFrom?: string;
  dateTo?: string;
  category?: string;
  minCitations?: number;
}

interface SearchFiltersProps {
  onApply: (filters: FilterOptions) => void;
  onClear?: () => void;
}

export const SearchFilters: React.FC<SearchFiltersProps> = ({ onApply, onClear }) => {
  const [filters, setFilters] = useState<FilterOptions>({});

  const handleApply = () => {
    onApply(filters);
  };

  const handleClear = () => {
    setFilters({});
    onClear?.();
  };

  return (
    <Card title="Filters" className="search-filters">
      <Collapse
        items={[
          {
            key: '1',
            label: 'Advanced Filters',
            children: (
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <label>Date Range:</label>
                  <DatePicker.RangePicker
                    onChange={(dates) => {
                      if (dates) {
                        setFilters({
                          ...filters,
                          dateFrom: dates[0]?.format('YYYY-MM-DD'),
                          dateTo: dates[1]?.format('YYYY-MM-DD'),
                        });
                      }
                    }}
                  />
                </div>

                <div>
                  <label>Category:</label>
                  <Select
                    placeholder="Select category"
                    onChange={(value) => setFilters({ ...filters, category: value })}
                    style={{ width: '100%' }}
                    options={[
                      { label: 'Computer Science', value: 'cs' },
                      { label: 'Biology', value: 'bio' },
                      { label: 'Physics', value: 'phys' },
                      { label: 'Mathematics', value: 'math' },
                    ]}
                  />
                </div>

                <div>
                  <label>Minimum Citations:</label>
                  <Select
                    placeholder="Select minimum citations"
                    onChange={(value) => setFilters({ ...filters, minCitations: value })}
                    style={{ width: '100%' }}
                    options={[
                      { label: 'No filter', value: 0 },
                      { label: '10+', value: 10 },
                      { label: '50+', value: 50 },
                      { label: '100+', value: 100 },
                    ]}
                  />
                </div>

                <Space>
                  <Button type="primary" onClick={handleApply}>
                    Apply Filters
                  </Button>
                  <Button onClick={handleClear}>
                    Clear
                  </Button>
                </Space>
              </Space>
            ),
          },
        ]}
      />
    </Card>
  );
};

export default SearchFilters;