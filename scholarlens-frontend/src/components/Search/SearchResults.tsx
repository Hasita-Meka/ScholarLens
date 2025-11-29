// src/components/Search/SearchResults.tsx
import React from 'react';
import { List, Card, Tag, Button, Space, Empty } from 'antd';
import { FileOutlined, LinkOutlined, HeartOutlined } from '@ant-design/icons';
import type { Paper } from '@schemas/papers';

interface SearchResultsProps {
  results: Paper[];
  loading?: boolean;
  onSelectPaper?: (paper: Paper) => void;
  onSavePaper?: (paper: Paper) => void;
}

export const SearchResults: React.FC<SearchResultsProps> = ({
  results,
  loading = false,
  onSelectPaper,
  onSavePaper,
}) => {
  if (!loading && (!results || results.length === 0)) {
    return <Empty description="No results found" />;
  }

  return (
    <List
      className="search-results"
      loading={loading}
      dataSource={results}
      renderItem={(paper) => (
        <List.Item key={paper.id}>
          <Card style={{ width: '100%' }} hoverable onClick={() => onSelectPaper?.(paper)}>
            <div className="paper-result">
              <h3>{paper.title}</h3>
              <p className="abstract">{paper.abstract?.substring(0, 200)}...</p>
              <Space style={{ marginTop: '10px' }}>
                {paper.category && <Tag color="blue">{paper.category}</Tag>}
                {paper.citationCount && (
                  <Tag color="green">Citations: {paper.citationCount}</Tag>
                )}
              </Space>
              <div className="paper-authors" style={{ marginTop: '8px' }}>
                <strong>Authors:</strong>{' '}
                {paper.authors?.map((a) => a.name).join(', ')}
              </div>
              <Space style={{ marginTop: '12px' }}>
                <Button
                  type="primary"
                  icon={<FileOutlined />}
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    window.open(paper.pdfUrl, '_blank');
                  }}
                >
                  View PDF
                </Button>
                <Button
                  icon={<HeartOutlined />}
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSavePaper?.(paper);
                  }}
                >
                  Save
                </Button>
              </Space>
            </div>
          </Card>
        </List.Item>
      )}
    />
  );
};

export default SearchResults;