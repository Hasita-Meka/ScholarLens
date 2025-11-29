import React, { useState } from 'react';
import { List, Pagination, Spin } from 'antd';
import PaperCard from './PaperCard';

interface Paper {
  paper_id: number;
  title: string;
  authors: string[];
  abstract: string;
  published_date: string;
  categories: string[];
  arxiv_id: string;
}

interface PaperListProps {
  papers: Paper[];
  loading: boolean;
  total: number;
  onPageChange: (page: number) => void;
}

const PaperList: React.FC<PaperListProps> = ({ papers, loading, total, onPageChange }) => {
  const [currentPage, setCurrentPage] = useState(1);

  if (loading) return <Spin size="large" />;

  return (
    <>
      <List
        dataSource={papers}
        renderItem={(paper) => (
          <List.Item key={paper.paper_id}>
            <PaperCard {...paper} />
          </List.Item>
        )}
      />
      <Pagination
        current={currentPage}
        total={total}
        pageSize={20}
        onChange={(page) => {
          setCurrentPage(page);
          onPageChange(page);
        }}
        style={{ marginTop: '24px', textAlign: 'center' }}
      />
    </>
  );
};

export default PaperList;
