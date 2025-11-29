// src/components/Analytics/TopAuthorsTable.tsx
import React from 'react';
import { Table, Tag } from 'antd';
import type { ColumnsType } from 'antd/es/table';

interface AuthorData {
  id: string;
  name: string;
  paperCount: number;
  citations: number;
  h_index?: number;
}

interface TopAuthorsTableProps {
  data: AuthorData[];
  loading?: boolean;
}

export const TopAuthorsTable: React.FC<TopAuthorsTableProps> = ({ data, loading = false }) => {
  const columns: ColumnsType<AuthorData> = [
    {
      title: 'Author Name',
      dataIndex: 'name',
      key: 'name',
      render: (text) => <strong>{text}</strong>,
    },
    {
      title: 'Papers',
      dataIndex: 'paperCount',
      key: 'paperCount',
      sorter: (a, b) => a.paperCount - b.paperCount,
    },
    {
      title: 'Citations',
      dataIndex: 'citations',
      key: 'citations',
      sorter: (a, b) => a.citations - b.citations,
    },
    {
      title: 'h-index',
      dataIndex: 'h_index',
      key: 'h_index',
      render: (index) => index ? <Tag color="blue">{index}</Tag> : <span>-</span>,
    },
  ];

  return (
    <div className="top-authors-table">
      <h3>Top Authors</h3>
      <Table
        columns={columns}
        dataSource={data}
        rowKey="id"
        loading={loading}
        pagination={{ pageSize: 10 }}
        size="small"
      />
    </div>
  );
};

export default TopAuthorsTable;