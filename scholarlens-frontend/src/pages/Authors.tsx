// src/pages/Authors.tsx
import React, { useState } from 'react';
import { Layout, Card, Input, Table } from 'antd';
import { SearchOutlined } from '@ant-design/icons';

const { Content } = Layout;

export const Authors: React.FC = () => {
  const [searchText, setSearchText] = useState('');

  return (
    <Layout>
      <Content style={{ padding: '20px' }}>
        <Card title="Authors" style={{ marginBottom: '20px' }}>
          <Input
            placeholder="Search authors..."
            prefix={<SearchOutlined />}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            style={{ marginBottom: '20px' }}
          />
          <Table
            columns={[
              { title: 'Name', dataIndex: 'name', key: 'name' },
              { title: 'Papers', dataIndex: 'papers', key: 'papers' },
              { title: 'Citations', dataIndex: 'citations', key: 'citations' },
            ]}
            dataSource={[]}
          />
        </Card>
      </Content>
    </Layout>
  );
};

export default Authors;