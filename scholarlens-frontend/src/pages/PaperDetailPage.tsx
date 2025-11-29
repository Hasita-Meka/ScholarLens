// src/pages/PaperDetailPage.tsx
import React from 'react';
import { Layout, Spin } from 'antd';
import { useParams } from 'react-router-dom';
import PaperDetail from '@components/Papers/PaperDetail';

const { Content } = Layout;

export const PaperDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [loading, setLoading] = React.useState(false);

  return (
    <Layout>
      <Content style={{ padding: '20px' }}>
        {loading ? <Spin /> : <PaperDetail paper={null} loading={loading} />}
      </Content>
    </Layout>
  );
};

export default PaperDetailPage;