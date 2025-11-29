import React, { useState } from 'react';
import { Row, Col, Card, Select, DatePicker } from 'antd';
import PaperList from '../components/Papers/PaperList';
import { usePapers } from '../hooks/usePapers';

const PapersPage: React.FC = () => {
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const { papers, loading, total } = usePapers(currentPage, 20);

  return (
    <div>
      <h1>Research Papers</h1>
      
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} md={6}>
          <Select
            defaultValue="all"
            style={{ width: '100%' }}
            onChange={setSelectedCategory}
            options={[
              { label: 'All Categories', value: 'all' },
              { label: 'Machine Learning', value: 'cs.LG' },
              { label: 'AI', value: 'cs.AI' },
              { label: 'NLP', value: 'cs.CL' },
            ]}
          />
        </Col>
      </Row>

      <Card>
        <PaperList 
          papers={papers}
          loading={loading}
          total={total}
          onPageChange={setCurrentPage}
        />
      </Card>
    </div>
  );
};

export default PapersPage;
