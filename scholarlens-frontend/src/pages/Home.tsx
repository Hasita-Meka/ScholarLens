import React, { useEffect, useState } from 'react';
import { Row, Col, Card, Button, Spin } from 'antd';
import { ArrowRightOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import ApiService from '../services/api';
import StatisticsCards from '../components/Analytics/StatisticsCards';
import PaperList from '../components/Papers/PaperList';
import { usePapers } from '../hooks/usePapers';

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState<any>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const { papers, loading, error, total } = usePapers(1, 10);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await ApiService.getStatistics();
        setStats(response.data);
      } catch (err) {
        console.error('Failed to fetch statistics:', err);
      } finally {
        setStatsLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (statsLoading) return <Spin size="large" />;

  return (
    <div>
      <h1>Welcome to ScholarLens</h1>
      <p>Explore research papers, discover knowledge graphs, and analyze trends</p>
      
      {stats && <StatisticsCards stats={stats} />}

      <Row gutter={[16, 16]} style={{ marginTop: '24px' }}>
        <Col span={24}>
          <Card 
            title="Recent Papers"
            extra={<Button 
              type="link" 
              icon={<ArrowRightOutlined />}
              onClick={() => navigate('/papers')}
            >
              View All
            </Button>}
          >
            <PaperList 
              papers={papers} 
              loading={loading} 
              total={total}
              onPageChange={(page) => {
                // Handle page change
              }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default HomePage;
