import React, { useEffect, useState } from 'react';
import { Row, Col, Card, Spin } from 'antd';
import ApiService from '../services/api';
import StatisticsCards from '../components/Analytics/StatisticsCards';
import TrendChart from '../components/Analytics/TrendChart';

const AnalyticsPage: React.FC = () => {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const response = await ApiService.getStatistics();
        setStats(response.data);
      } catch (err) {
        console.error('Failed to fetch analytics:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, []);

  if (loading) return <Spin size="large" />;

  const trendData = [
    { month: 'Jan', papers: 10 },
    { month: 'Feb', papers: 15 },
    { month: 'Mar', papers: 20 },
    // Add more data
  ];

  return (
    <div>
      <h1>Analytics Dashboard</h1>

      {stats && <StatisticsCards stats={stats} />}

      <Row gutter={[16, 16]} style={{ marginTop: '24px' }}>
        <Col xs={24} lg={12}>
          <TrendChart data={trendData} />
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Category Distribution">
            {/* Add pie chart here */}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default AnalyticsPage;
