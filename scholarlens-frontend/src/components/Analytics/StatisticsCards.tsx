import React from 'react';
import { Row, Col, Card, Statistic } from 'antd';
import { FileTextOutlined, TeamOutlined, FunctionOutlined, DatabaseOutlined } from '@ant-design/icons';

interface StatsData {
  papers: number;
  authors: number;
  methods: number;
  datasets: number;
}

interface StatisticsCardsProps {
  stats: StatsData;
}

const StatisticsCards: React.FC<StatisticsCardsProps> = ({ stats }) => {
  return (
    <Row gutter={[16, 16]}>
      <Col xs={24} sm={12} lg={6}>
        <Card>
          <Statistic
            title="Total Papers"
            value={stats.papers}
            prefix={<FileTextOutlined />}
            valueStyle={{ color: '#1890ff' }}
          />
        </Card>
      </Col>
      <Col xs={24} sm={12} lg={6}>
        <Card>
          <Statistic
            title="Authors"
            value={stats.authors}
            prefix={<TeamOutlined />}
            valueStyle={{ color: '#52c41a' }}
          />
        </Card>
      </Col>
      <Col xs={24} sm={12} lg={6}>
        <Card>
          <Statistic
            title="Methods"
            value={stats.methods}
            prefix={<FunctionOutlined />}
            valueStyle={{ color: '#faad14' }}
          />
        </Card>
      </Col>
      <Col xs={24} sm={12} lg={6}>
        <Card>
          <Statistic
            title="Datasets"
            value={stats.datasets}
            prefix={<DatabaseOutlined />}
            valueStyle={{ color: '#eb2f96' }}
          />
        </Card>
      </Col>
    </Row>
  );
};

export default StatisticsCards;
