import React from 'react';
import { Layout, Input, Button, Row, Col } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import SearchBar from '../Search/SearchBar';

const { Header } = Layout;

const AppHeader: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Header style={{ background: '#fff', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
      <Row justify="space-between" align="middle">
        <Col>
          <h1 
            onClick={() => navigate('/')}
            style={{ cursor: 'pointer', margin: 0, color: '#1890ff' }}
          >
            🔬 ScholarLens
          </h1>
        </Col>
        <Col span={8}>
          <SearchBar />
        </Col>
        <Col>
          <Button type="primary" onClick={() => navigate('/analytics')}>
            Analytics
          </Button>
        </Col>
      </Row>
    </Header>
  );
};

export default AppHeader;
