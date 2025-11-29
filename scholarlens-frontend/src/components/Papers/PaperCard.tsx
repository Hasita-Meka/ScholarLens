import React from 'react';
import { Card, Tag, Row, Col, Button } from 'antd';
import { DownloadOutlined, LinkOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';

interface PaperCardProps {
  paper_id: number;
  title: string;
  authors: string[];
  abstract: string;
  published_date: string;
  categories: string[];
  arxiv_id: string;
}

const PaperCard: React.FC<PaperCardProps> = ({
  paper_id,
  title,
  authors,
  abstract,
  published_date,
  categories,
  arxiv_id,
}) => {
  const navigate = useNavigate();

  return (
    <Card
      hoverable
      style={{ marginBottom: '16px' }}
      onClick={() => navigate(`/papers/${paper_id}`)}
    >
      <h3>{title}</h3>
      <p style={{ color: '#888' }}>
        Authors: {authors.slice(0, 3).join(', ')}
        {authors.length > 3 && ` +${authors.length - 3} more`}
      </p>
      <p>{abstract.substring(0, 200)}...</p>
      
      <Row justify="space-between" align="middle">
        <Col>
          {categories.map((cat) => (
            <Tag key={cat} color="blue">{cat}</Tag>
          ))}
        </Col>
        <Col>
          <Button 
            type="primary" 
            icon={<DownloadOutlined />}
            onClick={(e) => {
              e.stopPropagation();
              window.open(`https://arxiv.org/pdf/${arxiv_id}.pdf`, '_blank');
            }}
          >
            PDF
          </Button>
        </Col>
      </Row>
    </Card>
  );
};

export default PaperCard;
