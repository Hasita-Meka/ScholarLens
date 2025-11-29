// src/components/Papers/PaperDetail.tsx
import React from 'react';
import { Card, Tag, Button, Space, Divider, List, Descriptions } from 'antd';
import { DownloadOutlined, LinkOutlined, HeartOutlined } from '@ant-design/icons';
import type { Paper, Author } from '@schemas/papers';

interface PaperDetailProps {
  paper: Paper | null;
  loading?: boolean;
  onBack?: () => void;
}

export const PaperDetail: React.FC<PaperDetailProps> = ({ paper, loading = false, onBack }) => {
  if (!paper) {
    return <Card loading={loading}>No paper selected</Card>;
  }

  return (
    <Card loading={loading} className="paper-detail">
      <Button onClick={onBack} style={{ marginBottom: '15px' }}>
        ← Back
      </Button>

      <div className="paper-detail-header">
        <h1>{paper.title}</h1>
        <Space>
          <Tag color="blue">{paper.category}</Tag>
          {paper.citationCount && (
            <Tag color="green">Citations: {paper.citationCount}</Tag>
          )}
        </Space>
      </div>

      <Divider />

      <Descriptions column={1} bordered>
        <Descriptions.Item label="Abstract">
          <p>{paper.abstract}</p>
        </Descriptions.Item>
        <Descriptions.Item label="Publication Date">
          {new Date(paper.publicationDate).toLocaleDateString()}
        </Descriptions.Item>
        <Descriptions.Item label="Authors">
          {paper.authors && paper.authors.map((author: Author) => (
            <Tag key={author.id}>{author.name}</Tag>
          ))}
        </Descriptions.Item>
        {paper.keywords && (
          <Descriptions.Item label="Keywords">
            {paper.keywords.map((kw: string) => (
              <Tag key={kw} color="cyan">
                {kw}
              </Tag>
            ))}
          </Descriptions.Item>
        )}
        {paper.methods && (
          <Descriptions.Item label="Methods">
            <List
              size="small"
              dataSource={paper.methods}
              renderItem={(method: string) => <List.Item>{method}</List.Item>}
            />
          </Descriptions.Item>
        )}
      </Descriptions>

      <Divider />

      <Space style={{ marginTop: '20px' }}>
        <Button
          type="primary"
          icon={<DownloadOutlined />}
          onClick={() => window.open(paper.pdfUrl, '_blank')}
        >
          Download PDF
        </Button>
        <Button
          icon={<LinkOutlined />}
          onClick={() => window.open(paper.pdfUrl, '_blank')}
        >
          View Online
        </Button>
        <Button icon={<HeartOutlined />}>
          Save
        </Button>
      </Space>
    </Card>
  );
};

export default PaperDetail;