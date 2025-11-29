import React from 'react';
import { Card, Spin } from 'antd';
import GraphVisualization from '../components/Graph/GraphVisualization';
import { useGraph } from '../hooks/useGraph';

const GraphPage: React.FC = () => {
  const { graphData, loading, error } = useGraph();

  if (error) return <div>Error loading graph: {error}</div>;

  return (
    <div>
      <h1>Knowledge Graph</h1>
      <p>Author Collaboration Network and Research Relationships</p>
      
      <Card>
        <GraphVisualization 
          data={graphData} 
          loading={loading}
          width={1000}
          height={700}
        />
      </Card>
    </div>
  );
};

export default GraphPage;
