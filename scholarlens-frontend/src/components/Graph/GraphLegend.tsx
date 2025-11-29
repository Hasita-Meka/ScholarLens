// src/components/Graph/GraphLegend.tsx
import React from 'react';
import { Card } from 'antd';

interface LegendItem {
  color: string;
  label: string;
  description?: string;
}

interface GraphLegendProps {
  items: LegendItem[];
}

export const GraphLegend: React.FC<GraphLegendProps> = ({ items }) => {
  return (
    <Card className="graph-legend" title="Legend" style={{ marginTop: '15px' }} size="small">
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {items.map((item, index) => (
          <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div
              style={{
                width: '16px',
                height: '16px',
                backgroundColor: item.color,
                borderRadius: '2px',
              }}
            />
            <span style={{ fontWeight: '500' }}>{item.label}</span>
            {item.description && (
              <span style={{ fontSize: '12px', color: '#999' }}>- {item.description}</span>
            )}
          </div>
        ))}
      </div>
    </Card>
  );
};

export default GraphLegend;