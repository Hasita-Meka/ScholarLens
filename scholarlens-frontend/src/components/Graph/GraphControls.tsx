// src/components/Graph/GraphControls.tsx
import React from 'react';
import { Button, Select, Space, Slider } from 'antd';
import { ZoomInOutlined, ZoomOutOutlined, CompassOutlined } from '@ant-design/icons';


interface GraphControlsProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  onLayoutChange: (layout: string) => void;
  onNodeSizeChange: (size: number) => void;
}

export const GraphControls: React.FC<GraphControlsProps> = ({
  onZoomIn,
  onZoomOut,
  onReset,
  onLayoutChange,
  onNodeSizeChange,
}) => {
  return (
    <div className="graph-controls" style={{ padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Space>
          <Button icon={<ZoomInOutlined />} onClick={onZoomIn} type="primary">
            Zoom In
          </Button>
          <Button icon={<ZoomOutOutlined />} onClick={onZoomOut}>
            Zoom Out
          </Button>
          <Button icon={<CompassOutlined />} onClick={onReset}>
            Reset View
          </Button>
        </Space>

        <div>
          <label>Layout: </label>
          <Select
            defaultValue="force"
            onChange={onLayoutChange}
            style={{ width: 150 }}
            options={[
              { label: 'Force', value: 'force' },
              { label: 'Hierarchical', value: 'hierarchical' },
              { label: 'Circular', value: 'circular' },
            ]}
          />
        </div>

        <div>
          <label>Node Size: </label>
          <Slider
            min={10}
            max={50}
            defaultValue={30}
            onChange={onNodeSizeChange}
            style={{ width: 200 }}
          />
        </div>
      </Space>
    </div>
  );
};

export default GraphControls;