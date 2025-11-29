import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card } from 'antd';

interface TrendData {
  month: string;
  papers: number;
}

interface TrendChartProps {
  data: TrendData[];
  title?: string;
}

const TrendChart: React.FC<TrendChartProps> = ({ data, title = 'Publication Trends' }) => {
  return (
    <Card title={title}>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="month" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="papers" 
            stroke="#1890ff" 
            activeDot={{ r: 8 }} 
          />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );
};

export default TrendChart;
