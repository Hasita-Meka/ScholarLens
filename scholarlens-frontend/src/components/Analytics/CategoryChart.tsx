// src/components/Analytics/CategoryChart.tsx
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface CategoryData {
  name: string;
  count: number;
  percentage: number;
}

interface CategoryChartProps {
  data: CategoryData[];
  title?: string;
}

export const CategoryChart: React.FC<CategoryChartProps> = ({ data, title = 'Papers by Category' }) => {
  if (!data || data.length === 0) {
    return <div className="empty-chart">No data available</div>;
  }

  return (
    <div className="category-chart">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="count" fill="#8884d8" />
          <Bar dataKey="percentage" fill="#82ca9d" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default CategoryChart;