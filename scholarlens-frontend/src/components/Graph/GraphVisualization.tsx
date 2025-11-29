import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Spin } from 'antd';

interface Node {
  id: string;
  name: string;
  type: 'author' | 'paper';
  papers?: number;
}

interface Link {
  source: string;
  target: string;
  type: string;
}

interface GraphData {
  nodes: Node[];
  links: Link[];
}

interface GraphVisualizationProps {
  data: GraphData | null;
  loading?: boolean;
  width?: number;
  height?: number;
}

const GraphVisualization: React.FC<GraphVisualizationProps> = ({
  data,
  loading = false,
  width = 1000,
  height = 600,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data) return;

    // Clear previous content
    d3.select(svgRef.current).selectAll("*").remove();

    // Create simulation
    const simulation = d3.forceSimulation(data.nodes as any)
      .force('link', d3.forceLink(data.links as any)
        .id((d: any) => d.id)
        .distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    // Draw links
    const links = svg.append('g')
      .selectAll('line')
      .data(data.links)
      .enter()
      .append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', 2);

    // Draw nodes
    const nodes = svg.append('g')
      .selectAll('circle')
      .data(data.nodes)
      .enter()
      .append('circle')
      .attr('r', (d: any) => (d.type === 'author' ? 8 : 5))
      .attr('fill', (d: any) => (d.type === 'author' ? '#1890ff' : '#52c41a'))
      .call(d3.drag<SVGCircleElement, any>()
        .on('start', function(event: any) { dragstarted.call(this, event); })
        .on('drag', function(event: any) { dragged.call(this, event); })
        .on('end', function(event: any) { dragended.call(this, event); }));

    // Add labels
    const labels = svg.append('g')
      .selectAll('text')
      .data(data.nodes)
      .enter()
      .append('text')
      .text((d: any) => d.name.substring(0, 15))
      .attr('font-size', 12)
      .attr('text-anchor', 'middle');

    // Update on simulation tick
    simulation.on('tick', () => {
      links
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      nodes
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);

      labels
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y + 4);
    });

    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

  }, [data, width, height]);

  if (loading) return <Spin size="large" />;

  return <svg ref={svgRef} style={{ border: '1px solid #f0f0f0', borderRadius: '4px' }} />;
};

export default GraphVisualization;
