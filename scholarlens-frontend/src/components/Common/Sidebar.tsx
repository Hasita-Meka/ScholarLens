import React from 'react';
import { Menu, Layout } from 'antd';
import { HomeOutlined, FileTextOutlined, AreaChartOutlined, TeamOutlined, GlobalOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';

const { Sider } = Layout;

const AppSidebar: React.FC = () => {
  const navigate = useNavigate();

  const menuItems = [
    { key: '1', icon: <HomeOutlined />, label: 'Home', path: '/' },
    { key: '2', icon: <FileTextOutlined />, label: 'Papers', path: '/papers' },
    { key: '3', icon: <GlobalOutlined />, label: 'Knowledge Graph', path: '/graph' },
    { key: '4', icon: <AreaChartOutlined />, label: 'Analytics', path: '/analytics' },
    { key: '5', icon: <TeamOutlined />, label: 'Authors', path: '/authors' },
  ];

  return (
    <Sider>
      <Menu
        theme="dark"
        mode="inline"
        defaultSelectedKeys={['1']}
        items={menuItems}
        onClick={(e) => {
          const item = menuItems.find(m => m.key === e.key);
          if (item) navigate(item.path);
        }}
      />
    </Sider>
  );
};

export default AppSidebar;
