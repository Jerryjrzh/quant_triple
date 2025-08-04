import React from 'react';
import { Layout, Menu } from 'antd';
import { 
  BarChartOutlined, 
  DashboardOutlined, 
  SearchOutlined,
  SettingOutlined 
} from '@ant-design/icons';

const { Header: AntHeader } = Layout;

interface HeaderProps {
  selectedKey?: string;
  onMenuSelect?: (key: string) => void;
}

const Header: React.FC<HeaderProps> = ({ 
  selectedKey = 'analysis', 
  onMenuSelect 
}) => {
  const menuItems = [
    {
      key: 'analysis',
      icon: <BarChartOutlined />,
      label: '春节分析',
    },
    {
      key: 'dashboard',
      icon: <DashboardOutlined />,
      label: '仪表板',
    },
    {
      key: 'search',
      icon: <SearchOutlined />,
      label: '股票搜索',
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: '设置',
    },
  ];

  const handleMenuClick = ({ key }: { key: string }) => {
    onMenuSelect?.(key);
  };

  return (
    <AntHeader style={{ 
      display: 'flex', 
      alignItems: 'center',
      background: '#001529',
      padding: '0 24px'
    }}>
      <div className="logo" style={{ 
        color: 'white', 
        fontSize: '20px', 
        fontWeight: 'bold',
        marginRight: '40px',
        minWidth: '200px'
      }}>
        股票分析系统
      </div>
      
      <Menu
        theme="dark"
        mode="horizontal"
        selectedKeys={[selectedKey]}
        items={menuItems}
        onClick={handleMenuClick}
        style={{ 
          flex: 1, 
          minWidth: 0,
          background: 'transparent'
        }}
      />
    </AntHeader>
  );
};

export default Header;