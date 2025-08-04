import React from 'react';
import { Layout, ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import 'dayjs/locale/zh-cn';
import './App.css';
import Header from './components/Header';
import MainContent from './components/MainContent';

const { Content } = Layout;

const App: React.FC = () => {
  return (
    <ConfigProvider locale={zhCN}>
      <Layout className="app-container">
        <Header />
        <Content>
          <MainContent />
        </Content>
      </Layout>
    </ConfigProvider>
  );
};

export default App;