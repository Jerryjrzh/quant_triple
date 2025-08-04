import React, { useState } from 'react';
import { Card, Row, Col } from 'antd';
import StockSearch from './StockSearch';
import SpringFestivalChart from './SpringFestivalChart';
import ChartControls from './ChartControls';

interface StockInfo {
  symbol: string;
  name: string;
  market?: string;
  sector?: string;
}

interface ChartConfig {
  years: number[];
  chartType: string;
  showPatternInfo: boolean;
}

const MainContent: React.FC = () => {
  const [selectedStock, setSelectedStock] = useState<StockInfo | null>(null);
  const [chartConfig, setChartConfig] = useState<ChartConfig>({
    years: [2020, 2021, 2022, 2023],
    chartType: 'overlay',
    showPatternInfo: true
  });

  const handleStockSelect = (stock: StockInfo) => {
    setSelectedStock(stock);
  };

  const handleConfigChange = (config: Partial<ChartConfig>) => {
    setChartConfig(prev => ({ ...prev, ...config }));
  };

  return (
    <div className="main-content">
      <Row gutter={[24, 24]}>
        {/* 股票搜索区域 */}
        <Col xs={24} lg={8}>
          <Card 
            title="股票搜索" 
            className="search-container"
            style={{ height: 'fit-content' }}
          >
            <StockSearch onStockSelect={handleStockSelect} />
            
            {selectedStock && (
              <div className="stock-info">
                <h4>{selectedStock.name} ({selectedStock.symbol})</h4>
                {selectedStock.market && <p>市场: {selectedStock.market}</p>}
                {selectedStock.sector && <p>行业: {selectedStock.sector}</p>}
              </div>
            )}
          </Card>

          {/* 图表控制区域 */}
          {selectedStock && (
            <Card 
              title="图表设置" 
              style={{ marginTop: 16 }}
            >
              <ChartControls 
                config={chartConfig}
                onChange={handleConfigChange}
              />
            </Card>
          )}
        </Col>

        {/* 图表显示区域 */}
        <Col xs={24} lg={16}>
          <Card 
            title={selectedStock ? `${selectedStock.name} 春节分析` : '春节分析图表'}
            className="chart-container"
          >
            <SpringFestivalChart 
              stock={selectedStock}
              config={chartConfig}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default MainContent;