import React, { useState, useEffect } from 'react';
import { Spin, Alert, Button, Space, message } from 'antd';
import { DownloadOutlined, ReloadOutlined } from '@ant-design/icons';
import Plot from 'react-plotly.js';
import { getSpringFestivalChart, exportChart } from '../services/api';

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

interface SpringFestivalChartProps {
  stock: StockInfo | null;
  config: ChartConfig;
}

const SpringFestivalChart: React.FC<SpringFestivalChartProps> = ({ 
  stock, 
  config 
}) => {
  const [chartData, setChartData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 获取图表数据
  const fetchChartData = async () => {
    if (!stock) {
      setChartData(null);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await getSpringFestivalChart({
        symbol: stock.symbol,
        years: config.years,
        chart_type: config.chartType,
        show_pattern_info: config.showPatternInfo,
        title: `${stock.name} 春节分析`
      });

      setChartData(response.chart_data);
    } catch (err: any) {
      console.error('获取图表数据失败:', err);
      setError(err.message || '获取图表数据失败');
    } finally {
      setLoading(false);
    }
  };

  // 当股票或配置改变时重新获取数据
  useEffect(() => {
    fetchChartData();
  }, [stock, config]);

  // 导出图表
  const handleExport = async (format: string) => {
    if (!chartData) {
      message.warning('没有可导出的图表数据');
      return;
    }

    try {
      const filename = `${stock?.name || 'chart'}_春节分析.${format}`;
      await exportChart({
        chart_data: chartData,
        format,
        filename
      });
      message.success(`图表已导出为 ${format.toUpperCase()} 格式`);
    } catch (err: any) {
      console.error('导出图表失败:', err);
      message.error('导出图表失败');
    }
  };

  // 刷新图表
  const handleRefresh = () => {
    fetchChartData();
  };

  // 渲染空状态
  if (!stock) {
    return (
      <div className="loading-container">
        <div style={{ textAlign: 'center', color: '#999' }}>
          <p>请先搜索并选择一只股票</p>
          <p style={{ fontSize: '12px' }}>选择股票后将显示春节分析图表</p>
        </div>
      </div>
    );
  }

  // 渲染加载状态
  if (loading) {
    return (
      <div className="loading-container">
        <Spin size="large" tip="正在生成图表..." />
      </div>
    );
  }

  // 渲染错误状态
  if (error) {
    return (
      <div className="error-container">
        <Alert
          message="图表加载失败"
          description={error}
          type="error"
          showIcon
          action={
            <Button size="small" onClick={handleRefresh}>
              重试
            </Button>
          }
        />
      </div>
    );
  }

  // 渲染图表
  return (
    <div className="spring-festival-chart">
      {/* 工具栏 */}
      <div className="chart-toolbar">
        <div>
          <Space>
            <Button 
              icon={<ReloadOutlined />} 
              onClick={handleRefresh}
              size="small"
            >
              刷新
            </Button>
          </Space>
        </div>
        
        <div>
          <Space>
            <Button 
              icon={<DownloadOutlined />}
              onClick={() => handleExport('png')}
              size="small"
            >
              PNG
            </Button>
            <Button 
              icon={<DownloadOutlined />}
              onClick={() => handleExport('svg')}
              size="small"
            >
              SVG
            </Button>
            <Button 
              icon={<DownloadOutlined />}
              onClick={() => handleExport('html')}
              size="small"
            >
              HTML
            </Button>
          </Space>
        </div>
      </div>

      {/* 图表 */}
      {chartData && (
        <div className="plotly-chart">
          <Plot
            data={chartData.data}
            layout={{
              ...chartData.layout,
              autosize: true,
              responsive: true,
              margin: { l: 60, r: 60, t: 80, b: 60 }
            }}
            config={{
              displayModeBar: true,
              displaylogo: false,
              modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
              toImageButtonOptions: {
                format: 'png',
                filename: `${stock.name}_春节分析`,
                height: 600,
                width: 1000,
                scale: 2
              }
            }}
            style={{ width: '100%', height: '500px' }}
            useResizeHandler={true}
          />
        </div>
      )}

      {/* 图表说明 */}
      <div style={{ 
        marginTop: 16, 
        padding: 12, 
        background: '#f9f9f9', 
        borderRadius: 6,
        fontSize: '12px',
        color: '#666'
      }}>
        <p><strong>图表说明：</strong></p>
        <p>• 横轴表示相对春节的天数（负数为春节前，正数为春节后）</p>
        <p>• 纵轴表示标准化收益率（相对于基准价格的百分比变化）</p>
        <p>• 红色虚线标记春节日期</p>
        <p>• 不同颜色的线条代表不同年份的数据</p>
      </div>
    </div>
  );
};

export default SpringFestivalChart;