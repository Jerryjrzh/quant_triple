import React from 'react';
import { Form, Select, Switch, Space, Divider } from 'antd';

const { Option } = Select;

interface ChartConfig {
  years: number[];
  chartType: string;
  showPatternInfo: boolean;
}

interface ChartControlsProps {
  config: ChartConfig;
  onChange: (config: Partial<ChartConfig>) => void;
}

const ChartControls: React.FC<ChartControlsProps> = ({ config, onChange }) => {
  const availableYears = [2018, 2019, 2020, 2021, 2022, 2023, 2024];
  
  const chartTypes = [
    { value: 'overlay', label: '叠加图' },
    { value: 'comparison', label: '对比图' },
    { value: 'pattern', label: '模式图' }
  ];

  const handleYearsChange = (selectedYears: number[]) => {
    onChange({ years: selectedYears });
  };

  const handleChartTypeChange = (chartType: string) => {
    onChange({ chartType });
  };

  const handlePatternInfoChange = (showPatternInfo: boolean) => {
    onChange({ showPatternInfo });
  };

  return (
    <div className="chart-controls">
      <Form layout="vertical" size="small">
        <Form.Item label="选择年份">
          <Select
            mode="multiple"
            placeholder="选择要分析的年份"
            value={config.years}
            onChange={handleYearsChange}
            style={{ width: '100%' }}
            maxTagCount={3}
          >
            {availableYears.map(year => (
              <Option key={year} value={year}>
                {year}年
              </Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item label="图表类型">
          <Select
            value={config.chartType}
            onChange={handleChartTypeChange}
            style={{ width: '100%' }}
          >
            {chartTypes.map(type => (
              <Option key={type.value} value={type.value}>
                {type.label}
              </Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item>
          <Space direction="vertical">
            <Switch
              checked={config.showPatternInfo}
              onChange={handlePatternInfoChange}
              checkedChildren="显示模式信息"
              unCheckedChildren="隐藏模式信息"
            />
          </Space>
        </Form.Item>

        <Divider />

        <div style={{ fontSize: '12px', color: '#666' }}>
          <p>• 叠加图：多年数据叠加显示</p>
          <p>• 对比图：年份间对比分析</p>
          <p>• 模式图：显示季节性模式</p>
        </div>
      </Form>
    </div>
  );
};

export default ChartControls;