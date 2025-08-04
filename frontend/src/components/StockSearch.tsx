import React, { useState, useCallback } from 'react';
import { AutoComplete, Input, message } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import { debounce } from 'lodash';
import { searchStocks } from '../services/api';

interface StockInfo {
  symbol: string;
  name: string;
  market?: string;
  sector?: string;
}

interface StockSearchProps {
  onStockSelect: (stock: StockInfo) => void;
}

const StockSearch: React.FC<StockSearchProps> = ({ onStockSelect }) => {
  const [options, setOptions] = useState<{ value: string; label: string; stock: StockInfo }[]>([]);
  const [loading, setLoading] = useState(false);

  // 搜索股票的防抖函数
  const debouncedSearch = useCallback(
    debounce(async (searchText: string) => {
      if (!searchText || searchText.length < 2) {
        setOptions([]);
        return;
      }

      setLoading(true);
      try {
        const stocks = await searchStocks(searchText);
        const searchOptions = stocks.map((stock: StockInfo) => ({
          value: stock.symbol,
          label: `${stock.name} (${stock.symbol})`,
          stock: stock
        }));
        setOptions(searchOptions);
      } catch (error) {
        console.error('搜索股票失败:', error);
        message.error('搜索股票失败，请稍后重试');
        setOptions([]);
      } finally {
        setLoading(false);
      }
    }, 300),
    []
  );

  const handleSearch = (value: string) => {
    debouncedSearch(value);
  };

  const handleSelect = (value: string) => {
    const selectedOption = options.find(option => option.value === value);
    if (selectedOption) {
      onStockSelect(selectedOption.stock);
      message.success(`已选择股票: ${selectedOption.stock.name}`);
    }
  };

  return (
    <div className="stock-search">
      <AutoComplete
        style={{ width: '100%' }}
        options={options}
        onSearch={handleSearch}
        onSelect={handleSelect}
        placeholder="输入股票代码或名称搜索"
        allowClear
        notFoundContent={loading ? '搜索中...' : '暂无数据'}
      >
        <Input
          prefix={<SearchOutlined />}
          placeholder="例如: 000001 或 平安银行"
          size="large"
        />
      </AutoComplete>
      
      <div style={{ marginTop: 8, fontSize: '12px', color: '#666' }}>
        支持股票代码和名称搜索
      </div>
    </div>
  );
};

export default StockSearch;