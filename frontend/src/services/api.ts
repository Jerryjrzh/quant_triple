import axios from 'axios';

// 创建axios实例
const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 可以在这里添加认证token等
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    const message = error.response?.data?.detail || error.message || '请求失败';
    return Promise.reject(new Error(message));
  }
);

// 股票信息接口
export interface StockInfo {
  symbol: string;
  name: string;
  market?: string;
  sector?: string;
}

// 图表请求参数
export interface ChartRequest {
  symbol: string;
  years?: number[];
  start_date?: string;
  end_date?: string;
  chart_type?: string;
  title?: string;
  show_pattern_info?: boolean;
}

// 多股票图表请求参数
export interface MultiStockChartRequest {
  symbols: string[];
  years?: number[];
  chart_type?: string;
  title?: string;
}

// 导出请求参数
export interface ExportRequest {
  chart_data: any;
  format: string;
  filename?: string;
}

// API函数

/**
 * 搜索股票
 */
export const searchStocks = async (query: string): Promise<StockInfo[]> => {
  // 模拟股票搜索数据，实际应该调用后端API
  const mockStocks: StockInfo[] = [
    { symbol: '000001.SZ', name: '平安银行', market: '深圳', sector: '金融' },
    { symbol: '000002.SZ', name: '万科A', market: '深圳', sector: '房地产' },
    { symbol: '600000.SH', name: '浦发银行', market: '上海', sector: '金融' },
    { symbol: '600036.SH', name: '招商银行', market: '上海', sector: '金融' },
    { symbol: '000858.SZ', name: '五粮液', market: '深圳', sector: '食品饮料' },
    { symbol: '600519.SH', name: '贵州茅台', market: '上海', sector: '食品饮料' },
    { symbol: '000725.SZ', name: '京东方A', market: '深圳', sector: '电子' },
    { symbol: '002415.SZ', name: '海康威视', market: '深圳', sector: '电子' },
  ];

  // 简单的模糊搜索
  const filtered = mockStocks.filter(stock => 
    stock.symbol.toLowerCase().includes(query.toLowerCase()) ||
    stock.name.includes(query)
  );

  // 模拟网络延迟
  await new Promise(resolve => setTimeout(resolve, 300));
  
  return filtered;
};

/**
 * 获取股票列表
 */
export const getStocks = async (params?: {
  limit?: number;
  offset?: number;
  search?: string;
}): Promise<{ stocks: StockInfo[]; total: number }> => {
  try {
    const response = await api.get('/stocks', { params });
    return response as unknown as { stocks: StockInfo[]; total: number };
  } catch (error) {
    console.error('获取股票列表失败:', error);
    // 返回模拟数据作为fallback
    const mockStocks = await searchStocks('');
    return { stocks: mockStocks, total: mockStocks.length };
  }
};

/**
 * 获取春节分析图表
 */
export const getSpringFestivalChart = async (request: ChartRequest): Promise<any> => {
  try {
    const response = await api.post('/visualization/spring-festival-chart', request);
    return response;
  } catch (error) {
    console.error('获取春节图表失败:', error);
    
    // 如果后端不可用，返回示例图表数据
    const sampleResponse = await api.get(`/visualization/sample?symbol=${request.symbol}&format=json`);
    return sampleResponse;
  }
};

/**
 * 获取多股票对比图表
 */
export const getMultiStockChart = async (request: MultiStockChartRequest): Promise<any> => {
  const response = await api.post('/visualization/multi-stock-chart', request);
  return response;
};

/**
 * 导出图表
 */
export const exportChart = async (request: ExportRequest): Promise<void> => {
  const response = await api.post('/visualization/export', request, {
    responseType: 'blob'
  });
  
  // 创建下载链接
  const url = window.URL.createObjectURL(new Blob([response as any]));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', request.filename || `chart.${request.format}`);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
};

/**
 * 获取可用的图表类型
 */
export const getChartTypes = async (): Promise<any> => {
  const response = await api.get('/visualization/chart-types');
  return response;
};

/**
 * 获取图表配置
 */
export const getChartConfig = async (): Promise<any> => {
  const response = await api.get('/visualization/config');
  return response;
};

/**
 * 健康检查
 */
export const healthCheck = async (): Promise<any> => {
  const response = await api.get('/visualization/health');
  return response;
};

/**
 * 获取API信息
 */
export const getApiInfo = async (): Promise<any> => {
  const response = await api.get('/info');
  return response;
};

export default api;