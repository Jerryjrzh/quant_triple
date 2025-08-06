"""
涨停原因数据适配器

基于 tmp/core/crawling/stock_limitup_reason.py 创建的涨停原因数据适配器，
提供涨停原因数据获取和详情解析的统一接口。

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import requests
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LimitUpRequest:
    """涨停原因请求结构"""
    date: str = ""  # 查询日期，格式：YYYY-MM-DD
    include_detail: bool = True  # 是否包含详细原因
    symbols: Optional[List[str]] = None  # 指定股票代码列表


@dataclass
class LimitUpResponse:
    """涨停原因响应结构"""
    success: bool
    data: pd.DataFrame
    error_message: str = ""
    response_time: float = 0.0
    data_source: str = "tonghuashun_limitup"
    timestamp: datetime = None
    date: str = ""


class LimitUpReasonAdapter:
    """涨停原因数据适配器"""
    
    def __init__(self, timeout: int = 15, max_retries: int = 3):
        """
        初始化适配器
        
        Args:
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # API端点
        self.base_url = "http://zx.10jqka.com.cn/event/api/getharden/date/{date}/orderby/date/orderway/desc/charset/GBK/"
        self.detail_url = "http://zx.10jqka.com.cn/event/harden/stockreason/id/{id}"
        
        # 请求头
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
        }
        
        # 涨停原因分类关键词
        self.reason_categories = {
            "政策利好": ["政策", "政府", "国家", "部委", "发改委", "财政部", "央行", "监管"],
            "业绩预增": ["业绩", "预增", "盈利", "净利润", "营收", "收入"],
            "重组并购": ["重组", "并购", "收购", "资产注入", "股权转让"],
            "热点概念": ["概念", "题材", "热点", "风口", "新兴"],
            "技术突破": ["技术", "研发", "专利", "创新", "突破"],
            "合同订单": ["合同", "订单", "中标", "签约", "协议"],
            "资金推动": ["资金", "主力", "机构", "北向", "外资"],
            "其他": []  # 默认分类
        }
        
        # 错误统计
        self.error_stats = {
            'network_errors': 0,
            'data_format_errors': 0,
            'api_errors': 0,
            'total_requests': 0,
            'successful_requests': 0
        }
    
    async def get_limitup_reason_data(self, request: LimitUpRequest) -> LimitUpResponse:
        """
        获取涨停原因数据
        
        Args:
            request: 涨停原因请求对象
            
        Returns:
            涨停原因响应对象
        """
        start_time = time.time()
        self.error_stats['total_requests'] += 1
        
        try:
            # 格式化日期
            date = request.date or datetime.now().strftime("%Y-%m-%d")
            
            # 构建URL
            url = self.base_url.format(date=date)
            
            # 发送请求
            response_data = await self._make_request(url)
            if not response_data:
                return LimitUpResponse(
                    success=False,
                    data=pd.DataFrame(),
                    error_message="未获取到涨停原因数据",
                    response_time=time.time() - start_time,
                    date=date
                )
            
            # 检查数据
            data = response_data.get("data", [])
            if not data:
                return LimitUpResponse(
                    success=True,
                    data=pd.DataFrame(),
                    error_message="当日无涨停股票",
                    response_time=time.time() - start_time,
                    date=date
                )
            
            # 处理基础数据
            df = self._process_basic_data(data)
            
            # 过滤指定股票
            if request.symbols:
                df = df[df['代码'].isin(request.symbols)]
            
            # 获取详细原因
            if request.include_detail and not df.empty:
                df = await self._add_detailed_reasons(df)
            
            # 文本内容清洗和结构化处理
            df = self._clean_and_structure_text(df)
            
            # 涨停原因分类和标签化
            df = self._categorize_reasons(df)
            
            self.error_stats['successful_requests'] += 1
            
            return LimitUpResponse(
                success=True,
                data=df,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                date=date
            )
            
        except Exception as e:
            logger.error(f"获取涨停原因数据失败: {e}")
            self.error_stats['api_errors'] += 1
            return LimitUpResponse(
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                response_time=time.time() - start_time,
                date=request.date or ""
            )
    
    async def get_multi_date_data(self, start_date: str, end_date: str, 
                                 include_detail: bool = True) -> Dict[str, LimitUpResponse]:
        """
        获取多日期涨停原因数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            include_detail: 是否包含详细原因
            
        Returns:
            多日期数据字典
        """
        results = {}
        
        # 生成日期列表
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y-%m-%d")
            
            # 跳过周末
            if current_dt.weekday() < 5:  # 0-4 是周一到周五
                request = LimitUpRequest(
                    date=date_str,
                    include_detail=include_detail
                )
                
                response = await self.get_limitup_reason_data(request)
                results[date_str] = response
                
                # 避免请求过快
                await asyncio.sleep(1)
            
            current_dt += timedelta(days=1)
        
        return results
    
    async def _make_request(self, url: str) -> Optional[Dict]:
        """发送HTTP请求（带重试机制）"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, headers=self.headers, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                self.error_stats['network_errors'] += 1
                logger.warning(f"请求超时，尝试 {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
            except requests.exceptions.RequestException as e:
                self.error_stats['network_errors'] += 1
                logger.error(f"网络请求失败: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
            except ValueError as e:
                self.error_stats['data_format_errors'] += 1
                logger.error(f"JSON解析失败: {e}")
                break
                
        return None
    
    def _process_basic_data(self, data: List) -> pd.DataFrame:
        """处理基础数据"""
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # 根据列数设置列名
        if len(df.columns) < 7:
            df.columns = ["ID", "名称", "代码", "原因", "日期", "_"]
            # 添加缺失列
            df["最新价"] = np.nan
            df["涨跌额"] = np.nan
            df["涨跌幅"] = np.nan
            df["换手率"] = np.nan
            df["成交额"] = np.nan
            df["成交量"] = np.nan
            df["DDE"] = np.nan
        else:
            df.columns = [
                "ID", "名称", "代码", "原因", "日期", "最新价", "涨跌额",
                "涨跌幅", "换手率", "成交额", "成交量", "DDE", "_"
            ]
        
        # 选择需要的列
        df = df[[
            "日期", "代码", "名称", "原因", "最新价", "涨跌幅", "涨跌额",
            "换手率", "成交量", "成交额", "DDE", "ID"
        ]]
        
        # 数据类型转换
        numeric_cols = ["最新价", "涨跌幅", "涨跌额", "换手率", "成交量", "成交额", "DDE"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 换手率保留2位小数
        df["换手率"] = df["换手率"].round(2)
        
        return df
    
    async def _add_detailed_reasons(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加详细原因"""
        detailed_reasons = []
        
        for _, row in df.iterrows():
            try:
                detail = await self._get_stock_detail(row['ID'])
                detailed_reasons.append(detail)
                
                # 避免请求过快
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"获取股票 {row['代码']} 详细原因失败: {e}")
                detailed_reasons.append("")
        
        df["详细原因"] = detailed_reasons
        return df
    
    async def _get_stock_detail(self, stock_id: str) -> str:
        """获取股票详细原因"""
        url = self.detail_url.format(id=stock_id)
        
        try:
            response = self.session.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            data_text = response.text
            
            # 提取详细原因
            pattern_data = re.search(r"var data = '(.*?)';", data_text)
            if pattern_data:
                detail = pattern_data.group(1)
                # 清理HTML标签和转义字符
                detail = detail.replace("&lt;spanclass=&quot;hl&quot;&gt;", "")
                detail = detail.replace("&lt;/span&gt;", "")
                detail = detail.replace("&amp;quot;", "\"")
                detail = detail.replace("&nbsp;", " ")
                return detail
            
        except Exception as e:
            logger.warning(f"获取详细原因失败 {stock_id}: {e}")
        
        return ""
    
    def _clean_and_structure_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """文本内容清洗和结构化处理"""
        if df.empty:
            return df
        
        # 清洗原因文本
        if "原因" in df.columns:
            df["原因"] = df["原因"].astype(str).apply(self._clean_text)
        
        # 清洗详细原因文本
        if "详细原因" in df.columns:
            df["详细原因"] = df["详细原因"].astype(str).apply(self._clean_text)
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """清洗文本内容"""
        if pd.isna(text) or text == "nan":
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。；：！？、（）【】""''%]', '', text)
        
        return text
    
    def _categorize_reasons(self, df: pd.DataFrame) -> pd.DataFrame:
        """涨停原因分类和标签化"""
        if df.empty or "原因" not in df.columns:
            return df
        
        categories = []
        tags = []
        
        for _, row in df.iterrows():
            reason_text = str(row.get("原因", "")) + " " + str(row.get("详细原因", ""))
            
            # 分类
            category = self._classify_reason(reason_text)
            categories.append(category)
            
            # 提取标签
            tag_list = self._extract_tags(reason_text)
            tags.append(", ".join(tag_list))
        
        df["原因分类"] = categories
        df["相关标签"] = tags
        
        return df
    
    def _classify_reason(self, text: str) -> str:
        """分类涨停原因"""
        text_lower = text.lower()
        
        for category, keywords in self.reason_categories.items():
            if category == "其他":
                continue
            
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        
        return "其他"
    
    def _extract_tags(self, text: str) -> List[str]:
        """提取相关标签"""
        tags = []
        
        # 常见概念标签
        concept_patterns = [
            r'(\w*概念)', r'(\w*题材)', r'(\w*板块)', r'(\w*行业)',
            r'(人工智能|AI)', r'(新能源|锂电)', r'(芯片|半导体)',
            r'(5G|通信)', r'(医药|生物)', r'(军工|国防)'
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text)
            tags.extend(matches)
        
        # 去重并限制数量
        unique_tags = list(set(tags))[:5]
        
        return unique_tags
    
    def get_reason_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取涨停原因统计信息"""
        if df.empty:
            return {}
        
        stats = {
            "总涨停数量": len(df),
            "分类统计": {},
            "平均涨跌幅": df["涨跌幅"].mean() if "涨跌幅" in df.columns else 0,
            "平均换手率": df["换手率"].mean() if "换手率" in df.columns else 0,
            "热门标签": []
        }
        
        # 分类统计
        if "原因分类" in df.columns:
            category_counts = df["原因分类"].value_counts().to_dict()
            stats["分类统计"] = category_counts
        
        # 热门标签统计
        if "相关标签" in df.columns:
            all_tags = []
            for tags_str in df["相关标签"]:
                if tags_str:
                    all_tags.extend([tag.strip() for tag in tags_str.split(",")])
            
            if all_tags:
                from collections import Counter
                tag_counts = Counter(all_tags)
                stats["热门标签"] = tag_counts.most_common(10)
        
        return stats
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        total_requests = self.error_stats['total_requests']
        if total_requests == 0:
            return self.error_stats
        
        return {
            **self.error_stats,
            'success_rate': self.error_stats['successful_requests'] / total_requests,
            'error_rate': (total_requests - self.error_stats['successful_requests']) / total_requests
        }
    
    def reset_error_statistics(self):
        """重置错误统计"""
        self.error_stats = {
            'network_errors': 0,
            'data_format_errors': 0,
            'api_errors': 0,
            'total_requests': 0,
            'successful_requests': 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试获取昨日涨停原因数据
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            request = LimitUpRequest(
                date=yesterday,
                include_detail=False  # 健康检查不获取详细信息
            )
            
            response = await self.get_limitup_reason_data(request)
            
            return {
                'status': 'healthy' if response.success else 'unhealthy',
                'response_time': response.response_time,
                'error_message': response.error_message,
                'data_available': not response.data.empty,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }


# 测试和使用示例
async def test_limitup_reason_adapter():
    """测试涨停原因适配器"""
    print("🔍 测试涨停原因数据适配器")
    print("=" * 50)
    
    adapter = LimitUpReasonAdapter()
    
    # 测试获取涨停原因数据
    print("1. 测试涨停原因数据...")
    today = datetime.now().strftime("%Y-%m-%d")
    request = LimitUpRequest(
        date=today,
        include_detail=False,  # 先不获取详细信息以加快测试
        symbols=None
    )
    
    response = await adapter.get_limitup_reason_data(request)
    print(f"   成功: {response.success}")
    print(f"   响应时间: {response.response_time:.2f}秒")
    print(f"   数据行数: {len(response.data)}")
    if not response.data.empty:
        print(f"   数据列: {list(response.data.columns)}")
        print(f"   数据预览:")
        print(response.data[['代码', '名称', '原因', '涨跌幅']].head())
    
    # 测试原因统计
    if not response.data.empty:
        print("\n2. 测试原因统计...")
        stats = adapter.get_reason_statistics(response.data)
        print(f"   总涨停数量: {stats.get('总涨停数量', 0)}")
        print(f"   平均涨跌幅: {stats.get('平均涨跌幅', 0):.2f}%")
        if stats.get('分类统计'):
            print(f"   分类统计: {stats['分类统计']}")
    
    # 健康检查
    print("\n3. 健康检查...")
    health = await adapter.health_check()
    print(f"   状态: {health['status']}")
    print(f"   响应时间: {health.get('response_time', 0):.2f}秒")
    
    # 错误统计
    print("\n4. 错误统计...")
    error_stats = adapter.get_error_statistics()
    print(f"   总请求数: {error_stats['total_requests']}")
    print(f"   成功率: {error_stats.get('success_rate', 0):.2%}")
    
    print("\n✅ 涨停原因适配器测试完成!")


if __name__ == "__main__":
    asyncio.run(test_limitup_reason_adapter())