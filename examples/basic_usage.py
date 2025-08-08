#!/usr/bin/env python3
"""
基础使用示例

本示例展示了如何使用爬虫接口集成系统的基本功能，
包括认证、数据获取、错误处理等。
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List


class StockDataClient:
    """股票数据客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.token = None
        self.session = requests.Session()
    
    def login(self, username: str, password: str) -> bool:
        """用户登录获取令牌"""
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={"username": username, "password": password}
            )
            response.raise_for_status()
            
            data = response.json()
            self.token = data["access_token"]
            
            # 设置默认请求头
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}"
            })
            
            print(f"✅ 登录成功，令牌有效期: {data['expires_in']}秒")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 登录失败: {e}")
            return False
    
    def get_realtime_data(self, symbol: str, fields: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """获取股票实时数据"""
        try:
            url = f"{self.base_url}/data/realtime/{symbol}"
            params = {}
            if fields:
                params["fields"] = ",".join(fields)
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data["success"]:
                return data["data"]
            else:
                print(f"❌ 获取数据失败: {data.get('error', 'Unknown error')}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
            return None
    
    def get_history_data(self, symbol: str, start_date: str, end_date: str, 
                        frequency: str = "daily") -> Optional[List[Dict[str, Any]]]:
        """获取历史数据"""
        try:
            url = f"{self.base_url}/data/history/{symbol}"
            params = {
                "start_date": start_date,
                "end_date": end_date,
                "frequency": frequency
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data["success"]:
                return data["data"]
            else:
                print(f"❌ 获取历史数据失败: {data.get('error', 'Unknown error')}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
            return None
    
    def get_dragon_tiger_data(self, date: str, symbol: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """获取龙虎榜数据"""
        try:
            url = f"{self.base_url}/data/dragon-tiger"
            params = {"date": date}
            if symbol:
                params["symbol"] = symbol
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data["success"]:
                return data["data"]
            else:
                print(f"❌ 获取龙虎榜数据失败: {data.get('error', 'Unknown error')}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
            return None
    
    def get_fund_flow_data(self, symbol: str, period: str = "1d") -> Optional[Dict[str, Any]]:
        """获取资金流向数据"""
        try:
            url = f"{self.base_url}/data/fund-flow/{symbol}"
            params = {"period": period}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data["success"]:
                return data["data"]
            else:
                print(f"❌ 获取资金流向数据失败: {data.get('error', 'Unknown error')}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求失败: {e}")
            return None
    
    def batch_get_realtime_data(self, symbols: List[str]) -> Dict[str, Any]:
        """批量获取实时数据"""
        try:
            url = f"{self.base_url}/data/batch"
            requests_data = [
                {"data_type": "stock_realtime", "symbol": symbol}
                for symbol in symbols
            ]
            
            payload = {
                "requests": requests_data,
                "options": {"parallel": True, "timeout": 30}
            }
            
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 批量请求失败: {e}")
            return {"success": False, "error": str(e)}
    
    def check_system_health(self) -> Dict[str, Any]:
        """检查系统健康状态"""
        try:
            # 健康检查不需要认证
            response = requests.get(f"{self.base_url.replace('/api/v1', '')}/health")
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 健康检查失败: {e}")
            return {"status": "unhealthy", "error": str(e)}


def demo_basic_usage():
    """基础使用演示"""
    print("🚀 股票数据客户端基础使用演示")
    print("=" * 50)
    
    # 创建客户端
    client = StockDataClient()
    
    # 1. 系统健康检查
    print("\n1️⃣ 系统健康检查")
    health = client.check_system_health()
    print(f"系统状态: {health.get('status', 'unknown')}")
    
    if health.get("status") != "healthy":
        print("⚠️ 系统状态异常，请检查服务是否正常运行")
        return
    
    # 2. 用户登录
    print("\n2️⃣ 用户登录")
    if not client.login("admin", "password"):
        print("❌ 登录失败，请检查用户名和密码")
        return
    
    # 3. 获取实时数据
    print("\n3️⃣ 获取实时数据")
    symbol = "000001.SZ"
    realtime_data = client.get_realtime_data(symbol)
    
    if realtime_data:
        print(f"股票: {realtime_data['symbol']} - {realtime_data['name']}")
        print(f"价格: {realtime_data['price']}")
        print(f"涨跌: {realtime_data['change']:+.2f} ({realtime_data['change_pct']:+.2f}%)")
        print(f"成交量: {realtime_data['volume']:,}")
        print(f"成交额: {realtime_data['amount']:,.0f}")
    
    # 4. 获取指定字段的实时数据
    print("\n4️⃣ 获取指定字段数据")
    fields_data = client.get_realtime_data(symbol, fields=["price", "volume", "change_pct"])
    if fields_data:
        print(f"精简数据: {fields_data}")
    
    # 5. 获取历史数据
    print("\n5️⃣ 获取历史数据")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    history_data = client.get_history_data(symbol, start_date, end_date)
    if history_data:
        print(f"获取到 {len(history_data)} 条历史数据")
        print("最近3天数据:")
        for item in history_data[-3:]:
            print(f"  {item['date']}: 开盘={item['open']}, 收盘={item['close']}, 成交量={item['volume']:,}")
    
    # 6. 获取资金流向数据
    print("\n6️⃣ 获取资金流向数据")
    fund_flow = client.get_fund_flow_data(symbol, period="1d")
    if fund_flow:
        main_flow = fund_flow['main_net_inflow']
        flow_direction = "流入" if main_flow > 0 else "流出"
        print(f"主力资金: {flow_direction} {abs(main_flow)/10000:.2f}万元 ({fund_flow['main_net_inflow_pct']:+.2f}%)")
        print(f"超大单: {fund_flow['super_large_net_inflow']/10000:.2f}万元")
        print(f"大单: {fund_flow['large_net_inflow']/10000:.2f}万元")
    
    # 7. 获取龙虎榜数据
    print("\n7️⃣ 获取龙虎榜数据")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    dragon_tiger = client.get_dragon_tiger_data(yesterday)
    
    if dragon_tiger:
        print(f"龙虎榜数据: {len(dragon_tiger)} 只股票上榜")
        for item in dragon_tiger[:3]:  # 显示前3只
            print(f"  {item['symbol']} {item['name']}: {item['reason']}")
            print(f"    净买入: {item['net_amount']/10000:.2f}万元")
    
    # 8. 批量获取数据
    print("\n8️⃣ 批量获取数据")
    symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH"]
    batch_result = client.batch_get_realtime_data(symbols)
    
    if batch_result.get("success"):
        print(f"批量请求完成: 总数={batch_result['summary']['total']}, "
              f"成功={batch_result['summary']['successful']}, "
              f"失败={batch_result['summary']['failed']}")
        
        print("批量结果:")
        for result in batch_result["results"]:
            if result["success"]:
                data = result["data"]
                print(f"  ✅ {data['symbol']}: {data['price']} ({data['change_pct']:+.2f}%)")
            else:
                print(f"  ❌ {result['symbol']}: {result['error']}")
    
    print("\n✅ 基础使用演示完成！")


def demo_error_handling():
    """错误处理演示"""
    print("\n🛡️ 错误处理演示")
    print("=" * 30)
    
    client = StockDataClient()
    
    # 1. 未登录访问
    print("\n1️⃣ 未登录访问测试")
    data = client.get_realtime_data("000001.SZ")
    if not data:
        print("✅ 正确处理了未认证错误")
    
    # 2. 错误的登录信息
    print("\n2️⃣ 错误登录信息测试")
    success = client.login("wrong_user", "wrong_pass")
    if not success:
        print("✅ 正确处理了登录错误")
    
    # 3. 登录后测试无效股票代码
    print("\n3️⃣ 无效股票代码测试")
    if client.login("admin", "password"):
        data = client.get_realtime_data("INVALID.CODE")
        if not data:
            print("✅ 正确处理了无效股票代码")


def demo_performance_tips():
    """性能优化技巧演示"""
    print("\n⚡ 性能优化技巧演示")
    print("=" * 30)
    
    client = StockDataClient()
    if not client.login("admin", "password"):
        return
    
    # 1. 批量请求 vs 单个请求
    print("\n1️⃣ 批量请求性能对比")
    symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH"]
    
    # 单个请求
    start_time = time.time()
    individual_results = []
    for symbol in symbols:
        data = client.get_realtime_data(symbol)
        if data:
            individual_results.append(data)
    individual_time = time.time() - start_time
    
    # 批量请求
    start_time = time.time()
    batch_result = client.batch_get_realtime_data(symbols)
    batch_time = time.time() - start_time
    
    print(f"单个请求耗时: {individual_time:.3f}秒")
    print(f"批量请求耗时: {batch_time:.3f}秒")
    print(f"性能提升: {individual_time/batch_time:.1f}倍")
    
    # 2. 字段选择优化
    print("\n2️⃣ 字段选择优化")
    symbol = "000001.SZ"
    
    # 获取所有字段
    start_time = time.time()
    full_data = client.get_realtime_data(symbol)
    full_time = time.time() - start_time
    
    # 只获取需要的字段
    start_time = time.time()
    partial_data = client.get_realtime_data(symbol, fields=["price", "change_pct"])
    partial_time = time.time() - start_time
    
    print(f"全字段请求: {full_time:.3f}秒, 数据大小: {len(str(full_data))} 字符")
    print(f"部分字段请求: {partial_time:.3f}秒, 数据大小: {len(str(partial_data))} 字符")


if __name__ == "__main__":
    try:
        # 基础使用演示
        demo_basic_usage()
        
        # 错误处理演示
        demo_error_handling()
        
        # 性能优化演示
        demo_performance_tips()
        
    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()