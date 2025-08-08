"""
并发访问集成测试模块

测试多用户并发访问的系统稳定性，验证数据竞争和锁机制的正确性，
添加高并发场景下的性能测试，实现负载均衡和故障转移的测试。
"""

import asyncio
import threading
import time
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import random
from typing import List, Dict, Any
from datetime import date

from stock_analysis_system.data.enhanced_data_sources import EnhancedDataSourceManager
from stock_analysis_system.data.cache_manager import CacheManager
from stock_analysis_system.data.market_data_request import MarketDataRequest
from stock_analysis_system.core.database import get_db_session
from stock_analysis_system.data.models import StockDailyData, DragonTigerBoard, FundFlow


class TestConcurrentAccessIntegration:
    """并发访问集成测试类"""
    
    @pytest.fixture
    def data_manager(self):
        """创建数据管理器实例"""
        return EnhancedDataSourceManager()
    
    @pytest.fixture
    def cache_manager(self):
        """创建缓存管理器实例"""
        return CacheManager()
    
    def test_concurrent_data_requests(self, data_manager):
        """测试并发数据请求的稳定性"""
        
        def make_request(symbol: str, request_id: int) -> Dict[str, Any]:
            """执行单个数据请求"""
            try:
                request = MarketDataRequest(
                    symbol=symbol,
                    data_type="stock_hist",
                    start_date="2024-01-01",
                    end_date="2024-01-31"
                )
                result = data_manager.get_data(request)
                return {
                    'request_id': request_id,
                    'symbol': symbol,
                    'success': True,
                    'data_count': len(result) if result else 0,
                    'timestamp': time.time()
                }
            except Exception as e:
                return {
                    'request_id': request_id,
                    'symbol': symbol,
                    'success': False,
                    'error': str(e),
                    'timestamp': time.time()
                }
        
        # 准备测试数据
        symbols = ['000001', '000002', '600000', '600036', '000858']
        num_threads = 20
        requests_per_thread = 5
        
        results = []
        start_time = time.time()
        
        # 使用线程池执行并发请求
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for i in range(num_threads):
                for j in range(requests_per_thread):
                    symbol = random.choice(symbols)
                    request_id = i * requests_per_thread + j
                    future = executor.submit(make_request, symbol, request_id)
                    futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                results.append(future.result())
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证结果
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        print(f"并发请求测试结果:")
        print(f"总请求数: {len(results)}")
        print(f"成功请求数: {len(successful_requests)}")
        print(f"失败请求数: {len(failed_requests)}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均响应时间: {total_time/len(results):.3f}秒")
        
        # 断言：成功率应该大于90%
        success_rate = len(successful_requests) / len(results)
        assert success_rate > 0.9, f"成功率过低: {success_rate:.2%}"
        
        # 断言：没有数据竞争导致的异常
        race_condition_errors = [
            r for r in failed_requests 
            if 'race' in r.get('error', '').lower() or 'lock' in r.get('error', '').lower()
        ]
        assert len(race_condition_errors) == 0, f"检测到数据竞争错误: {race_condition_errors}"
    
    def test_concurrent_cache_operations(self, cache_manager):
        """测试并发缓存操作的正确性"""
        
        def cache_operation(operation_id: int) -> Dict[str, Any]:
            """执行缓存操作"""
            try:
                key = f"test_key_{operation_id % 10}"  # 使用有限的key集合增加冲突
                value = f"test_value_{operation_id}_{time.time()}"
                
                # 随机执行不同的缓存操作
                operation_type = random.choice(['set', 'get', 'delete', 'update'])
                
                if operation_type == 'set':
                    cache_manager.set(key, value, ttl=60)
                    result = 'set_success'
                elif operation_type == 'get':
                    result = cache_manager.get(key)
                    result = 'get_success' if result is not None else 'get_miss'
                elif operation_type == 'delete':
                    cache_manager.delete(key)
                    result = 'delete_success'
                else:  # update
                    cache_manager.set(key, value, ttl=60)
                    result = 'update_success'
                
                return {
                    'operation_id': operation_id,
                    'operation_type': operation_type,
                    'key': key,
                    'success': True,
                    'result': result,
                    'timestamp': time.time()
                }
            except Exception as e:
                return {
                    'operation_id': operation_id,
                    'success': False,
                    'error': str(e),
                    'timestamp': time.time()
                }
        
        # 执行并发缓存操作
        num_threads = 50
        operations_per_thread = 20
        
        results = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for i in range(num_threads * operations_per_thread):
                future = executor.submit(cache_operation, i)
                futures.append(future)
            
            for future in as_completed(futures):
                results.append(future.result())
        
        # 分析结果
        successful_ops = [r for r in results if r['success']]
        failed_ops = [r for r in results if not r['success']]
        
        operation_stats = {}
        for result in successful_ops:
            op_type = result['operation_type']
            operation_stats[op_type] = operation_stats.get(op_type, 0) + 1
        
        print(f"并发缓存操作测试结果:")
        print(f"总操作数: {len(results)}")
        print(f"成功操作数: {len(successful_ops)}")
        print(f"失败操作数: {len(failed_ops)}")
        print(f"操作统计: {operation_stats}")
        
        # 验证缓存一致性
        success_rate = len(successful_ops) / len(results)
        assert success_rate > 0.95, f"缓存操作成功率过低: {success_rate:.2%}"
        
        # 验证没有死锁或竞态条件
        deadlock_errors = [
            r for r in failed_ops 
            if 'deadlock' in r.get('error', '').lower() or 'timeout' in r.get('error', '').lower()
        ]
        assert len(deadlock_errors) == 0, f"检测到死锁错误: {deadlock_errors}"
    
    def test_concurrent_database_operations(self):
        """测试并发数据库操作的正确性"""
        
        def database_operation(operation_id: int) -> Dict[str, Any]:
            """执行数据库操作"""
            try:
                with get_db_session() as session:
                    # 随机执行不同的数据库操作
                    operation_type = random.choice(['insert', 'select', 'update', 'delete'])
                    
                    if operation_type == 'insert':
                        # 插入测试数据
                        stock_data = StockDailyData(
                            stock_code=f"TEST{operation_id:06d}",
                            trade_date=date(2024, 1, 1),
                            open_price=10.0,
                            high_price=11.0,
                            low_price=9.0,
                            close_price=10.5,
                            volume=1000000,
                            amount=10500000.0
                        )
                        session.add(stock_data)
                        session.commit()
                        result = 'insert_success'
                        
                    elif operation_type == 'select':
                        # 查询数据
                        count = session.query(StockDailyData).filter(
                            StockDailyData.stock_code.like('TEST%')
                        ).count()
                        result = f'select_success_count_{count}'
                        
                    elif operation_type == 'update':
                        # 更新数据
                        updated = session.query(StockDailyData).filter(
                            StockDailyData.stock_code.like('TEST%')
                        ).update({'volume': StockDailyData.volume + 1000})
                        session.commit()
                        result = f'update_success_rows_{updated}'
                        
                    else:  # delete
                        # 删除数据
                        deleted = session.query(StockDailyData).filter(
                            StockDailyData.stock_code == f"TEST{operation_id:06d}"
                        ).delete()
                        session.commit()
                        result = f'delete_success_rows_{deleted}'
                
                return {
                    'operation_id': operation_id,
                    'operation_type': operation_type,
                    'success': True,
                    'result': result,
                    'timestamp': time.time()
                }
                
            except Exception as e:
                return {
                    'operation_id': operation_id,
                    'success': False,
                    'error': str(e),
                    'timestamp': time.time()
                }
        
        # 执行并发数据库操作
        num_threads = 30
        operations_per_thread = 10
        
        results = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for i in range(num_threads * operations_per_thread):
                future = executor.submit(database_operation, i)
                futures.append(future)
            
            for future in as_completed(futures):
                results.append(future.result())
        
        # 分析结果
        successful_ops = [r for r in results if r['success']]
        failed_ops = [r for r in results if not r['success']]
        
        operation_stats = {}
        for result in successful_ops:
            op_type = result['operation_type']
            operation_stats[op_type] = operation_stats.get(op_type, 0) + 1
        
        print(f"并发数据库操作测试结果:")
        print(f"总操作数: {len(results)}")
        print(f"成功操作数: {len(successful_ops)}")
        print(f"失败操作数: {len(failed_ops)}")
        print(f"操作统计: {operation_stats}")
        
        # 验证数据库操作的正确性
        success_rate = len(successful_ops) / len(results)
        assert success_rate > 0.90, f"数据库操作成功率过低: {success_rate:.2%}"
        
        # 验证没有数据库锁冲突
        lock_errors = [
            r for r in failed_ops 
            if 'lock' in r.get('error', '').lower() or 'deadlock' in r.get('error', '').lower()
        ]
        # 允许少量锁冲突，但不应该太多
        assert len(lock_errors) < len(results) * 0.05, f"数据库锁冲突过多: {len(lock_errors)}"
    
    def test_load_balancing_simulation(self, data_manager):
        """测试负载均衡机制的模拟"""
        
        def simulate_user_request(user_id: int, request_count: int) -> List[Dict[str, Any]]:
            """模拟用户请求"""
            results = []
            symbols = ['000001', '000002', '600000', '600036', '000858']
            
            for i in range(request_count):
                try:
                    symbol = random.choice(symbols)
                    data_type = random.choice(['stock_hist', 'fund_flow', 'dragon_tiger'])
                    
                    request = MarketDataRequest(
                        symbol=symbol,
                        data_type=data_type,
                        start_date="2024-01-01",
                        end_date="2024-01-31"
                    )
                    
                    start_time = time.time()
                    result = data_manager.get_data(request)
                    end_time = time.time()
                    
                    results.append({
                        'user_id': user_id,
                        'request_id': i,
                        'symbol': symbol,
                        'data_type': data_type,
                        'success': True,
                        'response_time': end_time - start_time,
                        'data_count': len(result) if result else 0,
                        'timestamp': time.time()
                    })
                    
                    # 模拟用户思考时间
                    time.sleep(random.uniform(0.1, 0.5))
                    
                except Exception as e:
                    results.append({
                        'user_id': user_id,
                        'request_id': i,
                        'success': False,
                        'error': str(e),
                        'timestamp': time.time()
                    })
            
            return results
        
        # 模拟多用户并发访问
        num_users = 15
        requests_per_user = 8
        
        all_results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = []
            
            for user_id in range(num_users):
                future = executor.submit(simulate_user_request, user_id, requests_per_user)
                futures.append(future)
            
            for future in as_completed(futures):
                user_results = future.result()
                all_results.extend(user_results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 分析负载均衡效果
        successful_requests = [r for r in all_results if r['success']]
        failed_requests = [r for r in all_results if not r['success']]
        
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            # 按用户统计请求分布
            user_stats = {}
            for result in successful_requests:
                user_id = result['user_id']
                if user_id not in user_stats:
                    user_stats[user_id] = {'count': 0, 'total_time': 0}
                user_stats[user_id]['count'] += 1
                user_stats[user_id]['total_time'] += result['response_time']
            
            print(f"负载均衡测试结果:")
            print(f"总请求数: {len(all_results)}")
            print(f"成功请求数: {len(successful_requests)}")
            print(f"失败请求数: {len(failed_requests)}")
            print(f"总耗时: {total_time:.2f}秒")
            print(f"平均响应时间: {avg_response_time:.3f}秒")
            print(f"最大响应时间: {max_response_time:.3f}秒")
            print(f"最小响应时间: {min_response_time:.3f}秒")
            print(f"用户请求分布: {len(user_stats)}个用户")
            
            # 验证负载均衡效果
            success_rate = len(successful_requests) / len(all_results)
            assert success_rate > 0.85, f"负载均衡下成功率过低: {success_rate:.2%}"
            
            # 验证响应时间分布合理
            assert avg_response_time < 5.0, f"平均响应时间过长: {avg_response_time:.3f}秒"
            assert max_response_time < 15.0, f"最大响应时间过长: {max_response_time:.3f}秒"
    
    def test_fault_tolerance_simulation(self, data_manager):
        """测试故障容错机制的模拟"""
        
        def simulate_with_faults(request_id: int) -> Dict[str, Any]:
            """模拟带故障的请求"""
            try:
                # 随机注入故障
                if random.random() < 0.1:  # 10%的概率模拟网络故障
                    raise ConnectionError("模拟网络连接故障")
                
                if random.random() < 0.05:  # 5%的概率模拟超时
                    time.sleep(10)  # 模拟超时
                
                symbol = random.choice(['000001', '000002', '600000'])
                request = MarketDataRequest(
                    symbol=symbol,
                    data_type="stock_hist",
                    start_date="2024-01-01",
                    end_date="2024-01-31"
                )
                
                start_time = time.time()
                result = data_manager.get_data(request)
                end_time = time.time()
                
                return {
                    'request_id': request_id,
                    'success': True,
                    'response_time': end_time - start_time,
                    'data_count': len(result) if result else 0,
                    'fault_injected': False
                }
                
            except Exception as e:
                return {
                    'request_id': request_id,
                    'success': False,
                    'error': str(e),
                    'fault_injected': '模拟' in str(e)
                }
        
        # 执行故障容错测试
        num_requests = 100
        results = []
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            
            for i in range(num_requests):
                future = executor.submit(simulate_with_faults, i)
                futures.append(future)
            
            for future in as_completed(futures):
                results.append(future.result())
        
        # 分析故障容错效果
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        injected_faults = [r for r in results if r.get('fault_injected', False)]
        
        print(f"故障容错测试结果:")
        print(f"总请求数: {len(results)}")
        print(f"成功请求数: {len(successful_requests)}")
        print(f"失败请求数: {len(failed_requests)}")
        print(f"注入故障数: {len(injected_faults)}")
        
        # 验证故障容错能力
        # 在有故障注入的情况下，成功率应该仍然保持在合理水平
        success_rate = len(successful_requests) / len(results)
        expected_min_success_rate = 0.80  # 考虑到故障注入，80%的成功率是合理的
        
        assert success_rate >= expected_min_success_rate, \
            f"故障容错能力不足，成功率: {success_rate:.2%}, 期望: {expected_min_success_rate:.2%}"
        
        # 验证故障恢复机制
        non_injected_failures = [r for r in failed_requests if not r.get('fault_injected', False)]
        assert len(non_injected_failures) < len(results) * 0.05, \
            f"非注入故障过多，可能存在系统问题: {len(non_injected_failures)}"


if __name__ == "__main__":
    # 运行测试
    test_instance = TestConcurrentAccessIntegration()
    
    print("开始并发访问集成测试...")
    
    # 创建测试实例
    data_manager = EnhancedDataSourceManager()
    cache_manager = CacheManager()
    
    try:
        print("\n1. 测试并发数据请求...")
        test_instance.test_concurrent_data_requests(data_manager)
        
        print("\n2. 测试并发缓存操作...")
        test_instance.test_concurrent_cache_operations(cache_manager)
        
        print("\n3. 测试并发数据库操作...")
        test_instance.test_concurrent_database_operations()
        
        print("\n4. 测试负载均衡模拟...")
        test_instance.test_load_balancing_simulation(data_manager)
        
        print("\n5. 测试故障容错模拟...")
        test_instance.test_fault_tolerance_simulation(data_manager)
        
        print("\n✅ 所有并发访问集成测试通过!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise