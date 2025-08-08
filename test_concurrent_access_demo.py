"""
并发访问集成测试演示

测试多用户并发访问的系统稳定性，验证数据竞争和锁机制的正确性。
"""

import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from datetime import date

from stock_analysis_system.core.database import SessionLocal
from stock_analysis_system.data.models import StockDailyData


class SimpleCacheManager:
    """简单的缓存管理器用于测试"""
    
    def __init__(self):
        self.cache = {}
        self.lock = threading.RLock()
    
    def set(self, key: str, value: Any, ttl: int = 60):
        """设置缓存"""
        with self.lock:
            self.cache[key] = {
                'value': value,
                'expire_time': time.time() + ttl
            }
    
    def get(self, key: str):
        """获取缓存"""
        with self.lock:
            if key not in self.cache:
                return None
            
            item = self.cache[key]
            if time.time() > item['expire_time']:
                del self.cache[key]
                return None
            
            return item['value']
    
    def delete(self, key: str):
        """删除缓存"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()


class ConcurrentAccessTester:
    """并发访问测试器"""
    
    def __init__(self):
        self.cache_manager = SimpleCacheManager()
        self.results = []
    
    def test_concurrent_cache_operations(self, num_threads=20, operations_per_thread=10):
        """测试并发缓存操作"""
        print(f"开始并发缓存操作测试: {num_threads}线程 x {operations_per_thread}操作")
        
        def cache_operation(operation_id: int) -> Dict[str, Any]:
            """执行缓存操作"""
            try:
                key = f"test_key_{operation_id % 5}"  # 使用有限的key集合增加冲突
                value = f"test_value_{operation_id}_{time.time()}"
                
                # 随机执行不同的缓存操作
                operation_type = random.choice(['set', 'get', 'delete'])
                
                if operation_type == 'set':
                    self.cache_manager.set(key, value, ttl=60)
                    result = 'set_success'
                elif operation_type == 'get':
                    result = self.cache_manager.get(key)
                    result = 'get_success' if result is not None else 'get_miss'
                else:  # delete
                    self.cache_manager.delete(key)
                    result = 'delete_success'
                
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
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for i in range(num_threads * operations_per_thread):
                future = executor.submit(cache_operation, i)
                futures.append(future)
            
            for future in as_completed(futures):
                results.append(future.result())
        
        end_time = time.time()
        
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
        print(f"总耗时: {end_time - start_time:.2f}秒")
        print(f"操作统计: {operation_stats}")
        
        if failed_ops:
            print(f"失败操作示例: {failed_ops[:3]}")
        
        # 验证结果
        success_rate = len(successful_ops) / len(results)
        print(f"成功率: {success_rate:.2%}")
        
        return {
            'total_operations': len(results),
            'successful_operations': len(successful_ops),
            'failed_operations': len(failed_ops),
            'success_rate': success_rate,
            'operation_stats': operation_stats,
            'total_time': end_time - start_time
        }
    
    def test_concurrent_database_operations(self, num_threads=15, operations_per_thread=5):
        """测试并发数据库操作"""
        print(f"开始并发数据库操作测试: {num_threads}线程 x {operations_per_thread}操作")
        
        def database_operation(operation_id: int) -> Dict[str, Any]:
            """执行数据库操作"""
            try:
                session = SessionLocal()
                try:
                    # 随机执行不同的数据库操作
                    operation_type = random.choice(['insert', 'select', 'update'])
                    
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
                        
                    else:  # update
                        # 更新数据
                        updated = session.query(StockDailyData).filter(
                            StockDailyData.stock_code.like('TEST%')
                        ).update({'volume': StockDailyData.volume + 1000})
                        session.commit()
                        result = f'update_success_rows_{updated}'
                
                    return {
                        'operation_id': operation_id,
                        'operation_type': operation_type,
                        'success': True,
                        'result': result,
                        'timestamp': time.time()
                    }
                finally:
                    session.close()
                
            except Exception as e:
                return {
                    'operation_id': operation_id,
                    'success': False,
                    'error': str(e),
                    'timestamp': time.time()
                }
        
        # 执行并发数据库操作
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for i in range(num_threads * operations_per_thread):
                future = executor.submit(database_operation, i)
                futures.append(future)
            
            for future in as_completed(futures):
                results.append(future.result())
        
        end_time = time.time()
        
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
        print(f"总耗时: {end_time - start_time:.2f}秒")
        print(f"操作统计: {operation_stats}")
        
        if failed_ops:
            print(f"失败操作示例: {failed_ops[:3]}")
        
        # 验证结果
        success_rate = len(successful_ops) / len(results)
        print(f"成功率: {success_rate:.2%}")
        
        return {
            'total_operations': len(results),
            'successful_operations': len(successful_ops),
            'failed_operations': len(failed_ops),
            'success_rate': success_rate,
            'operation_stats': operation_stats,
            'total_time': end_time - start_time
        }
    
    def test_mixed_concurrent_operations(self, num_threads=25):
        """测试混合并发操作"""
        print(f"开始混合并发操作测试: {num_threads}线程")
        
        def mixed_operation(operation_id: int) -> Dict[str, Any]:
            """执行混合操作"""
            try:
                # 随机选择操作类型
                operation_type = random.choice(['cache', 'database'])
                
                if operation_type == 'cache':
                    key = f"mixed_key_{operation_id % 10}"
                    value = f"mixed_value_{operation_id}"
                    self.cache_manager.set(key, value, ttl=30)
                    retrieved = self.cache_manager.get(key)
                    result = 'cache_success' if retrieved == value else 'cache_mismatch'
                    
                else:  # database
                    session = SessionLocal()
                    try:
                        # 简单的查询操作
                        count = session.query(StockDailyData).filter(
                            StockDailyData.stock_code.like('TEST%')
                        ).count()
                        result = f'db_query_count_{count}'
                    finally:
                        session.close()
                
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
        
        # 执行混合并发操作
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for i in range(num_threads * 3):  # 每个线程执行3个操作
                future = executor.submit(mixed_operation, i)
                futures.append(future)
            
            for future in as_completed(futures):
                results.append(future.result())
        
        end_time = time.time()
        
        # 分析结果
        successful_ops = [r for r in results if r['success']]
        failed_ops = [r for r in results if not r['success']]
        
        operation_stats = {}
        for result in successful_ops:
            op_type = result['operation_type']
            operation_stats[op_type] = operation_stats.get(op_type, 0) + 1
        
        print(f"混合并发操作测试结果:")
        print(f"总操作数: {len(results)}")
        print(f"成功操作数: {len(successful_ops)}")
        print(f"失败操作数: {len(failed_ops)}")
        print(f"总耗时: {end_time - start_time:.2f}秒")
        print(f"操作统计: {operation_stats}")
        
        if failed_ops:
            print(f"失败操作示例: {failed_ops[:3]}")
        
        # 验证结果
        success_rate = len(successful_ops) / len(results)
        print(f"成功率: {success_rate:.2%}")
        
        return {
            'total_operations': len(results),
            'successful_operations': len(successful_ops),
            'failed_operations': len(failed_ops),
            'success_rate': success_rate,
            'operation_stats': operation_stats,
            'total_time': end_time - start_time
        }
    
    def test_stress_scenario(self):
        """压力测试场景"""
        print("开始压力测试场景...")
        
        # 高并发缓存操作
        cache_result = self.test_concurrent_cache_operations(num_threads=50, operations_per_thread=20)
        
        # 中等并发数据库操作
        db_result = self.test_concurrent_database_operations(num_threads=20, operations_per_thread=10)
        
        # 混合操作
        mixed_result = self.test_mixed_concurrent_operations(num_threads=30)
        
        print("\n=== 压力测试总结 ===")
        print(f"缓存操作成功率: {cache_result['success_rate']:.2%}")
        print(f"数据库操作成功率: {db_result['success_rate']:.2%}")
        print(f"混合操作成功率: {mixed_result['success_rate']:.2%}")
        
        total_operations = (cache_result['total_operations'] + 
                          db_result['total_operations'] + 
                          mixed_result['total_operations'])
        total_successful = (cache_result['successful_operations'] + 
                          db_result['successful_operations'] + 
                          mixed_result['successful_operations'])
        
        overall_success_rate = total_successful / total_operations
        print(f"总体成功率: {overall_success_rate:.2%}")
        
        return {
            'cache_result': cache_result,
            'db_result': db_result,
            'mixed_result': mixed_result,
            'overall_success_rate': overall_success_rate,
            'total_operations': total_operations
        }


def main():
    """主函数"""
    print("=== 并发访问集成测试演示 ===\n")
    
    tester = ConcurrentAccessTester()
    
    try:
        # 1. 基础并发缓存测试
        print("1. 基础并发缓存测试")
        print("-" * 50)
        cache_result = tester.test_concurrent_cache_operations(num_threads=10, operations_per_thread=5)
        print()
        
        # 2. 基础并发数据库测试
        print("2. 基础并发数据库测试")
        print("-" * 50)
        db_result = tester.test_concurrent_database_operations(num_threads=8, operations_per_thread=3)
        print()
        
        # 3. 混合并发操作测试
        print("3. 混合并发操作测试")
        print("-" * 50)
        mixed_result = tester.test_mixed_concurrent_operations(num_threads=15)
        print()
        
        # 4. 压力测试
        print("4. 压力测试场景")
        print("-" * 50)
        stress_result = tester.test_stress_scenario()
        
        print("\n=== 测试完成 ===")
        print("✅ 所有并发访问测试完成!")
        
        # 清理测试数据
        print("\n清理测试数据...")
        try:
            session = SessionLocal()
            try:
                deleted = session.query(StockDailyData).filter(
                    StockDailyData.stock_code.like('TEST%')
                ).delete()
                session.commit()
                print(f"清理了 {deleted} 条测试数据")
            finally:
                session.close()
        except Exception as e:
            print(f"清理数据时出错: {e}")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()