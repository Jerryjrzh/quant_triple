#!/usr/bin/env python3
"""
生产环境验收测试

在生产环境执行完整的验收测试，验证系统性能和稳定性指标。
"""

import asyncio
import aiohttp
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionAcceptanceTest:
    """生产环境验收测试"""
    
    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url
        self.auth_token = auth_token
        self.results = []
        self.performance_data = []
    
    async def run_acceptance_tests(self):
        """运行验收测试"""
        logger.info("🚀 开始生产环境验收测试")
        
        async with aiohttp.ClientSession() as session:
            # 功能测试
            await self._test_api_functionality(session)
            
            # 性能测试
            await self._test_performance(session)
            
            # 稳定性测试
            await self._test_stability(session)
            
            # 安全测试
            await self._test_security(session)
        
        # 生成报告
        self._generate_acceptance_report()
    
    async def _test_api_functionality(self, session):
        """测试API功能"""
        logger.info("🔧 测试API功能")
        
        test_cases = [
            {
                'name': '健康检查',
                'method': 'GET',
                'url': f'{self.base_url}/health',
                'expected_status': 200
            },
            {
                'name': 'API文档访问',
                'method': 'GET',
                'url': f'{self.base_url}/docs',
                'expected_status': 200
            },
            {
                'name': '股票基础信息查询',
                'method': 'GET',
                'url': f'{self.base_url}/api/v1/stocks/000001.SZ/info',
                'expected_status': 200
            },
            {
                'name': '春节效应分析',
                'method': 'GET',
                'url': f'{self.base_url}/api/v1/analysis/spring-festival/000001.SZ',
                'expected_status': 200
            },
            {
                'name': '数据质量检查',
                'method': 'GET',
                'url': f'{self.base_url}/api/v1/data/quality/report',
                'expected_status': 200
            },
            {
                'name': '系统监控指标',
                'method': 'GET',
                'url': f'{self.base_url}/api/v1/monitoring/metrics',
                'expected_status': 200
            }
        ]
        
        for test_case in test_cases:
            try:
                start_time = time.time()
                
                kwargs = {
                    'url': test_case['url'],
                    'headers': test_case.get('headers', {})
                }
                
                if test_case.get('data'):
                    kwargs['json'] = test_case['data']
                
                async with session.request(test_case['method'], **kwargs) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == test_case['expected_status']:
                        self._record_result(test_case['name'], True, f"响应时间: {response_time:.3f}s")
                    else:
                        self._record_result(test_case['name'], False, f"状态码: {response.status}")
                        
            except Exception as e:
                self._record_result(test_case['name'], False, str(e))
    
    async def _test_performance(self, session):
        """测试性能"""
        logger.info("⚡ 测试性能")
        
        # 并发测试
        concurrent_users = 50
        test_duration = 30  # 秒
        
        async def make_request():
            try:
                start_time = time.time()
                async with session.get(
                    f'{self.base_url}/api/v1/data/realtime/000001.SZ',
                    headers={'Authorization': f'Bearer {self.auth_token}'}
                ) as response:
                    response_time = time.time() - start_time
                    return {
                        'status': response.status,
                        'response_time': response_time,
                        'success': response.status == 200
                    }
            except Exception as e:
                return {
                    'status': 0,
                    'response_time': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # 执行并发测试
        start_time = time.time()
        tasks = []
        
        while time.time() - start_time < test_duration:
            # 创建并发任务
            batch_tasks = [make_request() for _ in range(concurrent_users)]
            batch_results = await asyncio.gather(*batch_tasks)
            
            self.performance_data.extend(batch_results)
            
            # 短暂休息
            await asyncio.sleep(1)
        
        # 分析性能数据
        successful_requests = [r for r in self.performance_data if r['success']]
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            success_rate = len(successful_requests) / len(self.performance_data) * 100
            
            # 性能指标验证
            if avg_response_time < 1.0:
                self._record_result("性能-平均响应时间", True, f"{avg_response_time:.3f}s")
            else:
                self._record_result("性能-平均响应时间", False, f"{avg_response_time:.3f}s")
            
            if p95_response_time < 2.0:
                self._record_result("性能-P95响应时间", True, f"{p95_response_time:.3f}s")
            else:
                self._record_result("性能-P95响应时间", False, f"{p95_response_time:.3f}s")
            
            if success_rate >= 99.0:
                self._record_result("性能-成功率", True, f"{success_rate:.1f}%")
            else:
                self._record_result("性能-成功率", False, f"{success_rate:.1f}%")
    
    async def _test_stability(self, session):
        """测试稳定性"""
        logger.info("🛡️ 测试稳定性")
        
        # 长时间运行测试
        test_duration = 300  # 5分钟
        request_interval = 5  # 每5秒一次请求
        
        start_time = time.time()
        stability_results = []
        
        while time.time() - start_time < test_duration:
            try:
                async with session.get(
                    f'{self.base_url}/health'
                ) as response:
                    stability_results.append({
                        'timestamp': time.time(),
                        'status': response.status,
                        'success': response.status == 200
                    })
                    
            except Exception as e:
                stability_results.append({
                    'timestamp': time.time(),
                    'status': 0,
                    'success': False,
                    'error': str(e)
                })
            
            await asyncio.sleep(request_interval)
        
        # 分析稳定性
        successful_checks = sum(1 for r in stability_results if r['success'])
        stability_rate = successful_checks / len(stability_results) * 100
        
        if stability_rate >= 99.5:
            self._record_result("稳定性-可用性", True, f"{stability_rate:.1f}%")
        else:
            self._record_result("稳定性-可用性", False, f"{stability_rate:.1f}%")
    
    async def _test_security(self, session):
        """测试安全性"""
        logger.info("🔒 测试安全性")
        
        # 测试未授权访问
        try:
            async with session.get(f'{self.base_url}/api/v1/data/realtime/000001.SZ') as response:
                if response.status == 401:
                    self._record_result("安全-未授权访问拒绝", True)
                else:
                    self._record_result("安全-未授权访问拒绝", False, f"状态码: {response.status}")
        except Exception as e:
            self._record_result("安全-未授权访问拒绝", False, str(e))
        
        # 测试HTTPS
        if self.base_url.startswith('https'):
            self._record_result("安全-HTTPS启用", True)
        else:
            self._record_result("安全-HTTPS启用", False, "使用HTTP协议")
    
    def _record_result(self, test_name: str, passed: bool, details: str = ""):
        """记录测试结果"""
        result = {
            'name': test_name,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(result)
        
        status = "✅" if passed else "❌"
        logger.info(f"{status} {test_name}: {details}")
    
    def _generate_acceptance_report(self):
        """生成验收报告"""
        passed = sum(1 for r in self.results if r['passed'])
        total = len(self.results)
        success_rate = (passed / total * 100) if total > 0 else 0
        
        report = {
            'test_summary': {
                'total_tests': total,
                'passed_tests': passed,
                'failed_tests': total - passed,
                'success_rate': success_rate
            },
            'test_results': self.results,
            'performance_summary': self._get_performance_summary(),
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存报告
        report_file = f"production_acceptance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 验收报告已保存: {report_file}")
        
        # 打印摘要
        print(f"\n生产环境验收测试完成: {success_rate:.1f}% ({passed}/{total})")
        
        if success_rate >= 95:
            print("🎉 验收测试通过！系统可以上线")
            return True
        else:
            print("⚠️ 验收测试未通过，请修复问题后重新测试")
            return False
    
    def _get_performance_summary(self):
        """获取性能摘要"""
        if not self.performance_data:
            return {}
        
        successful_requests = [r for r in self.performance_data if r['success']]
        if not successful_requests:
            return {'error': 'No successful requests'}
        
        response_times = [r['response_time'] for r in successful_requests]
        
        return {
            'total_requests': len(self.performance_data),
            'successful_requests': len(successful_requests),
            'success_rate': len(successful_requests) / len(self.performance_data) * 100,
            'avg_response_time': statistics.mean(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'p95_response_time': statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times)
        }

async def main():
    # 配置测试参数 - 使用本地开发环境进行模拟测试
    base_url = "http://localhost:8000"  # 本地开发环境URL
    auth_token = "test_token_123"  # 测试认证令牌
    
    # 检查是否提供了生产环境参数
    import sys
    if len(sys.argv) > 2:
        base_url = sys.argv[1]
        auth_token = sys.argv[2]
        print(f"使用提供的生产环境参数: {base_url}")
    else:
        print("使用本地开发环境进行验收测试模拟")
    
    tester = ProductionAcceptanceTest(base_url, auth_token)
    success = await tester.run_acceptance_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(asyncio.run(main()))