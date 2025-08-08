#!/usr/bin/env python3
"""
实时行情分析端到端测试演示

演示完整的实时行情分析流程，包括数据获取、处理、存储、分析和API响应的端到端测试。
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.test_realtime_market_analysis_e2e import RealtimeMarketAnalysisE2ETest

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('realtime_e2e_test.log')
    ]
)
logger = logging.getLogger(__name__)


async def run_comprehensive_e2e_test():
    """运行综合端到端测试"""
    print("🚀 开始实时行情分析端到端测试演示")
    print("=" * 80)
    
    test_suite = RealtimeMarketAnalysisE2ETest()
    
    try:
        # 1. 设置测试环境
        print("\n📋 步骤 1: 设置测试环境")
        print("-" * 40)
        await test_suite.setup_test_environment()
        print("✅ 测试环境设置完成")
        
        # 2. 执行完整流程测试
        print("\n🔄 步骤 2: 执行完整分析流程测试")
        print("-" * 40)
        
        flow_results = await test_suite.test_complete_realtime_analysis_flow()
        
        # 显示性能指标
        print("\n📊 性能指标:")
        performance = flow_results.get('performance_metrics', {})
        for metric, data in performance.items():
            if isinstance(data, dict) and 'duration' in data:
                status = "✅" if data.get('meets_threshold', False) else "⚠️"
                print(f"  {status} {metric}: {data['duration']:.2f}秒")
        
        total_time = performance.get('total_duration', 0)
        sla_status = "✅" if performance.get('meets_sla', False) else "⚠️"
        print(f"  {sla_status} 总耗时: {total_time:.2f}秒")
        
        # 显示数据质量指标
        print("\n📈 数据质量指标:")
        quality = flow_results.get('data_quality_metrics', {})
        print(f"  完整性评分: {quality.get('completeness_score', 0):.2f}")
        print(f"  准确性评分: {quality.get('accuracy_score', 0):.2f}")
        print(f"  时效性评分: {quality.get('timeliness_score', 0):.2f}")
        print(f"  一致性评分: {quality.get('consistency_score', 0):.2f}")
        print(f"  总体评分: {quality.get('overall_score', 0):.2f}")
        
        # 显示错误信息
        if flow_results.get('error_details'):
            print("\n❌ 错误详情:")
            for error in flow_results['error_details']:
                print(f"  • {error}")
        
        # 3. 执行异常恢复测试
        print("\n🛡️ 步骤 3: 执行异常恢复测试")
        print("-" * 40)
        
        recovery_results = await test_suite.test_exception_recovery()
        
        print("异常恢复能力测试结果:")
        recovery_tests = [
            ('网络异常恢复', recovery_results.get('network_failure_recovery', False)),
            ('数据源异常恢复', recovery_results.get('data_source_failure_recovery', False)),
            ('数据库异常恢复', recovery_results.get('database_failure_recovery', False)),
            ('缓存异常恢复', recovery_results.get('cache_failure_recovery', False))
        ]
        
        for test_name, passed in recovery_tests:
            status = "✅" if passed else "❌"
            print(f"  {status} {test_name}")
        
        # 4. 生成测试报告
        print("\n📄 步骤 4: 生成测试报告")
        print("-" * 40)
        
        report = generate_test_report(flow_results, recovery_results)
        
        # 保存报告到文件
        report_filename = f"realtime_e2e_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ 测试报告已保存到: {report_filename}")
        
        # 5. 总结
        print("\n🎯 测试总结")
        print("-" * 40)
        
        overall_success = flow_results.get('success', False) and all(recovery_results.values())
        status_icon = "🎉" if overall_success else "⚠️"
        status_text = "全部通过" if overall_success else "部分失败"
        
        print(f"{status_icon} 端到端测试结果: {status_text}")
        print(f"📊 数据质量评分: {quality.get('overall_score', 0):.2f}/1.00")
        print(f"⏱️ 总执行时间: {total_time:.2f}秒")
        print(f"🔧 异常恢复能力: {sum(recovery_results.values())}/{len(recovery_results)} 项通过")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"端到端测试执行失败: {e}")
        print(f"❌ 测试执行失败: {e}")
        return False
        
    finally:
        # 清理测试环境
        print("\n🧹 清理测试环境...")
        await test_suite.teardown_test_environment()
        print("✅ 环境清理完成")


def generate_test_report(flow_results, recovery_results):
    """生成测试报告"""
    report = {
        'test_info': {
            'test_name': '实时行情分析端到端测试',
            'test_time': datetime.now().isoformat(),
            'test_version': '1.0.0'
        },
        'flow_test_results': flow_results,
        'recovery_test_results': recovery_results,
        'summary': {
            'overall_success': flow_results.get('success', False) and all(recovery_results.values()),
            'total_duration': flow_results.get('performance_metrics', {}).get('total_duration', 0),
            'data_quality_score': flow_results.get('data_quality_metrics', {}).get('overall_score', 0),
            'recovery_success_rate': sum(recovery_results.values()) / len(recovery_results) if recovery_results else 0
        },
        'recommendations': []
    }
    
    # 生成建议
    quality_score = report['summary']['data_quality_score']
    if quality_score < 0.8:
        report['recommendations'].append("数据质量评分较低，建议检查数据源和验证规则")
    
    total_time = report['summary']['total_duration']
    if total_time > 5.0:
        report['recommendations'].append("响应时间超过SLA要求，建议优化性能")
    
    recovery_rate = report['summary']['recovery_success_rate']
    if recovery_rate < 1.0:
        report['recommendations'].append("部分异常恢复测试失败，建议加强容错机制")
    
    return report


async def run_performance_benchmark():
    """运行性能基准测试"""
    print("\n🏃‍♂️ 性能基准测试")
    print("=" * 50)
    
    test_suite = RealtimeMarketAnalysisE2ETest()
    
    try:
        await test_suite.setup_test_environment()
        
        # 多次运行测试以获得平均性能
        iterations = 3
        total_times = []
        
        for i in range(iterations):
            print(f"\n第 {i+1}/{iterations} 次测试...")
            
            results = await test_suite.test_complete_realtime_analysis_flow()
            total_time = results.get('performance_metrics', {}).get('total_duration', 0)
            total_times.append(total_time)
            
            print(f"本次耗时: {total_time:.2f}秒")
        
        # 计算统计数据
        avg_time = sum(total_times) / len(total_times)
        min_time = min(total_times)
        max_time = max(total_times)
        
        print(f"\n📊 性能统计:")
        print(f"  平均耗时: {avg_time:.2f}秒")
        print(f"  最短耗时: {min_time:.2f}秒")
        print(f"  最长耗时: {max_time:.2f}秒")
        print(f"  性能稳定性: {'良好' if (max_time - min_time) < 1.0 else '一般'}")
        
    finally:
        await test_suite.teardown_test_environment()


async def run_stress_test():
    """运行压力测试"""
    print("\n💪 压力测试")
    print("=" * 50)
    
    test_suite = RealtimeMarketAnalysisE2ETest()
    
    try:
        await test_suite.setup_test_environment()
        
        # 增加测试股票数量进行压力测试
        original_symbols = test_suite.test_symbols
        stress_symbols = original_symbols * 3  # 增加到3倍
        test_suite.test_symbols = stress_symbols
        
        print(f"压力测试股票数量: {len(stress_symbols)}")
        
        start_time = asyncio.get_event_loop().time()
        results = await test_suite.test_complete_realtime_analysis_flow()
        end_time = asyncio.get_event_loop().time()
        
        total_time = end_time - start_time
        throughput = len(stress_symbols) / total_time if total_time > 0 else 0
        
        print(f"压力测试结果:")
        print(f"  处理股票数量: {len(stress_symbols)}")
        print(f"  总耗时: {total_time:.2f}秒")
        print(f"  处理吞吐量: {throughput:.2f} 股票/秒")
        print(f"  测试结果: {'通过' if results.get('success', False) else '失败'}")
        
        # 恢复原始设置
        test_suite.test_symbols = original_symbols
        
    finally:
        await test_suite.teardown_test_environment()


async def main():
    """主函数"""
    print("🎯 实时行情分析端到端测试演示系统")
    print("=" * 80)
    
    try:
        # 1. 综合端到端测试
        success = await run_comprehensive_e2e_test()
        
        if success:
            # 2. 性能基准测试
            await run_performance_benchmark()
            
            # 3. 压力测试
            await run_stress_test()
        
        print("\n" + "=" * 80)
        print("🏁 所有测试完成")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
    except Exception as e:
        logger.error(f"测试执行出现异常: {e}")
        print(f"❌ 测试执行出现异常: {e}")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())