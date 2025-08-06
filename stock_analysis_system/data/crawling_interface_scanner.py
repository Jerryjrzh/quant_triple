"""
爬虫接口扫描器

该模块负责自动扫描 tmp/core/crawling/ 目录中的所有爬虫接口，
提取接口元数据，并生成接口兼容性报告。

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import ast
import inspect
import importlib.util
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class FunctionSignature:
    """函数签名信息"""
    name: str
    parameters: List[Dict[str, Any]]
    return_type: str
    docstring: Optional[str]
    source_file: str
    line_number: int


@dataclass
class InterfaceMetadata:
    """接口元数据"""
    module_name: str
    file_path: str
    functions: List[FunctionSignature]
    data_types: List[str]
    update_frequency: str
    reliability_score: float
    last_modified: datetime
    dependencies: List[str]
    api_endpoints: List[str]
    data_sources: List[str]


class CrawlingInterfaceScanner:
    """爬虫接口扫描器"""
    
    def __init__(self, crawling_dir: str = "tmp/core/crawling"):
        """
        初始化扫描器
        
        Args:
            crawling_dir: 爬虫接口目录路径
        """
        self.crawling_dir = crawling_dir
        self.interfaces_metadata = {}
        self.compatibility_issues = []
        
        # 数据类型映射
        self.data_type_keywords = {
            'stock': ['股票', 'stock', 'equity', '个股', 'zh_a'],
            'fund': ['基金', 'fund', 'etf', '资金'],
            'dragon_tiger': ['龙虎榜', 'lhb', 'billboard', 'dragon', 'tiger'],
            'limitup': ['涨停', 'limitup', 'limit_up', 'harden'],
            'chip': ['筹码', 'chip', 'race'],
            'realtime': ['实时', 'spot', 'realtime', 'real_time'],
            'history': ['历史', 'hist', 'history', 'historical'],
            'financial': ['财务', 'financial', 'finance']
        }
        
        # 更新频率关键词
        self.frequency_keywords = {
            'realtime': ['实时', 'spot', 'realtime', 'current'],
            'daily': ['日', 'daily', 'day'],
            'weekly': ['周', 'weekly', 'week'],
            'monthly': ['月', 'monthly', 'month'],
            'quarterly': ['季', 'quarterly', 'quarter'],
            'yearly': ['年', 'yearly', 'annual']
        }
    
    def scan_crawling_directory(self) -> Dict[str, InterfaceMetadata]:
        """
        扫描爬虫目录中的所有接口
        
        Returns:
            接口元数据字典
        """
        logger.info(f"开始扫描爬虫接口目录: {self.crawling_dir}")
        
        if not os.path.exists(self.crawling_dir):
            logger.error(f"爬虫目录不存在: {self.crawling_dir}")
            return {}
        
        python_files = [f for f in os.listdir(self.crawling_dir) 
                       if f.endswith('.py') and f != '__init__.py']
        
        logger.info(f"发现 {len(python_files)} 个Python文件")
        
        for file_name in python_files:
            try:
                file_path = os.path.join(self.crawling_dir, file_name)
                module_name = file_name[:-3]  # 移除.py扩展名
                
                metadata = self._analyze_interface_file(file_path, module_name)
                if metadata:
                    self.interfaces_metadata[module_name] = metadata
                    logger.info(f"成功分析接口: {module_name}")
                
            except Exception as e:
                logger.error(f"分析接口文件 {file_name} 时出错: {e}")
                self.compatibility_issues.append({
                    'file': file_name,
                    'error': str(e),
                    'type': 'analysis_error'
                })
        
        logger.info(f"扫描完成，共分析 {len(self.interfaces_metadata)} 个接口")
        return self.interfaces_metadata
    
    def _analyze_interface_file(self, file_path: str, module_name: str) -> Optional[InterfaceMetadata]:
        """
        分析单个接口文件
        
        Args:
            file_path: 文件路径
            module_name: 模块名称
            
        Returns:
            接口元数据
        """
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # 解析AST
            tree = ast.parse(source_code)
            
            # 提取函数信息
            functions = self._extract_functions(tree, source_code, file_path)
            
            # 分析数据类型
            data_types = self._analyze_data_types(source_code, functions)
            
            # 分析更新频率
            update_frequency = self._analyze_update_frequency(source_code, functions)
            
            # 计算可靠性评分
            reliability_score = self._calculate_reliability_score(source_code, functions)
            
            # 提取依赖
            dependencies = self._extract_dependencies(tree)
            
            # 提取API端点
            api_endpoints = self._extract_api_endpoints(source_code)
            
            # 分析数据源
            data_sources = self._analyze_data_sources(source_code, api_endpoints)
            
            # 获取文件修改时间
            last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            return InterfaceMetadata(
                module_name=module_name,
                file_path=file_path,
                functions=functions,
                data_types=data_types,
                update_frequency=update_frequency,
                reliability_score=reliability_score,
                last_modified=last_modified,
                dependencies=dependencies,
                api_endpoints=api_endpoints,
                data_sources=data_sources
            )
            
        except Exception as e:
            logger.error(f"分析文件 {file_path} 时出错: {e}")
            return None
    
    def _extract_functions(self, tree: ast.AST, source_code: str, file_path: str) -> List[FunctionSignature]:
        """提取函数签名信息"""
        functions = []
        source_lines = source_code.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # 提取参数信息
                parameters = []
                for arg in node.args.args:
                    param_info = {
                        'name': arg.arg,
                        'type': self._get_annotation_string(arg.annotation) if arg.annotation else 'Any',
                        'default': None
                    }
                    parameters.append(param_info)
                
                # 处理默认参数
                defaults = node.args.defaults
                if defaults:
                    for i, default in enumerate(defaults):
                        param_idx = len(parameters) - len(defaults) + i
                        if param_idx >= 0:
                            parameters[param_idx]['default'] = ast.unparse(default)
                
                # 提取返回类型
                return_type = 'Any'
                if node.returns:
                    return_type = self._get_annotation_string(node.returns)
                
                # 提取文档字符串
                docstring = ast.get_docstring(node)
                
                functions.append(FunctionSignature(
                    name=node.name,
                    parameters=parameters,
                    return_type=return_type,
                    docstring=docstring,
                    source_file=file_path,
                    line_number=node.lineno
                ))
        
        return functions
    
    def _get_annotation_string(self, annotation) -> str:
        """获取类型注解字符串"""
        try:
            return ast.unparse(annotation)
        except:
            return 'Any'
    
    def _analyze_data_types(self, source_code: str, functions: List[FunctionSignature]) -> List[str]:
        """分析数据类型"""
        data_types = set()
        
        # 基于文件内容分析
        content_lower = source_code.lower()
        for data_type, keywords in self.data_type_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                data_types.add(data_type)
        
        # 基于函数名分析
        for func in functions:
            func_name_lower = func.name.lower()
            for data_type, keywords in self.data_type_keywords.items():
                if any(keyword in func_name_lower for keyword in keywords):
                    data_types.add(data_type)
        
        return list(data_types)
    
    def _analyze_update_frequency(self, source_code: str, functions: List[FunctionSignature]) -> str:
        """分析更新频率"""
        content_lower = source_code.lower()
        
        # 检查频率关键词
        for frequency, keywords in self.frequency_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return frequency
        
        # 基于函数名推断
        for func in functions:
            func_name_lower = func.name.lower()
            for frequency, keywords in self.frequency_keywords.items():
                if any(keyword in func_name_lower for keyword in keywords):
                    return frequency
        
        return 'unknown'
    
    def _calculate_reliability_score(self, source_code: str, functions: List[FunctionSignature]) -> float:
        """计算可靠性评分"""
        score = 50.0  # 基础分数
        
        # 有文档字符串加分
        documented_functions = sum(1 for func in functions if func.docstring)
        if functions:
            doc_ratio = documented_functions / len(functions)
            score += doc_ratio * 20
        
        # 有类型注解加分
        typed_functions = sum(1 for func in functions 
                            if any(param['type'] != 'Any' for param in func.parameters))
        if functions:
            type_ratio = typed_functions / len(functions)
            score += type_ratio * 15
        
        # 有错误处理加分
        if 'try:' in source_code and 'except' in source_code:
            score += 10
        
        # 有日志记录加分
        if 'logging' in source_code or 'logger' in source_code:
            score += 5
        
        return min(score, 100.0)
    
    def _extract_dependencies(self, tree: ast.AST) -> List[str]:
        """提取依赖包"""
        dependencies = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.add(node.module.split('.')[0])
        
        return list(dependencies)
    
    def _extract_api_endpoints(self, source_code: str) -> List[str]:
        """提取API端点"""
        endpoints = []
        
        # 查找URL模式
        import re
        url_pattern = r'["\']https?://[^"\']+["\']'
        urls = re.findall(url_pattern, source_code)
        
        for url in urls:
            clean_url = url.strip('"\'')
            endpoints.append(clean_url)
        
        return endpoints
    
    def _analyze_data_sources(self, source_code: str, api_endpoints: List[str]) -> List[str]:
        """分析数据源"""
        sources = set()
        
        # 基于API端点分析
        for endpoint in api_endpoints:
            if 'eastmoney' in endpoint:
                sources.add('东方财富')
            elif 'sina' in endpoint:
                sources.add('新浪财经')
            elif '10jqka' in endpoint:
                sources.add('同花顺')
            elif 'tencent' in endpoint:
                sources.add('腾讯财经')
        
        # 基于代码内容分析
        content_lower = source_code.lower()
        if 'eastmoney' in content_lower:
            sources.add('东方财富')
        if 'sina' in content_lower:
            sources.add('新浪财经')
        if '10jqka' in content_lower or 'tonghuashun' in content_lower:
            sources.add('同花顺')
        
        return list(sources)
    
    def check_interface_compatibility(self, interface_name: str) -> Dict[str, Any]:
        """检查接口兼容性"""
        if interface_name not in self.interfaces_metadata:
            return {'compatible': False, 'error': 'Interface not found'}
        
        metadata = self.interfaces_metadata[interface_name]
        compatibility_report = {
            'compatible': True,
            'issues': [],
            'recommendations': []
        }
        
        # 检查必要的依赖
        required_deps = ['pandas', 'requests']
        missing_deps = [dep for dep in required_deps if dep not in metadata.dependencies]
        if missing_deps:
            compatibility_report['issues'].append(f"缺少必要依赖: {missing_deps}")
        
        # 检查函数返回类型
        for func in metadata.functions:
            if func.return_type == 'Any':
                compatibility_report['recommendations'].append(
                    f"建议为函数 {func.name} 添加返回类型注解"
                )
        
        # 检查可靠性评分
        if metadata.reliability_score < 60:
            compatibility_report['issues'].append(
                f"可靠性评分较低: {metadata.reliability_score:.1f}"
            )
            compatibility_report['recommendations'].append("建议添加更多文档和错误处理")
        
        return compatibility_report
    
    def generate_interface_report(self) -> pd.DataFrame:
        """生成接口报告"""
        if not self.interfaces_metadata:
            return pd.DataFrame()
        
        report_data = []
        for name, metadata in self.interfaces_metadata.items():
            report_data.append({
                '接口名称': name,
                '函数数量': len(metadata.functions),
                '数据类型': ', '.join(metadata.data_types),
                '更新频率': metadata.update_frequency,
                '可靠性评分': metadata.reliability_score,
                '数据源': ', '.join(metadata.data_sources),
                '依赖包数量': len(metadata.dependencies),
                'API端点数量': len(metadata.api_endpoints),
                '最后修改时间': metadata.last_modified.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return pd.DataFrame(report_data)
    
    def get_interface_metadata_dict(self) -> Dict[str, Dict]:
        """获取接口元数据字典格式"""
        return {name: asdict(metadata) for name, metadata in self.interfaces_metadata.items()}
    
    def export_metadata_to_json(self, output_path: str):
        """导出元数据到JSON文件"""
        import json
        
        # 转换datetime对象为字符串
        def datetime_converter(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        metadata_dict = self.get_interface_metadata_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, ensure_ascii=False, indent=2, default=datetime_converter)
        
        logger.info(f"接口元数据已导出到: {output_path}")


# 测试和使用示例
def test_crawling_interface_scanner():
    """测试爬虫接口扫描器"""
    print("🔍 测试爬虫接口扫描器")
    print("=" * 50)
    
    scanner = CrawlingInterfaceScanner()
    
    # 扫描接口
    print("1. 扫描爬虫接口...")
    interfaces = scanner.scan_crawling_directory()
    print(f"   发现 {len(interfaces)} 个接口")
    
    # 生成报告
    print("2. 生成接口报告...")
    report = scanner.generate_interface_report()
    print(f"   报告包含 {len(report)} 行数据")
    print("\n接口概览:")
    print(report.to_string(index=False))
    
    # 检查兼容性
    print("\n3. 检查接口兼容性...")
    for interface_name in list(interfaces.keys())[:3]:  # 检查前3个接口
        compatibility = scanner.check_interface_compatibility(interface_name)
        print(f"   {interface_name}: {'✅' if compatibility['compatible'] else '❌'}")
        if compatibility.get('issues'):
            for issue in compatibility['issues']:
                print(f"     问题: {issue}")
    
    # 导出元数据
    print("\n4. 导出元数据...")
    output_path = "crawling_interfaces_metadata.json"
    scanner.export_metadata_to_json(output_path)
    
    print("\n✅ 爬虫接口扫描器测试完成!")
    return scanner


if __name__ == "__main__":
    test_crawling_interface_scanner()