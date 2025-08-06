"""
Quality Assurance Processes for Stock Analysis System

This module provides comprehensive quality assurance functionality including:
- Code review and quality gates
- Static code analysis and security scanning
- Performance benchmarking and regression testing
- Documentation and API testing

Author: Stock Analysis System Team
Date: 2024-01-20
"""

import os
import json
import subprocess
import ast
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    """Configuration for quality assurance processes"""
    project_root: str = "."
    code_quality_threshold: float = 8.0  # Out of 10
    security_scan_enabled: bool = True
    performance_baseline_file: str = "performance_baseline.json"
    documentation_coverage_threshold: float = 80.0
    api_test_timeout: int = 30
    max_complexity: int = 10
    max_line_length: int = 120
    enable_type_checking: bool = True


@dataclass
class CodeQualityResult:
    """Code quality analysis result"""
    file_path: str
    score: float
    issues: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    suggestions: List[str]


@dataclass
class SecurityScanResult:
    """Security scan result"""
    severity: str
    issue_type: str
    file_path: str
    line_number: int
    description: str
    recommendation: str


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result"""
    test_name: str
    execution_time: float
    memory_usage: float
    baseline_time: Optional[float]
    baseline_memory: Optional[float]
    regression_detected: bool


@dataclass
class QualityReport:
    """Comprehensive quality report"""
    timestamp: datetime
    overall_score: float
    code_quality: List[CodeQualityResult]
    security_issues: List[SecurityScanResult]
    performance_benchmarks: List[PerformanceBenchmark]
    documentation_coverage: float
    api_test_results: Dict[str, Any]
    recommendations: List[str]


class CodeQualityAnalyzer:
    """Analyzes code quality using multiple tools"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        
    def analyze_file(self, file_path: str) -> CodeQualityResult:
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Calculate metrics
            metrics = self._calculate_metrics(content, tree)
            
            # Find issues
            issues = self._find_issues(content, tree, file_path)
            
            # Calculate score
            score = self._calculate_score(metrics, issues)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(metrics, issues)
            
            return CodeQualityResult(
                file_path=file_path,
                score=score,
                issues=issues,
                metrics=metrics,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return CodeQualityResult(
                file_path=file_path,
                score=0.0,
                issues=[{"type": "error", "message": str(e)}],
                metrics={},
                suggestions=[]
            )
    
    def analyze_project(self) -> List[CodeQualityResult]:
        """Analyze all Python files in the project"""
        results = []
        python_files = self._find_python_files()
        
        logger.info(f"Analyzing {len(python_files)} Python files")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.analyze_file, file_path): file_path 
                      for file_path in python_files}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    file_path = futures[future]
                    logger.error(f"Failed to analyze {file_path}: {e}")
        
        return results
    
    def _find_python_files(self) -> List[str]:
        """Find all Python files in the project"""
        python_files = []
        project_path = Path(self.config.project_root)
        
        for file_path in project_path.rglob("*.py"):
            # Skip virtual environments and build directories
            if any(part in str(file_path) for part in ['.venv', 'venv', '__pycache__', '.git', 'build']):
                continue
            python_files.append(str(file_path))
        
        return python_files
    
    def _calculate_metrics(self, content: str, tree: ast.AST) -> Dict[str, Any]:
        """Calculate code metrics"""
        lines = content.split('\n')
        
        metrics = {
            "lines_of_code": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            "total_lines": len(lines),
            "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
            "blank_lines": len([line for line in lines if not line.strip()]),
            "functions": len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
            "classes": len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
            "imports": len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]),
            "complexity": self._calculate_complexity(tree),
            "max_line_length": max(len(line) for line in lines) if lines else 0
        }
        
        # Calculate derived metrics
        if metrics["total_lines"] > 0:
            metrics["comment_ratio"] = metrics["comment_lines"] / metrics["total_lines"]
        else:
            metrics["comment_ratio"] = 0
        
        return metrics
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _find_issues(self, content: str, tree: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        """Find code quality issues"""
        issues = []
        lines = content.split('\n')
        
        # Check line length
        for i, line in enumerate(lines, 1):
            if len(line) > self.config.max_line_length:
                issues.append({
                    "type": "style",
                    "severity": "warning",
                    "line": i,
                    "message": f"Line too long ({len(line)} > {self.config.max_line_length})",
                    "rule": "line-length"
                })
        
        # Check complexity
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_complexity = self._calculate_function_complexity(node)
                if func_complexity > self.config.max_complexity:
                    issues.append({
                        "type": "complexity",
                        "severity": "warning",
                        "line": node.lineno,
                        "message": f"Function '{node.name}' is too complex ({func_complexity} > {self.config.max_complexity})",
                        "rule": "complexity"
                    })
        
        # Check for missing docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    issues.append({
                        "type": "documentation",
                        "severity": "info",
                        "line": node.lineno,
                        "message": f"{type(node).__name__} '{node.name}' missing docstring",
                        "rule": "missing-docstring"
                    })
        
        # Check for unused imports
        imports = []
        used_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))
            elif isinstance(node, ast.Name):
                used_names.add(node.id)
        
        for import_name, line_no in imports:
            if import_name not in used_names and not import_name.startswith('_'):
                issues.append({
                    "type": "unused",
                    "severity": "info",
                    "line": line_no,
                    "message": f"Unused import '{import_name}'",
                    "rule": "unused-import"
                })
        
        return issues
    
    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate complexity for a specific function"""
        complexity = 1
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _calculate_score(self, metrics: Dict[str, Any], issues: List[Dict[str, Any]]) -> float:
        """Calculate overall quality score"""
        base_score = 10.0
        
        # Deduct points for issues
        for issue in issues:
            if issue["severity"] == "error":
                base_score -= 2.0
            elif issue["severity"] == "warning":
                base_score -= 1.0
            elif issue["severity"] == "info":
                base_score -= 0.5
        
        # Bonus for good practices
        if metrics.get("comment_ratio", 0) > 0.2:
            base_score += 0.5
        
        if metrics.get("complexity", 0) < 5:
            base_score += 0.5
        
        return max(0.0, min(10.0, base_score))
    
    def _generate_suggestions(self, metrics: Dict[str, Any], issues: List[Dict[str, Any]]) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if metrics.get("comment_ratio", 0) < 0.1:
            suggestions.append("Add more comments to improve code readability")
        
        if metrics.get("complexity", 0) > self.config.max_complexity:
            suggestions.append("Consider breaking down complex functions into smaller ones")
        
        if len([i for i in issues if i["type"] == "documentation"]) > 0:
            suggestions.append("Add docstrings to functions and classes")
        
        if len([i for i in issues if i["type"] == "unused"]) > 0:
            suggestions.append("Remove unused imports and variables")
        
        return suggestions


class SecurityScanner:
    """Scans code for security vulnerabilities"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        
    def scan_project(self) -> List[SecurityScanResult]:
        """Scan the entire project for security issues"""
        results = []
        
        # Run bandit security scanner
        try:
            bandit_results = self._run_bandit()
            results.extend(bandit_results)
        except Exception as e:
            logger.warning(f"Bandit scan failed: {e}")
        
        # Run custom security checks
        custom_results = self._run_custom_checks()
        results.extend(custom_results)
        
        return results
    
    def _run_bandit(self) -> List[SecurityScanResult]:
        """Run bandit security scanner"""
        try:
            result = subprocess.run([
                "bandit", "-r", self.config.project_root, "-f", "json"
            ], capture_output=True, text=True, check=False)
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                return self._parse_bandit_results(bandit_data)
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Bandit not available or failed: {e}")
        
        return []
    
    def _parse_bandit_results(self, bandit_data: Dict[str, Any]) -> List[SecurityScanResult]:
        """Parse bandit results"""
        results = []
        
        for result in bandit_data.get("results", []):
            results.append(SecurityScanResult(
                severity=result.get("issue_severity", "UNKNOWN"),
                issue_type=result.get("test_name", "Unknown"),
                file_path=result.get("filename", ""),
                line_number=result.get("line_number", 0),
                description=result.get("issue_text", ""),
                recommendation=result.get("issue_cwe", {}).get("message", "")
            ))
        
        return results
    
    def _run_custom_checks(self) -> List[SecurityScanResult]:
        """Run custom security checks"""
        results = []
        python_files = self._find_python_files()
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_results = self._check_file_security(file_path, content)
                results.extend(file_results)
                
            except Exception as e:
                logger.warning(f"Failed to scan {file_path}: {e}")
        
        return results
    
    def _check_file_security(self, file_path: str, content: str) -> List[SecurityScanResult]:
        """Check a file for security issues"""
        results = []
        lines = content.split('\n')
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, description in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    results.append(SecurityScanResult(
                        severity="HIGH",
                        issue_type="hardcoded_secret",
                        file_path=file_path,
                        line_number=i,
                        description=description,
                        recommendation="Use environment variables or secure configuration"
                    ))
        
        # Check for SQL injection risks
        sql_patterns = [
            r'execute\s*\(\s*["\'][^"\']*%[^"\']*["\']',
            r'query\s*\(\s*["\'][^"\']*\+[^"\']*["\']',
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in sql_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    results.append(SecurityScanResult(
                        severity="MEDIUM",
                        issue_type="sql_injection",
                        file_path=file_path,
                        line_number=i,
                        description="Potential SQL injection vulnerability",
                        recommendation="Use parameterized queries"
                    ))
        
        return results
    
    def _find_python_files(self) -> List[str]:
        """Find all Python files in the project"""
        python_files = []
        project_path = Path(self.config.project_root)
        
        for file_path in project_path.rglob("*.py"):
            if any(part in str(file_path) for part in ['.venv', 'venv', '__pycache__', '.git']):
                continue
            python_files.append(str(file_path))
        
        return python_files


class PerformanceBenchmarker:
    """Benchmarks performance and detects regressions"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.baseline = self._load_baseline()
        
    def run_benchmarks(self) -> List[PerformanceBenchmark]:
        """Run performance benchmarks"""
        benchmarks = []
        
        # Define benchmark tests
        benchmark_tests = [
            ("data_loading", self._benchmark_data_loading),
            ("spring_festival_analysis", self._benchmark_spring_festival),
            ("risk_calculation", self._benchmark_risk_calculation),
            ("database_query", self._benchmark_database),
            ("api_response", self._benchmark_api),
        ]
        
        for test_name, test_func in benchmark_tests:
            try:
                execution_time, memory_usage = test_func()
                
                baseline_time = self.baseline.get(test_name, {}).get("execution_time")
                baseline_memory = self.baseline.get(test_name, {}).get("memory_usage")
                
                # Check for regression (20% slower or 50% more memory)
                regression_detected = False
                if baseline_time and execution_time > baseline_time * 1.2:
                    regression_detected = True
                if baseline_memory and memory_usage > baseline_memory * 1.5:
                    regression_detected = True
                
                benchmark = PerformanceBenchmark(
                    test_name=test_name,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    baseline_time=baseline_time,
                    baseline_memory=baseline_memory,
                    regression_detected=regression_detected
                )
                
                benchmarks.append(benchmark)
                
            except Exception as e:
                logger.error(f"Benchmark {test_name} failed: {e}")
        
        return benchmarks
    
    def _load_baseline(self) -> Dict[str, Any]:
        """Load performance baseline"""
        try:
            with open(self.config.performance_baseline_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.info("No performance baseline found, creating new one")
            return {}
    
    def save_baseline(self, benchmarks: List[PerformanceBenchmark]):
        """Save current results as baseline"""
        baseline = {}
        
        for benchmark in benchmarks:
            baseline[benchmark.test_name] = {
                "execution_time": benchmark.execution_time,
                "memory_usage": benchmark.memory_usage,
                "timestamp": datetime.now().isoformat()
            }
        
        with open(self.config.performance_baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        logger.info(f"Saved performance baseline with {len(benchmarks)} benchmarks")
    
    def _benchmark_data_loading(self) -> Tuple[float, float]:
        """Benchmark data loading performance"""
        import psutil
        import time
        
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        # Simulate data loading
        data = list(range(100000))
        processed_data = [x * 2 for x in data]
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return end_time - start_time, end_memory - start_memory
    
    def _benchmark_spring_festival(self) -> Tuple[float, float]:
        """Benchmark Spring Festival analysis"""
        import psutil
        import time
        
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        # Simulate Spring Festival analysis
        dates = [f"2024-{i:02d}-{j:02d}" for i in range(1, 13) for j in range(1, 29)]
        analysis_results = [hash(date) % 100 for date in dates]
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024
        
        return end_time - start_time, end_memory - start_memory
    
    def _benchmark_risk_calculation(self) -> Tuple[float, float]:
        """Benchmark risk calculation"""
        import psutil
        import time
        import numpy as np
        
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        # Simulate risk calculation
        returns = np.random.normal(0, 0.02, 1000)
        var_95 = np.percentile(returns, 5)
        volatility = np.std(returns)
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024
        
        return end_time - start_time, end_memory - start_memory
    
    def _benchmark_database(self) -> Tuple[float, float]:
        """Benchmark database operations"""
        import psutil
        import time
        
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        # Simulate database operations
        mock_data = [{"id": i, "value": i * 2} for i in range(10000)]
        filtered_data = [item for item in mock_data if item["value"] > 5000]
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024
        
        return end_time - start_time, end_memory - start_memory
    
    def _benchmark_api(self) -> Tuple[float, float]:
        """Benchmark API response time"""
        import psutil
        import time
        
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        # Simulate API processing
        request_data = {"symbol": "000001.SZ", "start_date": "2024-01-01", "end_date": "2024-12-31"}
        response_data = {"status": "success", "data": [1, 2, 3, 4, 5] * 1000}
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024
        
        return end_time - start_time, end_memory - start_memory


class DocumentationChecker:
    """Checks documentation coverage and quality"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        
    def check_coverage(self) -> float:
        """Check documentation coverage"""
        python_files = self._find_python_files()
        total_items = 0
        documented_items = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                file_total, file_documented = self._analyze_file_documentation(tree)
                total_items += file_total
                documented_items += file_documented
                
            except Exception as e:
                logger.warning(f"Failed to analyze documentation in {file_path}: {e}")
        
        if total_items == 0:
            return 100.0
        
        return (documented_items / total_items) * 100.0
    
    def _analyze_file_documentation(self, tree: ast.AST) -> Tuple[int, int]:
        """Analyze documentation in a single file"""
        total_items = 0
        documented_items = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                total_items += 1
                if ast.get_docstring(node):
                    documented_items += 1
        
        return total_items, documented_items
    
    def _find_python_files(self) -> List[str]:
        """Find all Python files in the project"""
        python_files = []
        project_path = Path(self.config.project_root)
        
        for file_path in project_path.rglob("*.py"):
            if any(part in str(file_path) for part in ['.venv', 'venv', '__pycache__', '.git']):
                continue
            python_files.append(str(file_path))
        
        return python_files


class APITester:
    """Tests API endpoints"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        
    def test_endpoints(self, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
        """Test API endpoints"""
        endpoints = [
            ("GET", "/health"),
            ("GET", "/api/stocks"),
            ("POST", "/api/analysis/spring-festival"),
            ("GET", "/api/risk/var"),
        ]
        
        results = {
            "total_tests": len(endpoints),
            "passed": 0,
            "failed": 0,
            "endpoints": {}
        }
        
        for method, endpoint in endpoints:
            try:
                result = self._test_endpoint(base_url, method, endpoint)
                results["endpoints"][f"{method} {endpoint}"] = result
                
                if result["status"] == "success":
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to test {method} {endpoint}: {e}")
                results["failed"] += 1
                results["endpoints"][f"{method} {endpoint}"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    def _test_endpoint(self, base_url: str, method: str, endpoint: str) -> Dict[str, Any]:
        """Test a single endpoint"""
        url = f"{base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=self.config.api_test_timeout)
            elif method == "POST":
                response = requests.post(url, json={}, timeout=self.config.api_test_timeout)
            else:
                return {"status": "error", "error": f"Unsupported method: {method}"}
            
            return {
                "status": "success" if response.status_code < 400 else "failure",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "content_length": len(response.content)
            }
            
        except requests.RequestException as e:
            return {
                "status": "error",
                "error": str(e)
            }


class QualityAssuranceManager:
    """Main quality assurance orchestrator"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.code_analyzer = CodeQualityAnalyzer(config)
        self.security_scanner = SecurityScanner(config)
        self.performance_benchmarker = PerformanceBenchmarker(config)
        self.documentation_checker = DocumentationChecker(config)
        self.api_tester = APITester(config)
        
    def run_full_qa(self, api_base_url: str = None) -> QualityReport:
        """Run complete quality assurance process"""
        logger.info("Starting comprehensive quality assurance process")
        start_time = time.time()
        
        # Code quality analysis
        logger.info("Running code quality analysis...")
        code_quality_results = self.code_analyzer.analyze_project()
        
        # Security scanning
        logger.info("Running security scan...")
        security_results = []
        if self.config.security_scan_enabled:
            security_results = self.security_scanner.scan_project()
        
        # Performance benchmarking
        logger.info("Running performance benchmarks...")
        performance_results = self.performance_benchmarker.run_benchmarks()
        
        # Documentation coverage
        logger.info("Checking documentation coverage...")
        doc_coverage = self.documentation_checker.check_coverage()
        
        # API testing
        api_results = {}
        if api_base_url:
            logger.info("Testing API endpoints...")
            api_results = self.api_tester.test_endpoints(api_base_url)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            code_quality_results, security_results, performance_results, doc_coverage, api_results
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            code_quality_results, security_results, performance_results, doc_coverage, api_results
        )
        
        duration = time.time() - start_time
        logger.info(f"Quality assurance completed in {duration:.2f} seconds")
        
        return QualityReport(
            timestamp=datetime.now(),
            overall_score=overall_score,
            code_quality=code_quality_results,
            security_issues=security_results,
            performance_benchmarks=performance_results,
            documentation_coverage=doc_coverage,
            api_test_results=api_results,
            recommendations=recommendations
        )
    
    def _calculate_overall_score(self, 
                                code_quality: List[CodeQualityResult],
                                security_issues: List[SecurityScanResult],
                                performance_benchmarks: List[PerformanceBenchmark],
                                doc_coverage: float,
                                api_results: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        scores = []
        
        # Code quality score (40% weight)
        if code_quality:
            avg_code_score = sum(result.score for result in code_quality) / len(code_quality)
            scores.append(("code_quality", avg_code_score, 0.4))
        
        # Security score (25% weight)
        security_score = 10.0
        for issue in security_issues:
            if issue.severity == "HIGH":
                security_score -= 3.0
            elif issue.severity == "MEDIUM":
                security_score -= 1.5
            elif issue.severity == "LOW":
                security_score -= 0.5
        security_score = max(0.0, security_score)
        scores.append(("security", security_score, 0.25))
        
        # Performance score (20% weight)
        performance_score = 10.0
        regressions = sum(1 for b in performance_benchmarks if b.regression_detected)
        performance_score -= regressions * 2.0
        performance_score = max(0.0, performance_score)
        scores.append(("performance", performance_score, 0.2))
        
        # Documentation score (10% weight)
        doc_score = min(10.0, doc_coverage / 10.0)
        scores.append(("documentation", doc_score, 0.1))
        
        # API score (5% weight)
        api_score = 10.0
        if api_results:
            if api_results.get("failed", 0) > 0:
                api_score = 5.0
        scores.append(("api", api_score, 0.05))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        return min(10.0, max(0.0, total_score))
    
    def _generate_recommendations(self,
                                 code_quality: List[CodeQualityResult],
                                 security_issues: List[SecurityScanResult],
                                 performance_benchmarks: List[PerformanceBenchmark],
                                 doc_coverage: float,
                                 api_results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Code quality recommendations
        low_quality_files = [r for r in code_quality if r.score < self.config.code_quality_threshold]
        if low_quality_files:
            recommendations.append(f"Improve code quality in {len(low_quality_files)} files with scores below {self.config.code_quality_threshold}")
        
        # Security recommendations
        high_security_issues = [i for i in security_issues if i.severity == "HIGH"]
        if high_security_issues:
            recommendations.append(f"Address {len(high_security_issues)} high-severity security issues immediately")
        
        # Performance recommendations
        regressions = [b for b in performance_benchmarks if b.regression_detected]
        if regressions:
            recommendations.append(f"Investigate {len(regressions)} performance regressions")
        
        # Documentation recommendations
        if doc_coverage < self.config.documentation_coverage_threshold:
            recommendations.append(f"Improve documentation coverage from {doc_coverage:.1f}% to {self.config.documentation_coverage_threshold}%")
        
        # API recommendations
        if api_results and api_results.get("failed", 0) > 0:
            recommendations.append(f"Fix {api_results['failed']} failing API endpoints")
        
        return recommendations
    
    def generate_report(self, report: QualityReport) -> str:
        """Generate a comprehensive quality report"""
        return f"""
# Quality Assurance Report

**Generated:** {report.timestamp}
**Overall Score:** {report.overall_score:.1f}/10.0

## Code Quality
- **Files Analyzed:** {len(report.code_quality)}
- **Average Score:** {sum(r.score for r in report.code_quality) / len(report.code_quality) if report.code_quality else 0:.1f}/10.0
- **Issues Found:** {sum(len(r.issues) for r in report.code_quality)}

## Security
- **Total Issues:** {len(report.security_issues)}
- **High Severity:** {len([i for i in report.security_issues if i.severity == "HIGH"])}
- **Medium Severity:** {len([i for i in report.security_issues if i.severity == "MEDIUM"])}
- **Low Severity:** {len([i for i in report.security_issues if i.severity == "LOW"])}

## Performance
- **Benchmarks Run:** {len(report.performance_benchmarks)}
- **Regressions Detected:** {len([b for b in report.performance_benchmarks if b.regression_detected])}

## Documentation
- **Coverage:** {report.documentation_coverage:.1f}%

## API Testing
- **Total Tests:** {report.api_test_results.get('total_tests', 0)}
- **Passed:** {report.api_test_results.get('passed', 0)}
- **Failed:** {report.api_test_results.get('failed', 0)}

## Recommendations
{chr(10).join(f"- {rec}" for rec in report.recommendations)}

## Summary
{'✅ Quality gates passed' if report.overall_score >= 8.0 else '❌ Quality gates failed'} - Overall score: {report.overall_score:.1f}/10.0
"""


# Example usage and demo
def create_demo_qa():
    """Create a demo quality assurance setup"""
    config = QualityConfig(
        project_root=".",
        code_quality_threshold=7.0,
        security_scan_enabled=True,
        documentation_coverage_threshold=75.0
    )
    
    return QualityAssuranceManager(config)


if __name__ == "__main__":
    # Demo the quality assurance system
    qa_manager = create_demo_qa()
    
    print("🔍 Starting Quality Assurance Demo")
    print("=" * 50)
    
    # Run full QA process
    report = qa_manager.run_full_qa()
    
    # Generate and display report
    report_text = qa_manager.generate_report(report)
    print(report_text)
    
    print(f"\n📊 Quality Score: {report.overall_score:.1f}/10.0")
    print(f"🔧 Recommendations: {len(report.recommendations)}")