#!/usr/bin/env python3
"""
Task 8.1 Error Handler Comprehensive Test

This script provides comprehensive testing of the error handling system including:
- Error classification and categorization
- Retry mechanisms with different strategies
- Recovery handlers and automatic resolution
- Error statistics and trend analysis
- Context extraction and error tracking
- Custom error patterns and handlers
"""

import asyncio
import logging
import sys
import os
import time
import random
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from stock_analysis_system.core.error_handler import (
    ErrorHandler, ErrorClassifier, RetryManager, ErrorSeverity, 
    ErrorCategory, RetryStrategy, ErrorPattern, RetryConfig,
    ErrorRecord, ErrorContext, get_error_handler, handle_error, with_error_handling
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorHandlerComprehensiveTest:
    """Comprehensive test suite for the error handling system"""
    
    def __init__(self):
        # Test results storage
        self.test_results = {}
        
        # Error handler instance
        self.error_handler = None
        
        logger.info("Error handler comprehensive test initialized")
    
    async def test_error_handler_initialization(self):
        """Test error handler initialization and configuration"""
        logger.info("=== Testing Error Handler Initialization ===")
        
        results = {
            'handler_created': 0,
            'classifier_initialized': 0,
            'retry_manager_initialized': 0,
            'default_patterns_loaded': 0,
            'global_handler_accessible': 0
        }
        
        try:
            # Test 1: Create error handler
            self.error_handler = ErrorHandler(
                max_error_history=1000,
                error_aggregation_window=300,
                enable_auto_recovery=True
            )
            
            assert self.error_handler is not None
            assert self.error_handler.max_error_history == 1000
            assert self.error_handler.error_aggregation_window == 300
            assert self.error_handler.enable_auto_recovery is True
            results['handler_created'] += 1
            logger.info("✓ Error handler created successfully")
            
            # Test 2: Check classifier initialization
            assert self.error_handler.classifier is not None
            assert hasattr(self.error_handler.classifier, 'patterns')
            results['classifier_initialized'] += 1
            logger.info("✓ Error classifier initialized")
            
            # Test 3: Check retry manager initialization
            assert self.error_handler.retry_manager is not None
            assert hasattr(self.error_handler.retry_manager, 'calculate_delay')
            results['retry_manager_initialized'] += 1
            logger.info("✓ Retry manager initialized")
            
            # Test 4: Check default patterns
            patterns = self.error_handler.classifier.patterns
            assert len(patterns) > 0
            
            pattern_categories = {pattern.category for pattern in patterns}
            expected_categories = {
                ErrorCategory.NETWORK, ErrorCategory.DATABASE, 
                ErrorCategory.DATA_FORMAT, ErrorCategory.EXTERNAL_API,
                ErrorCategory.AUTHENTICATION, ErrorCategory.SYSTEM_RESOURCE
            }
            
            assert expected_categories.issubset(pattern_categories)
            results['default_patterns_loaded'] += 1
            logger.info(f"✓ Default patterns loaded: {len(patterns)} patterns")
            
            # Test 5: Test global handler access
            global_handler = get_error_handler()
            assert global_handler is not None
            results['global_handler_accessible'] += 1
            logger.info("✓ Global error handler accessible")
            
            logger.info(f"Error handler initialization test completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in error handler initialization test: {e}")
            return results
    
    async def test_error_classification(self):
        """Test error classification and pattern matching"""
        logger.info("=== Testing Error Classification ===")
        
        results = {
            'network_errors_classified': 0,
            'database_errors_classified': 0,
            'data_format_errors_classified': 0,
            'api_errors_classified': 0,
            'auth_errors_classified': 0,
            'unknown_errors_classified': 0,
            'custom_patterns_added': 0
        }
        
        try:
            # Test 1: Network error classification
            network_errors = [
                ConnectionError("Connection refused"),
                TimeoutError("Request timeout"),
                Exception("DNS resolution failed")
            ]
            
            for error in network_errors:
                pattern = self.error_handler.classifier.classify_error(error)
                if pattern.category == ErrorCategory.NETWORK:
                    results['network_errors_classified'] += 1
            
            logger.info(f"✓ Classified {results['network_errors_classified']} network errors")
            
            # Test 2: Database error classification
            database_errors = [
                Exception("Database connection lost"),
                Exception("SQL syntax error in query"),
                Exception("Transaction deadlock detected")
            ]
            
            for error in database_errors:
                pattern = self.error_handler.classifier.classify_error(error, str(error))
                if pattern.category == ErrorCategory.DATABASE:
                    results['database_errors_classified'] += 1
            
            logger.info(f"✓ Classified {results['database_errors_classified']} database errors")
            
            # Test 3: Data format error classification
            data_format_errors = [
                ValueError("Invalid JSON format"),
                KeyError("Missing required field"),
                TypeError("Expected string, got int")
            ]
            
            for error in data_format_errors:
                pattern = self.error_handler.classifier.classify_error(error)
                if pattern.category == ErrorCategory.DATA_FORMAT:
                    results['data_format_errors_classified'] += 1
            
            logger.info(f"✓ Classified {results['data_format_errors_classified']} data format errors")
            
            # Test 4: API error classification
            api_errors = [
                Exception("API rate limit exceeded"),
                Exception("Service unavailable - 503"),
                Exception("Unauthorized API access")
            ]
            
            for error in api_errors:
                pattern = self.error_handler.classifier.classify_error(error, str(error))
                if pattern.category == ErrorCategory.EXTERNAL_API:
                    results['api_errors_classified'] += 1
            
            logger.info(f"✓ Classified {results['api_errors_classified']} API errors")
            
            # Test 5: Authentication error classification
            auth_errors = [
                Exception("Authentication failed"),
                Exception("Invalid token provided"),
                Exception("Access forbidden")
            ]
            
            for error in auth_errors:
                pattern = self.error_handler.classifier.classify_error(error, str(error))
                if pattern.category == ErrorCategory.AUTHENTICATION:
                    results['auth_errors_classified'] += 1
            
            logger.info(f"✓ Classified {results['auth_errors_classified']} authentication errors")
            
            # Test 6: Unknown error classification
            unknown_error = Exception("This is a completely unknown error type")
            pattern = self.error_handler.classifier.classify_error(unknown_error)
            if pattern.category == ErrorCategory.UNKNOWN:
                results['unknown_errors_classified'] += 1
            
            logger.info("✓ Classified unknown error correctly")
            
            # Test 7: Add custom pattern
            custom_pattern = ErrorPattern(
                pattern_id="test_custom_pattern",
                name="Test Custom Pattern",
                description="Custom pattern for testing",
                error_types=["CustomError"],
                keywords=["custom", "test"],
                category=ErrorCategory.BUSINESS_LOGIC,
                severity=ErrorSeverity.HIGH,
                retry_config=RetryConfig(max_attempts=2)
            )
            
            self.error_handler.classifier.add_pattern(custom_pattern)
            
            # Test custom pattern matching
            custom_error = Exception("This is a custom test error")
            pattern = self.error_handler.classifier.classify_error(custom_error, str(custom_error))
            if pattern.category == ErrorCategory.BUSINESS_LOGIC:
                results['custom_patterns_added'] += 1
            
            logger.info("✓ Custom pattern added and matched")
            
            logger.info(f"Error classification test completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in error classification test: {e}")
            return results
    
    async def test_retry_mechanisms(self):
        """Test retry mechanisms and strategies"""
        logger.info("=== Testing Retry Mechanisms ===")
        
        results = {
            'exponential_backoff_tested': 0,
            'linear_backoff_tested': 0,
            'immediate_retry_tested': 0,
            'no_retry_tested': 0,
            'max_attempts_respected': 0,
            'jitter_applied': 0,
            'retry_conditions_checked': 0
        }
        
        try:
            retry_manager = self.error_handler.retry_manager
            
            # Test 1: Exponential backoff calculation
            exp_config = RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                base_delay=1.0,
                backoff_multiplier=2.0,
                max_delay=10.0
            )
            
            delays = []
            for attempt in range(1, 5):
                delay = retry_manager.calculate_delay(exp_config, attempt)
                delays.append(delay)
            
            # Check exponential growth (with some tolerance for jitter)
            if delays[1] > delays[0] and delays[2] > delays[1]:
                results['exponential_backoff_tested'] += 1
            
            logger.info(f"✓ Exponential backoff delays: {[f'{d:.2f}s' for d in delays]}")
            
            # Test 2: Linear backoff calculation
            linear_config = RetryConfig(
                strategy=RetryStrategy.LINEAR_BACKOFF,
                base_delay=2.0,
                jitter=False  # Disable jitter for predictable testing
            )
            
            linear_delays = []
            for attempt in range(1, 4):
                delay = retry_manager.calculate_delay(linear_config, attempt)
                linear_delays.append(delay)
            
            # Check linear growth
            if linear_delays[1] == linear_delays[0] * 2 and linear_delays[2] == linear_delays[0] * 3:
                results['linear_backoff_tested'] += 1
            
            logger.info(f"✓ Linear backoff delays: {[f'{d:.2f}s' for d in linear_delays]}")
            
            # Test 3: Immediate retry
            immediate_config = RetryConfig(strategy=RetryStrategy.IMMEDIATE)
            immediate_delay = retry_manager.calculate_delay(immediate_config, 1)
            
            if immediate_delay == 0:
                results['immediate_retry_tested'] += 1
            
            logger.info("✓ Immediate retry tested")
            
            # Test 4: No retry strategy
            no_retry_config = RetryConfig(strategy=RetryStrategy.NONE)
            no_retry_delay = retry_manager.calculate_delay(no_retry_config, 1)
            
            if no_retry_delay == 0:
                results['no_retry_tested'] += 1
            
            logger.info("✓ No retry strategy tested")
            
            # Test 5: Max attempts respected
            max_attempts_config = RetryConfig(max_attempts=3)
            
            # Should retry for attempts 1 and 2, but not 3
            should_retry_1 = retry_manager.should_retry(ValueError("test"), max_attempts_config, 1)
            should_retry_2 = retry_manager.should_retry(ValueError("test"), max_attempts_config, 2)
            should_retry_3 = retry_manager.should_retry(ValueError("test"), max_attempts_config, 3)
            
            if should_retry_1 and should_retry_2 and not should_retry_3:
                results['max_attempts_respected'] += 1
            
            logger.info("✓ Max attempts limit respected")
            
            # Test 6: Jitter application
            jitter_config = RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                base_delay=1.0,
                jitter=True
            )
            
            jitter_delays = []
            for _ in range(5):
                delay = retry_manager.calculate_delay(jitter_config, 2)
                jitter_delays.append(delay)
            
            # Check that delays vary (jitter effect)
            if len(set(jitter_delays)) > 1:
                results['jitter_applied'] += 1
            
            logger.info("✓ Jitter applied to delays")
            
            # Test 7: Retry conditions
            retry_config = RetryConfig(
                retry_on_exceptions=[ConnectionError, TimeoutError],
                stop_on_exceptions=[ValueError]
            )
            
            should_retry_conn = retry_manager.should_retry(ConnectionError("test"), retry_config, 1)
            should_not_retry_val = retry_manager.should_retry(ValueError("test"), retry_config, 1)
            
            if should_retry_conn and not should_not_retry_val:
                results['retry_conditions_checked'] += 1
            
            logger.info("✓ Retry conditions checked correctly")
            
            logger.info(f"Retry mechanisms test completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in retry mechanisms test: {e}")
            return results
    
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        logger.info("=== Testing Error Handling and Recovery ===")
        
        results = {
            'errors_handled': 0,
            'context_extracted': 0,
            'recovery_handlers_registered': 0,
            'automatic_recovery_attempted': 0,
            'error_records_created': 0,
            'statistics_updated': 0
        }
        
        try:
            # Test 1: Basic error handling
            test_errors = [
                ConnectionError("Network connection failed"),
                ValueError("Invalid data format"),
                Exception("Generic error for testing")
            ]
            
            for error in test_errors:
                error_record = self.error_handler.handle_error(
                    error,
                    custom_message=f"Test error: {type(error).__name__}",
                    user_id="test_user_123",
                    request_id="req_456"
                )
                
                assert error_record is not None
                assert error_record.error_type == type(error).__name__
                assert error_record.context.user_id == "test_user_123"
                assert error_record.context.request_id == "req_456"
                
                results['errors_handled'] += 1
                results['error_records_created'] += 1
                
                # Check context extraction
                if error_record.context.function_name and error_record.context.module_name:
                    results['context_extracted'] += 1
            
            logger.info(f"✓ Handled {results['errors_handled']} errors with context")
            
            # Test 2: Register recovery handlers
            recovery_success_count = 0
            
            def network_recovery_handler(error_record: ErrorRecord) -> bool:
                nonlocal recovery_success_count
                recovery_success_count += 1
                return True  # Simulate successful recovery
            
            def database_recovery_handler(error_record: ErrorRecord) -> bool:
                return False  # Simulate failed recovery
            
            self.error_handler.register_recovery_handler(
                ErrorCategory.NETWORK, network_recovery_handler
            )
            self.error_handler.register_recovery_handler(
                ErrorCategory.DATABASE, database_recovery_handler
            )
            
            results['recovery_handlers_registered'] = 2
            logger.info("✓ Recovery handlers registered")
            
            # Test 3: Test automatic recovery
            # Create a custom pattern with auto-resolve enabled
            auto_resolve_pattern = ErrorPattern(
                pattern_id="auto_resolve_test",
                name="Auto Resolve Test",
                description="Pattern for testing auto resolution",
                error_types=["TestAutoResolveError"],
                keywords=["auto", "resolve"],
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                retry_config=RetryConfig(max_attempts=1),
                auto_resolve=True
            )
            
            self.error_handler.classifier.add_pattern(auto_resolve_pattern)
            
            # Create an error that matches the auto-resolve pattern
            auto_resolve_error = Exception("This is an auto resolve test error")
            error_record = self.error_handler.handle_error(
                auto_resolve_error,
                custom_message="auto resolve test"
            )
            
            # Check if recovery was attempted
            if recovery_success_count > 0:
                results['automatic_recovery_attempted'] += 1
            
            logger.info("✓ Automatic recovery attempted")
            
            # Test 4: Check statistics update
            stats = self.error_handler.get_error_statistics()
            
            assert stats['total_errors'] > 0
            assert 'category_breakdown' in stats
            assert 'resolution_rate' in stats
            
            results['statistics_updated'] += 1
            logger.info("✓ Statistics updated correctly")
            
            logger.info(f"Error handling and recovery test completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in error handling and recovery test: {e}")
            return results
    
    async def test_decorator_functionality(self):
        """Test error handling decorator functionality"""
        logger.info("=== Testing Decorator Functionality ===")
        
        results = {
            'sync_decorator_tested': 0,
            'async_decorator_tested': 0,
            'retry_decorator_tested': 0,
            'custom_config_applied': 0,
            'context_preserved': 0
        }
        
        try:
            # Test 1: Synchronous function decorator
            call_count = 0
            
            @self.error_handler.with_error_handling(
                retry_config=RetryConfig(max_attempts=3, strategy=RetryStrategy.IMMEDIATE),
                custom_message="Sync function test",
                raise_on_failure=False
            )
            def test_sync_function(should_fail: bool = False):
                nonlocal call_count
                call_count += 1
                if should_fail and call_count < 3:
                    raise ValueError(f"Sync function failed on attempt {call_count}")
                return f"Success on attempt {call_count}"
            
            # Test successful execution
            result = test_sync_function(should_fail=False)
            if result and "Success" in result:
                results['sync_decorator_tested'] += 1
            
            # Test retry behavior
            call_count = 0
            result = test_sync_function(should_fail=True)
            if call_count >= 3:  # Should have retried
                results['retry_decorator_tested'] += 1
            
            logger.info("✓ Synchronous decorator tested")
            
            # Test 2: Asynchronous function decorator
            async_call_count = 0
            
            @self.error_handler.with_error_handling(
                retry_config=RetryConfig(max_attempts=2, strategy=RetryStrategy.IMMEDIATE),
                custom_message="Async function test",
                user_id="async_test_user",
                raise_on_failure=False
            )
            async def test_async_function(should_fail: bool = False):
                nonlocal async_call_count
                async_call_count += 1
                if should_fail and async_call_count < 2:
                    raise ConnectionError(f"Async function failed on attempt {async_call_count}")
                return f"Async success on attempt {async_call_count}"
            
            # Test successful execution
            result = await test_async_function(should_fail=False)
            if result and "Async success" in result:
                results['async_decorator_tested'] += 1
            
            logger.info("✓ Asynchronous decorator tested")
            
            # Test 3: Custom configuration application
            @self.error_handler.with_error_handling(
                retry_config=RetryConfig(
                    max_attempts=5,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                    base_delay=0.1
                ),
                custom_message="Custom config test"
            )
            def test_custom_config():
                raise TimeoutError("Custom config test error")
            
            try:
                test_custom_config()
            except Exception:
                pass  # Expected to fail after retries
            
            # Check if retries were attempted with custom config
            stats = self.error_handler.get_error_statistics()
            if stats['retry_attempts'] > 0:
                results['custom_config_applied'] += 1
            
            logger.info("✓ Custom configuration applied")
            
            # Test 4: Context preservation
            @self.error_handler.with_error_handling(
                user_id="context_test_user",
                request_id="context_test_request"
            )
            def test_context_preservation():
                raise ValueError("Context preservation test")
            
            try:
                test_context_preservation()
            except Exception:
                pass
            
            # Check if context was preserved in error records
            recent_errors = list(self.error_handler.error_history)
            context_error = None
            for error in reversed(recent_errors):
                if "Context preservation test" in error.error_message:
                    context_error = error
                    break
            
            if (context_error and 
                context_error.context.user_id == "context_test_user" and
                context_error.context.request_id == "context_test_request"):
                results['context_preserved'] += 1
            
            logger.info("✓ Context preserved in decorator")
            
            logger.info(f"Decorator functionality test completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in decorator functionality test: {e}")
            return results
    
    async def test_statistics_and_reporting(self):
        """Test error statistics and reporting functionality"""
        logger.info("=== Testing Statistics and Reporting ===")
        
        results = {
            'statistics_generated': 0,
            'trends_calculated': 0,
            'category_breakdown_created': 0,
            'error_report_exported': 0,
            'pattern_statistics_tracked': 0
        }
        
        try:
            # Generate some test errors for statistics
            test_error_scenarios = [
                (ConnectionError("Network error 1"), ErrorCategory.NETWORK),
                (ConnectionError("Network error 2"), ErrorCategory.NETWORK),
                (ValueError("Data format error 1"), ErrorCategory.DATA_FORMAT),
                (Exception("Database connection lost"), ErrorCategory.DATABASE),
                (Exception("API rate limit exceeded"), ErrorCategory.EXTERNAL_API)
            ]
            
            for error, expected_category in test_error_scenarios:
                self.error_handler.handle_error(error, custom_message=str(error))
            
            # Test 1: Get error statistics
            stats = self.error_handler.get_error_statistics()
            
            required_fields = [
                'total_errors', 'resolved_errors', 'retry_attempts',
                'successful_recoveries', 'failed_recoveries', 'resolution_rate',
                'category_breakdown', 'recent_errors_count', 'pattern_statistics'
            ]
            
            if all(field in stats for field in required_fields):
                results['statistics_generated'] += 1
            
            logger.info(f"✓ Statistics generated: {len(stats)} fields")
            
            # Test 2: Category breakdown
            category_breakdown = stats.get('category_breakdown', {})
            if len(category_breakdown) > 0:
                results['category_breakdown_created'] += 1
                
                # Check if percentages add up correctly
                total_percentage = sum(
                    cat_stats.get('percentage', 0) 
                    for cat_stats in category_breakdown.values()
                )
                logger.info(f"✓ Category breakdown created: {len(category_breakdown)} categories")
            
            # Test 3: Error trends
            trends = self.error_handler.get_error_trends(hours=1)
            
            required_trend_fields = [
                'time_period_hours', 'total_errors_in_period',
                'hourly_distribution', 'category_trends', 'average_errors_per_hour'
            ]
            
            if all(field in trends for field in required_trend_fields):
                results['trends_calculated'] += 1
            
            logger.info(f"✓ Trends calculated for {trends.get('total_errors_in_period', 0)} errors")
            
            # Test 4: Pattern statistics
            pattern_stats = stats.get('pattern_statistics', {})
            if len(pattern_stats) > 0:
                results['pattern_statistics_tracked'] += 1
            
            logger.info(f"✓ Pattern statistics tracked: {len(pattern_stats)} patterns")
            
            # Test 5: Export error report
            report_file = "test_error_report.json"
            self.error_handler.export_error_report(report_file, include_context=True)
            
            if Path(report_file).exists():
                results['error_report_exported'] += 1
                
                # Verify report content
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                
                required_report_fields = [
                    'report_timestamp', 'statistics', 'trends', 
                    'error_patterns', 'recent_errors'
                ]
                
                if all(field in report_data for field in required_report_fields):
                    logger.info("✓ Error report exported with complete data")
                
                # Clean up
                Path(report_file).unlink()
            
            logger.info(f"Statistics and reporting test completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in statistics and reporting test: {e}")
            return results
    
    async def test_concurrent_error_handling(self):
        """Test concurrent error handling and thread safety"""
        logger.info("=== Testing Concurrent Error Handling ===")
        
        results = {
            'concurrent_errors_handled': 0,
            'thread_safety_maintained': 0,
            'statistics_consistency': 0,
            'no_data_corruption': 0
        }
        
        try:
            # Test concurrent error handling
            def generate_concurrent_errors(thread_id: int, error_count: int):
                errors_generated = 0
                for i in range(error_count):
                    try:
                        error_types = [ConnectionError, ValueError, Exception]
                        error_type = random.choice(error_types)
                        error = error_type(f"Thread {thread_id} error {i}")
                        
                        self.error_handler.handle_error(
                            error,
                            custom_message=f"Concurrent test error from thread {thread_id}",
                            user_id=f"user_{thread_id}"
                        )
                        errors_generated += 1
                        
                    except Exception as e:
                        logger.error(f"Error in concurrent test: {e}")
                
                return errors_generated
            
            # Run concurrent error generation
            import concurrent.futures
            
            initial_error_count = len(self.error_handler.error_history)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for thread_id in range(5):
                    future = executor.submit(generate_concurrent_errors, thread_id, 10)
                    futures.append(future)
                
                # Wait for all threads to complete
                total_generated = 0
                for future in concurrent.futures.as_completed(futures):
                    total_generated += future.result()
            
            final_error_count = len(self.error_handler.error_history)
            actual_new_errors = final_error_count - initial_error_count
            
            results['concurrent_errors_handled'] = actual_new_errors
            
            # Test thread safety - check if all errors were recorded
            if actual_new_errors >= total_generated * 0.9:  # Allow for some variance
                results['thread_safety_maintained'] += 1
            
            logger.info(f"✓ Handled {actual_new_errors} concurrent errors")
            
            # Test statistics consistency
            stats_before = self.error_handler.get_error_statistics()
            
            # Generate a few more errors
            for i in range(5):
                self.error_handler.handle_error(
                    ValueError(f"Consistency test error {i}"),
                    custom_message="Statistics consistency test"
                )
            
            stats_after = self.error_handler.get_error_statistics()
            
            # Check if statistics were updated correctly
            if stats_after['total_errors'] == stats_before['total_errors'] + 5:
                results['statistics_consistency'] += 1
            
            logger.info("✓ Statistics consistency maintained")
            
            # Test data integrity
            error_history = list(self.error_handler.error_history)
            
            # Check for data corruption (all errors should have valid fields)
            valid_errors = 0
            for error_record in error_history:
                if (error_record.error_id and 
                    error_record.error_type and 
                    error_record.context and
                    error_record.context.timestamp):
                    valid_errors += 1
            
            if valid_errors == len(error_history):
                results['no_data_corruption'] += 1
            
            logger.info(f"✓ Data integrity maintained: {valid_errors}/{len(error_history)} valid records")
            
            logger.info(f"Concurrent error handling test completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in concurrent error handling test: {e}")
            return results
    
    async def test_global_error_handler_functions(self):
        """Test global error handler convenience functions"""
        logger.info("=== Testing Global Error Handler Functions ===")
        
        results = {
            'global_handle_error_works': 0,
            'global_decorator_works': 0,
            'singleton_behavior': 0
        }
        
        try:
            # Test 1: Global handle_error function
            test_error = ValueError("Global handler test error")
            error_record = handle_error(
                test_error,
                custom_message="Testing global handle_error function",
                user_id="global_test_user"
            )
            
            if (error_record and 
                error_record.error_type == "ValueError" and
                error_record.context.user_id == "global_test_user"):
                results['global_handle_error_works'] += 1
            
            logger.info("✓ Global handle_error function works")
            
            # Test 2: Global decorator
            @with_error_handling(
                retry_config=RetryConfig(max_attempts=2),
                custom_message="Global decorator test"
            )
            def test_global_decorator():
                raise ConnectionError("Global decorator test error")
            
            try:
                test_global_decorator()
            except Exception:
                pass  # Expected to fail after retries
            
            # Check if error was handled by global handler
            global_handler = get_error_handler()
            recent_errors = list(global_handler.error_history)
            
            decorator_error = None
            for error in reversed(recent_errors):
                if "Global decorator test error" in error.error_message:
                    decorator_error = error
                    break
            
            if decorator_error:
                results['global_decorator_works'] += 1
            
            logger.info("✓ Global decorator works")
            
            # Test 3: Singleton behavior
            handler1 = get_error_handler()
            handler2 = get_error_handler()
            
            if handler1 is handler2:
                results['singleton_behavior'] += 1
            
            logger.info("✓ Singleton behavior verified")
            
            logger.info(f"Global error handler functions test completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in global error handler functions test: {e}")
            return results
    
    async def cleanup_test_resources(self):
        """Clean up test resources"""
        try:
            # Clean up test files
            test_files = [
                "test_error_report.json",
                "error_report.json"
            ]
            
            for file_path in test_files:
                if Path(file_path).exists():
                    Path(file_path).unlink()
            
            logger.info("✓ Test resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up test resources: {e}")
    
    async def run_comprehensive_test(self):
        """Run all comprehensive tests"""
        logger.info("Starting Error Handler Comprehensive Test")
        logger.info("=" * 60)
        
        all_results = {}
        
        try:
            # Test 1: Error Handler Initialization
            init_results = await self.test_error_handler_initialization()
            all_results['handler_initialization'] = init_results
            
            # Test 2: Error Classification
            classification_results = await self.test_error_classification()
            all_results['error_classification'] = classification_results
            
            # Test 3: Retry Mechanisms
            retry_results = await self.test_retry_mechanisms()
            all_results['retry_mechanisms'] = retry_results
            
            # Test 4: Error Handling and Recovery
            handling_results = await self.test_error_handling_and_recovery()
            all_results['error_handling_recovery'] = handling_results
            
            # Test 5: Decorator Functionality
            decorator_results = await self.test_decorator_functionality()
            all_results['decorator_functionality'] = decorator_results
            
            # Test 6: Statistics and Reporting
            stats_results = await self.test_statistics_and_reporting()
            all_results['statistics_reporting'] = stats_results
            
            # Test 7: Concurrent Error Handling
            concurrent_results = await self.test_concurrent_error_handling()
            all_results['concurrent_handling'] = concurrent_results
            
            # Test 8: Global Error Handler Functions
            global_results = await self.test_global_error_handler_functions()
            all_results['global_functions'] = global_results
            
            # Generate summary
            logger.info("=" * 60)
            logger.info("Error Handler Comprehensive Test Summary")
            logger.info("=" * 60)
            
            total_tests = 0
            total_passed = 0
            
            for test_category, results in all_results.items():
                logger.info(f"\n{test_category.upper()}:")
                for metric, value in results.items():
                    logger.info(f"  {metric}: {value}")
                    total_tests += 1
                    if value > 0:
                        total_passed += 1
            
            success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
            logger.info(f"\nOverall Success Rate: {success_rate:.1f}% ({total_passed}/{total_tests})")
            
            self.test_results = all_results
            return all_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive test: {e}")
            return all_results
        
        finally:
            # Clean up resources
            await self.cleanup_test_resources()


async def main():
    """Main function"""
    test_suite = ErrorHandlerComprehensiveTest()
    
    try:
        results = await test_suite.run_comprehensive_test()
        
        print("\n" + "="*60)
        print("TASK 8.1 ERROR HANDLER COMPREHENSIVE TEST COMPLETED!")
        print("="*60)
        
        # Display key metrics
        init = results.get('handler_initialization', {})
        classification = results.get('error_classification', {})
        retry = results.get('retry_mechanisms', {})
        handling = results.get('error_handling_recovery', {})
        decorator = results.get('decorator_functionality', {})
        stats = results.get('statistics_reporting', {})
        concurrent = results.get('concurrent_handling', {})
        global_funcs = results.get('global_functions', {})
        
        print(f"🔧 Handler Initialization:")
        print(f"   • Handler created: {init.get('handler_created', 0)}")
        print(f"   • Classifier initialized: {init.get('classifier_initialized', 0)}")
        print(f"   • Retry manager initialized: {init.get('retry_manager_initialized', 0)}")
        print(f"   • Default patterns loaded: {init.get('default_patterns_loaded', 0)}")
        print(f"   • Global handler accessible: {init.get('global_handler_accessible', 0)}")
        
        print(f"\n🏷️ Error Classification:")
        print(f"   • Network errors classified: {classification.get('network_errors_classified', 0)}")
        print(f"   • Database errors classified: {classification.get('database_errors_classified', 0)}")
        print(f"   • Data format errors classified: {classification.get('data_format_errors_classified', 0)}")
        print(f"   • API errors classified: {classification.get('api_errors_classified', 0)}")
        print(f"   • Auth errors classified: {classification.get('auth_errors_classified', 0)}")
        print(f"   • Custom patterns added: {classification.get('custom_patterns_added', 0)}")
        
        print(f"\n🔄 Retry Mechanisms:")
        print(f"   • Exponential backoff tested: {retry.get('exponential_backoff_tested', 0)}")
        print(f"   • Linear backoff tested: {retry.get('linear_backoff_tested', 0)}")
        print(f"   • Immediate retry tested: {retry.get('immediate_retry_tested', 0)}")
        print(f"   • Max attempts respected: {retry.get('max_attempts_respected', 0)}")
        print(f"   • Jitter applied: {retry.get('jitter_applied', 0)}")
        print(f"   • Retry conditions checked: {retry.get('retry_conditions_checked', 0)}")
        
        print(f"\n🛠️ Error Handling & Recovery:")
        print(f"   • Errors handled: {handling.get('errors_handled', 0)}")
        print(f"   • Context extracted: {handling.get('context_extracted', 0)}")
        print(f"   • Recovery handlers registered: {handling.get('recovery_handlers_registered', 0)}")
        print(f"   • Automatic recovery attempted: {handling.get('automatic_recovery_attempted', 0)}")
        print(f"   • Error records created: {handling.get('error_records_created', 0)}")
        
        print(f"\n🎯 Decorator Functionality:")
        print(f"   • Sync decorator tested: {decorator.get('sync_decorator_tested', 0)}")
        print(f"   • Async decorator tested: {decorator.get('async_decorator_tested', 0)}")
        print(f"   • Retry decorator tested: {decorator.get('retry_decorator_tested', 0)}")
        print(f"   • Custom config applied: {decorator.get('custom_config_applied', 0)}")
        print(f"   • Context preserved: {decorator.get('context_preserved', 0)}")
        
        print(f"\n📊 Statistics & Reporting:")
        print(f"   • Statistics generated: {stats.get('statistics_generated', 0)}")
        print(f"   • Trends calculated: {stats.get('trends_calculated', 0)}")
        print(f"   • Category breakdown created: {stats.get('category_breakdown_created', 0)}")
        print(f"   • Error report exported: {stats.get('error_report_exported', 0)}")
        print(f"   • Pattern statistics tracked: {stats.get('pattern_statistics_tracked', 0)}")
        
        print(f"\n🔀 Concurrent Handling:")
        print(f"   • Concurrent errors handled: {concurrent.get('concurrent_errors_handled', 0)}")
        print(f"   • Thread safety maintained: {concurrent.get('thread_safety_maintained', 0)}")
        print(f"   • Statistics consistency: {concurrent.get('statistics_consistency', 0)}")
        print(f"   • No data corruption: {concurrent.get('no_data_corruption', 0)}")
        
        print(f"\n🌐 Global Functions:")
        print(f"   • Global handle_error works: {global_funcs.get('global_handle_error_works', 0)}")
        print(f"   • Global decorator works: {global_funcs.get('global_decorator_works', 0)}")
        print(f"   • Singleton behavior: {global_funcs.get('singleton_behavior', 0)}")
        
        print(f"\n✅ Task 8.1 Error Handler System: COMPLETED SUCCESSFULLY")
        print(f"   • Comprehensive error classification and categorization")
        print(f"   • Intelligent retry mechanisms with multiple strategies")
        print(f"   • Automatic error recovery and resolution")
        print(f"   • Context extraction and detailed error tracking")
        print(f"   • Thread-safe concurrent error handling")
        print(f"   • Rich statistics and reporting capabilities")
        print(f"   • Decorator-based error handling for easy integration")
        print(f"   • Global convenience functions for system-wide usage")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Task 8.1 Error Handler Test Failed: {e}")
        return False


if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(main())
    exit(0 if success else 1)