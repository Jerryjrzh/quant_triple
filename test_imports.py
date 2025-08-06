#!/usr/bin/env python3
"""
Simple import test for alert system
"""

try:
    print("Testing imports...")
    
    # Test database models
    from stock_analysis_system.data.models import Alert, NotificationLog
    print("✓ Database models imported successfully")
    
    # Test alert engine
    from stock_analysis_system.alerts.alert_engine import AlertEngine, Alert as AlertClass
    print("✓ Alert engine imported successfully")
    
    # Test notification system
    from stock_analysis_system.alerts.notification_system import NotificationSystem
    print("✓ Notification system imported successfully")
    
    # Test alert filtering
    from stock_analysis_system.alerts.alert_filtering import SmartAlertFilter, AlertAggregator
    print("✓ Alert filtering imported successfully")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()