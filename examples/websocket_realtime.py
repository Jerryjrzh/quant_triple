#!/usr/bin/env python3
"""
WebSocket实时数据示例

本示例展示了如何使用WebSocket接收实时股票数据，
包括连接管理、数据订阅、错误处理等。
"""

import websocket
import json
import threading
import time
import queue
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealtimeDataClient:
    """实时数据WebSocket客户端"""
    
    def __init__(self, ws_url: str, token: str):
        self.ws_url = ws_url
        self.token = token
        self.ws = None
        self.connected = False
        self.subscriptions = set()
        self.data_queue = queue.Queue()
        
        # 回调函数
        self.on_data_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        self.on_connect_callback: Optional[Callable] = None
        self.on_disconnect_callback: Optional[Callable] = None
        
        # 统计信息
        self.stats = {
            'messages_received': 0,
            'connection_time': None,
            'last_message_time': None,
            'reconnect_count': 0
        }
        
        # 重连配置
        self.auto_reconnect = True
        self.reconnect_interval = 5
        self.max_reconnect_attempts = 10
        
        logger.info(f"RealtimeDataClient initialized for {ws_url}")
    
    def on_message(self, ws, message):
        """处理接收到的消息"""
        try:
            data = json.loads(message)
            self.stats['messages_received'] += 1
            self.stats['last_message_time'] = datetime.now()
            
            if data['type'] == 'data':
                self.handle_data_message(data)
            elif data['type'] == 'auth_success':
                logger.info("✅ WebSocket认证成功")
                self.connected = True
                if self.on_connect_callback:
                    self.on_connect_callback()
            elif data['type'] == 'error':
                logger.error(f"❌ 服务器错误: {data['message']}")
                if self.on_error_callback:
                    self.on_error_callback(data['message'])
            elif data['type'] == 'subscription_success':
                logger.info(f"✅ 订阅成功: {data.get('channels', [])}")
            elif data['type'] == 'heartbeat':
                logger.debug("💓 收到心跳")
            else:
                logger.warning(f"⚠️ 未知消息类型: {data['type']}")
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON解析错误: {e}")
        except Exception as e:
            logger.error(f"❌ 消息处理错误: {e}")
    
    def handle_data_message(self, data):
        """处理数据消息"""
        channel = data['channel']
        stock_data = data['data']
        timestamp = data['timestamp']
        
        # 添加到队列
        self.data_queue.put({
            'channel': channel,
            'data': stock_data,
            'timestamp': timestamp,
            'received_time': datetime.now()
        })
        
        # 调用回调函数
        if self.on_data_callback:
            self.on_data_callback(channel, stock_data, timestamp)
        
        logger.debug(f"📊 收到数据: {channel} - {stock_data.get('symbol', 'unknown')}")
    
    def on_error(self, ws, error):
        """处理WebSocket错误"""
        logger.error(f"❌ WebSocket错误: {error}")
        if self.on_error_callback:
            self.on_error_callback(error)
    
    def on_close(self, ws, close_status_code, close_msg):
        """连接关闭处理"""
        self.connected = False
        logger.warning(f"🔌 WebSocket连接已关闭: {close_status_code} - {close_msg}")
        
        if self.on_disconnect_callback:
            self.on_disconnect_callback(close_status_code, close_msg)
        
        # 自动重连
        if self.auto_reconnect and self.stats['reconnect_count'] < self.max_reconnect_attempts:
            logger.info(f"🔄 {self.reconnect_interval}秒后尝试重连...")
            time.sleep(self.reconnect_interval)
            self.reconnect()
    
    def on_open(self, ws):
        """连接建立处理"""
        logger.info("🔗 WebSocket连接已建立")
        self.stats['connection_time'] = datetime.now()
        
        # 发送认证消息
        auth_msg = {
            'type': 'auth',
            'token': self.token
        }
        ws.send(json.dumps(auth_msg))
        logger.info("🔐 发送认证信息")
    
    def connect(self):
        """建立WebSocket连接"""
        try:
            websocket.enableTrace(False)  # 禁用调试输出
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # 在新线程中运行
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            logger.info("🚀 WebSocket连接线程已启动")
            
        except Exception as e:
            logger.error(f"❌ 连接失败: {e}")
            raise
    
    def reconnect(self):
        """重新连接"""
        self.stats['reconnect_count'] += 1
        logger.info(f"🔄 尝试重连 (第{self.stats['reconnect_count']}次)")
        
        try:
            if self.ws:
                self.ws.close()
            self.connect()
            
            # 重新订阅
            if self.subscriptions:
                time.sleep(2)  # 等待连接稳定
                self.subscribe(list(self.subscriptions))
                
        except Exception as e:
            logger.error(f"❌ 重连失败: {e}")
    
    def subscribe(self, symbols: List[str]):
        """订阅股票数据"""
        if not self.connected:
            logger.warning("⚠️ 未连接，无法订阅")
            return False
        
        channels = [f'stock.{symbol}' for symbol in symbols]
        self.subscriptions.update(channels)
        
        subscribe_msg = {
            'type': 'subscribe',
            'channels': channels
        }
        
        try:
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"📡 订阅股票: {', '.join(symbols)}")
            return True
        except Exception as e:
            logger.error(f"❌ 订阅失败: {e}")
            return False
    
    def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        if not self.connected:
            logger.warning("⚠️ 未连接，无法取消订阅")
            return False
        
        channels = [f'stock.{symbol}' for symbol in symbols]
        
        unsubscribe_msg = {
            'type': 'unsubscribe',
            'channels': channels
        }
        
        try:
            self.ws.send(json.dumps(unsubscribe_msg))
            self.subscriptions -= set(channels)
            logger.info(f"📡 取消订阅: {', '.join(symbols)}")
            return True
        except Exception as e:
            logger.error(f"❌ 取消订阅失败: {e}")
            return False
    
    def get_latest_data(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """获取最新数据"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'connected': self.connected,
            'subscriptions': len(self.subscriptions),
            'messages_received': self.stats['messages_received'],
            'connection_time': self.stats['connection_time'].isoformat() if self.stats['connection_time'] else None,
            'last_message_time': self.stats['last_message_time'].isoformat() if self.stats['last_message_time'] else None,
            'reconnect_count': self.stats['reconnect_count'],
            'queue_size': self.data_queue.qsize()
        }
    
    def close(self):
        """关闭连接"""
        self.auto_reconnect = False
        if self.ws:
            self.ws.close()
        logger.info("🔌 WebSocket连接已关闭")


class RealtimeDataProcessor:
    """实时数据处理器"""
    
    def __init__(self):
        self.data_buffer = {}
        self.price_alerts = {}
        self.volume_alerts = {}
        
    def process_realtime_data(self, channel: str, data: Dict[str, Any], timestamp: str):
        """处理实时数据"""
        symbol = data['symbol']
        
        # 更新数据缓冲区
        self.data_buffer[symbol] = {
            'data': data,
            'timestamp': timestamp,
            'received_time': datetime.now()
        }
        
        # 价格变化分析
        self.analyze_price_change(symbol, data)
        
        # 成交量分析
        self.analyze_volume_change(symbol, data)
        
        # 打印实时信息
        self.print_realtime_info(symbol, data)
    
    def analyze_price_change(self, symbol: str, data: Dict[str, Any]):
        """分析价格变化"""
        price = data['price']
        change_pct = data['change_pct']
        
        # 价格告警
        if abs(change_pct) >= 5:
            alert_type = "涨停" if change_pct > 0 else "跌停"
            logger.warning(f"🚨 {symbol} {alert_type}告警: {change_pct:+.2f}%")
        elif abs(change_pct) >= 3:
            alert_type = "大涨" if change_pct > 0 else "大跌"
            logger.info(f"📈 {symbol} {alert_type}: {change_pct:+.2f}%")
    
    def analyze_volume_change(self, symbol: str, data: Dict[str, Any]):
        """分析成交量变化"""
        volume = data.get('volume', 0)
        
        # 简单的成交量异常检测
        if symbol in self.data_buffer:
            prev_data = self.data_buffer[symbol]['data']
            prev_volume = prev_data.get('volume', 0)
            
            if prev_volume > 0:
                volume_change = (volume - prev_volume) / prev_volume
                if volume_change > 0.5:  # 成交量增长50%以上
                    logger.info(f"📊 {symbol} 成交量激增: {volume_change:+.1%}")
    
    def print_realtime_info(self, symbol: str, data: Dict[str, Any]):
        """打印实时信息"""
        price = data['price']
        change = data['change']
        change_pct = data['change_pct']
        volume = data.get('volume', 0)
        
        # 颜色标识
        color_symbol = "🔴" if change > 0 else "🟢" if change < 0 else "⚪"
        
        print(f"{color_symbol} {symbol}: {price:.2f} "
              f"({change:+.2f}, {change_pct:+.2f}%) "
              f"成交量: {volume:,}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        if not self.data_buffer:
            return {'message': '暂无数据'}
        
        # 统计涨跌情况
        rising = sum(1 for data in self.data_buffer.values() if data['data']['change'] > 0)
        falling = sum(1 for data in self.data_buffer.values() if data['data']['change'] < 0)
        unchanged = len(self.data_buffer) - rising - falling
        
        # 找出涨跌幅最大的股票
        max_gainer = max(self.data_buffer.items(), key=lambda x: x[1]['data']['change_pct'])
        max_loser = min(self.data_buffer.items(), key=lambda x: x[1]['data']['change_pct'])
        
        return {
            'total_stocks': len(self.data_buffer),
            'rising': rising,
            'falling': falling,
            'unchanged': unchanged,
            'max_gainer': {
                'symbol': max_gainer[0],
                'change_pct': max_gainer[1]['data']['change_pct']
            },
            'max_loser': {
                'symbol': max_loser[0],
                'change_pct': max_loser[1]['data']['change_pct']
            }
        }


def demo_basic_websocket():
    """基础WebSocket使用演示"""
    print("🌐 基础WebSocket实时数据演示")
    print("=" * 50)
    
    # 创建客户端
    ws_url = "ws://localhost:8000/ws/realtime"
    token = "your_token_here"  # 需要替换为实际token
    
    client = RealtimeDataClient(ws_url, token)
    processor = RealtimeDataProcessor()
    
    # 设置回调函数
    client.on_data_callback = processor.process_realtime_data
    client.on_connect_callback = lambda: print("✅ 连接成功！")
    client.on_disconnect_callback = lambda code, msg: print(f"❌ 连接断开: {code} - {msg}")
    client.on_error_callback = lambda error: print(f"❌ 发生错误: {error}")
    
    try:
        # 建立连接
        client.connect()
        
        # 等待连接建立
        time.sleep(3)
        
        if not client.connected:
            print("❌ 连接失败，请检查服务器状态和token")
            return
        
        # 订阅股票
        symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH"]
        client.subscribe(symbols)
        
        print(f"\n📡 已订阅 {len(symbols)} 只股票，开始接收实时数据...")
        print("按 Ctrl+C 停止")
        
        # 接收数据
        start_time = time.time()
        while time.time() - start_time < 60:  # 运行1分钟
            time.sleep(5)
            
            # 显示统计信息
            stats = client.get_stats()
            summary = processor.get_summary()
            
            print(f"\n📊 统计信息:")
            print(f"  连接状态: {'✅' if stats['connected'] else '❌'}")
            print(f"  接收消息: {stats['messages_received']}")
            print(f"  队列大小: {stats['queue_size']}")
            
            if summary.get('total_stocks', 0) > 0:
                print(f"  股票总数: {summary['total_stocks']}")
                print(f"  上涨: {summary['rising']}, 下跌: {summary['falling']}, 平盘: {summary['unchanged']}")
                print(f"  最大涨幅: {summary['max_gainer']['symbol']} {summary['max_gainer']['change_pct']:+.2f}%")
                print(f"  最大跌幅: {summary['max_loser']['symbol']} {summary['max_loser']['change_pct']:+.2f}%")
    
    except KeyboardInterrupt:
        print("\n👋 用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        client.close()


def demo_advanced_websocket():
    """高级WebSocket功能演示"""
    print("\n🚀 高级WebSocket功能演示")
    print("=" * 40)
    
    class AdvancedProcessor(RealtimeDataProcessor):
        """高级数据处理器"""
        
        def __init__(self):
            super().__init__()
            self.price_history = {}
            self.alerts_sent = set()
        
        def process_realtime_data(self, channel: str, data: Dict[str, Any], timestamp: str):
            """高级数据处理"""
            symbol = data['symbol']
            
            # 记录价格历史
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append({
                'price': data['price'],
                'timestamp': timestamp,
                'volume': data.get('volume', 0)
            })
            
            # 保持最近50个数据点
            if len(self.price_history[symbol]) > 50:
                self.price_history[symbol] = self.price_history[symbol][-50:]
            
            # 技术分析
            self.technical_analysis(symbol, data)
            
            # 调用父类方法
            super().process_realtime_data(channel, data, timestamp)
        
        def technical_analysis(self, symbol: str, data: Dict[str, Any]):
            """简单技术分析"""
            if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
                return
            
            prices = [item['price'] for item in self.price_history[symbol]]
            
            # 计算简单移动平均
            ma5 = sum(prices[-5:]) / 5
            ma10 = sum(prices[-10:]) / 10
            current_price = prices[-1]
            
            # 金叉/死叉检测
            if len(prices) >= 11:
                prev_ma5 = sum(prices[-6:-1]) / 5
                prev_ma10 = sum(prices[-11:-1]) / 10
                
                # 金叉
                if ma5 > ma10 and prev_ma5 <= prev_ma10:
                    alert_key = f"{symbol}_golden_cross"
                    if alert_key not in self.alerts_sent:
                        logger.warning(f"🌟 {symbol} 金叉信号: MA5({ma5:.2f}) > MA10({ma10:.2f})")
                        self.alerts_sent.add(alert_key)
                
                # 死叉
                elif ma5 < ma10 and prev_ma5 >= prev_ma10:
                    alert_key = f"{symbol}_death_cross"
                    if alert_key not in self.alerts_sent:
                        logger.warning(f"💀 {symbol} 死叉信号: MA5({ma5:.2f}) < MA10({ma10:.2f})")
                        self.alerts_sent.add(alert_key)
            
            # 突破检测
            if current_price > ma10 * 1.02:  # 突破MA10 2%
                alert_key = f"{symbol}_breakout"
                if alert_key not in self.alerts_sent:
                    logger.info(f"📈 {symbol} 突破MA10: {current_price:.2f} > {ma10:.2f}")
                    self.alerts_sent.add(alert_key)
    
    # 使用高级处理器
    ws_url = "ws://localhost:8000/ws/realtime"
    token = "your_token_here"
    
    client = RealtimeDataClient(ws_url, token)
    processor = AdvancedProcessor()
    
    client.on_data_callback = processor.process_realtime_data
    
    try:
        client.connect()
        time.sleep(3)
        
        if client.connected:
            # 订阅更多股票
            symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH", 
                      "000858.SZ", "002415.SZ", "300059.SZ", "600519.SH"]
            client.subscribe(symbols)
            
            print(f"📡 订阅了 {len(symbols)} 只股票，启用高级分析...")
            
            # 运行更长时间以观察技术信号
            time.sleep(30)
        
    except Exception as e:
        print(f"❌ 高级演示失败: {e}")
    finally:
        client.close()


def demo_data_recording():
    """数据记录演示"""
    print("\n💾 数据记录演示")
    print("=" * 30)
    
    import csv
    from datetime import datetime
    
    class DataRecorder:
        """数据记录器"""
        
        def __init__(self, filename: str):
            self.filename = filename
            self.file = open(filename, 'w', newline='', encoding='utf-8')
            self.writer = csv.writer(self.file)
            
            # 写入表头
            self.writer.writerow([
                'timestamp', 'symbol', 'price', 'change', 'change_pct', 
                'volume', 'amount', 'received_time'
            ])
            self.file.flush()
        
        def record_data(self, channel: str, data: Dict[str, Any], timestamp: str):
            """记录数据"""
            self.writer.writerow([
                timestamp,
                data['symbol'],
                data['price'],
                data['change'],
                data['change_pct'],
                data.get('volume', 0),
                data.get('amount', 0),
                datetime.now().isoformat()
            ])
            self.file.flush()
            
            print(f"💾 记录数据: {data['symbol']} - {data['price']}")
        
        def close(self):
            """关闭文件"""
            self.file.close()
    
    # 创建记录器
    filename = f"realtime_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    recorder = DataRecorder(filename)
    
    ws_url = "ws://localhost:8000/ws/realtime"
    token = "your_token_here"
    
    client = RealtimeDataClient(ws_url, token)
    client.on_data_callback = recorder.record_data
    
    try:
        client.connect()
        time.sleep(3)
        
        if client.connected:
            client.subscribe(["000001.SZ", "000002.SZ"])
            print(f"📝 开始记录数据到文件: {filename}")
            time.sleep(20)  # 记录20秒
            
    except Exception as e:
        print(f"❌ 记录失败: {e}")
    finally:
        recorder.close()
        client.close()
        print(f"✅ 数据已保存到: {filename}")


if __name__ == "__main__":
    try:
        # 基础WebSocket演示
        demo_basic_websocket()
        
        # 高级功能演示
        demo_advanced_websocket()
        
        # 数据记录演示
        demo_data_recording()
        
    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()