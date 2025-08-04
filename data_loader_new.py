import os
import struct
import pandas as pd
from datetime import datetime

def get_daily_data(file_path):
    """从.day文件读取完整的日线数据"""
    data = []
    record_size = 32
    unpack_format = '<IIIIIfI'
    # 计算解包格式的字节大小
    unpack_size = struct.calcsize(unpack_format)
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(record_size)
            if len(chunk) < record_size: break
            try:
                # 只解包格式定义的部分
                date, open_p, high_p, low_p, close_p, amount, volume = struct.unpack(unpack_format, chunk[:unpack_size])
                # 价格要除以100
                open_p, high_p, low_p, close_p = open_p / 100, high_p / 100, low_p / 100, close_p / 100
                if open_p <= 0: continue
                data.append({
                    'date': datetime.strptime(str(date), '%Y%m%d'), 'open': open_p, 'high': high_p,
                    'low': low_p, 'close': close_p, 'volume': volume
                })
            except (struct.error, ValueError): continue
    if not data: return None
    return pd.DataFrame(data).sort_values('date').reset_index(drop=True)

def get_5min_data(file_path):
    """
    从.lc5文件读取5分钟线数据
    文件格式说明: 每32字节一条记录
    - 2字节: 日期 (ushort), (year - 2004) * 2048 + month * 100 + day
    - 2字节: 时间 (ushort), hour * 60 + minute
    - 4字节: open (float)
    - 4字节: high (float)
    - 4字节: low (float)
    - 4字节: close (float)
    - 4字节: volume (float)
    - 4字节: amount (float)
    - 4字节: (保留)
    """
    data = []
    record_size = 32
    # 定义解包格式：2个unsigned short, 6个float
    unpack_format = '<HHffffff'
    unpack_size = struct.calcsize(unpack_format)

    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(record_size)
            if len(chunk) < record_size:
                break
            try:
                packed_date, packed_time, open_p, high_p, low_p, close_p, volume, amount = struct.unpack(unpack_format, chunk[:unpack_size])

                # 解码日期
                year = packed_date // 2048 + 2004
                month = (packed_date % 2048) // 100
                day = packed_date % 100

                # 解码时间
                hour = packed_time // 60
                minute = packed_time % 60
                
                # 合成datetime对象
                dt = datetime(year, month, day, hour, minute)
                
                if open_p <= 0: continue

                data.append({
                    'datetime': dt,
                    'open': open_p,
                    'high': high_p,
                    'low': low_p,
                    'close': close_p,
                    'volume': volume,
                    'amount': amount
                })
            except (struct.error, ValueError):
                continue
    
    if not data:
        return None
        
    return pd.DataFrame(data).sort_values('datetime').reset_index(drop=True)

def get_multi_timeframe_data(stock_code, base_path=None):
    """获取多周期数据（日线 + 5分钟线）"""
    if base_path is None:
        base_path = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/vipdoc")

    market = stock_code[:2]

    # 构建文件路径
    daily_file = os.path.join(base_path, market, 'lday', f'{stock_code}.day')
    min5_file = os.path.join(base_path, market, 'fzline', f'{stock_code}.lc5')

    result = {
        'stock_code': stock_code,
        'daily_data': None,
        'min5_data': None,
        'data_status': {
            'daily_available': False,
            'min5_available': False
        }
    }

    # 加载日线数据
    if os.path.exists(daily_file):
        try:
            result['daily_data'] = get_daily_data(daily_file)
            result['data_status']['daily_available'] = result['daily_data'] is not None
        except Exception as e:
            print(f"加载日线数据失败 {stock_code}: {e}")

    # 加载5分钟线数据
    if os.path.exists(min5_file):
        try:
            result['min5_data'] = get_5min_data(min5_file)
            result['data_status']['min5_available'] = result['min5_data'] is not None
        except Exception as e:
            print(f"加载5分钟线数据失败 {stock_code}: {e}")

    return result

def resample_5min_to_other_timeframes(df_5min):
    """将5分钟数据重采样为其他时间周期"""
    if df_5min is None or df_5min.empty:
        return {}

    # 设置datetime为索引
    df_5min = df_5min.set_index('datetime')

    timeframes = {}

    try:
        # 15分钟线
        timeframes['15min'] = df_5min.resample('15T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'
        }).dropna()

        # 30分钟线
        timeframes['30min'] = df_5min.resample('30T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'
        }).dropna()

        # 60分钟线
        timeframes['60min'] = df_5min.resample('60T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'
        }).dropna()

        # 重置索引并添加日期时间列
        for tf_name, tf_data in timeframes.items():
            tf_data.reset_index(inplace=True)
            tf_data['date'] = tf_data['datetime'].dt.date
            tf_data['time'] = tf_data['datetime'].dt.time

    except Exception as e:
        print(f"重采样数据失败: {e}")
        return {}

    return timeframes