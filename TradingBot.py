
# DAYTRADE CRYPTO

import pandas as pd
import numpy as np
import time
import ccxt
import datetime as dt
import logging

# Configurações iniciais
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EXCHANGE = ccxt.binance()
SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "BNB/USDT", "SOL/USDT", "DOGE/USDT","PEPE/USDT",
    "TRX/USDT", "ADA/USDT", "WBTC/USDT",
    "SUI/USDT", "XLM/USDT", "LINK/USDT", "HBAR/USDT",
    "BCH/USDT", "AVAX/USDT", "MATIC/USDT", "ARB/USDT",
    "AAVE/USDT", "LTC/USDT", "FIL/USDT", "XMR/USDT", "TON/USDT",
    "NEAR/USDT", "SAND/USDT", "DOT/USDT", "UNI/USDT", "XEC/USDT", "CHZ/USDT",
    "QNT/USDT", "ALGO/USDT", "ICP/USDT", "EOS/USDT", "ZEC/USDT", "MANA/USDT",
    "THETA/USDT", "AXS/USDT", "KLAY/USDT", "FLOW/USDT", "OKB/USDT",
    "CRV/USDT", "XDC/USDT", "BAT/USDT", "ZIL/USDT", "KSM/USDT", "DASH/USDT","WIF/USDT","TRUMP/USDT"]
TIMEFRAME = "1h" #Intervalo de tempo para análise
EMA_PERIODS = [9, 21, 50]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ENABLE_LOOP = True  # Defina True para rodar em loop
SLEEP_TIME = 60  # segundos entre iterações
# Funções de indicadores
def calculate_indicators(df):
    for p in EMA_PERIODS:
        df[f'EMA_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(RSI_PERIOD).mean()
    avg_loss = pd.Series(loss).rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['Signal'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    return df

# Função de análise
def analyze_symbol(symbol):
    try:
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = calculate_indicators(df)
        last = df.iloc[-1]
        signal = 'HOLD'
        reasons = []
        if last['EMA_9'] > last['EMA_21'] and last['RSI'] < 70 and last['MACD'] > last['Signal']:
            signal = 'BUY'
            reasons.append('Tendência de alta confirmada')
        elif last['EMA_9'] < last['EMA_21'] and last['RSI'] > 30 and last['MACD'] < last['Signal']:
            signal = 'SELL'
            reasons.append('Tendência de baixa confirmada')
        return {
            'symbol': symbol,
            'price': last['close'],
            'signal': signal,
            'reasons': reasons
        }
    except Exception as e:
        logging.error(f"Erro analisando {symbol}: {e}")
        return None

# Execução única ou em loop
def run_analysis():
    while True:
        for s in SYMBOLS:
            result = analyze_symbol(s)
            if result:
                logging.info(f"{result['symbol']} | {result['signal']} | {result['price']:.2f} | {', '.join(result['reasons'])}")
        if not ENABLE_LOOP:
            break
        time.sleep(SLEEP_TIME)

# Execute as outras 2 codigos
# Executar análise
run_analysis()
