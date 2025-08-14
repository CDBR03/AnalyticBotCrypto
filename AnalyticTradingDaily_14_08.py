# DAYTRADE CRYPTO — revisão e otimização

import pandas as pd
import numpy as np
import time
import ccxt
import logging
from collections import OrderedDict

# ========== Configurações iniciais ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

EXCHANGE = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "spot"}  # mude para "future" se usar futuros
})

# Remover duplicados preservando a ordem
SYMBOLS = list(OrderedDict.fromkeys([
    "BTC/USDT", "ETH/USDT", "XRP/USDT", "BNB/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT",
    "TRX/USDT", "ADA/USDT", "WBTC/USDT", "SUI/USDT", "XLM/USDT", "HBAR/USDT",
    "BCH/USDT", "AVAX/USDT", "MATIC/USDT", "ARB/USDT", "LTC/USDT", "FIL/USDT",
    "XMR/USDT", "TON/USDT", "NEAR/USDT", "SAND/USDT", "DOT/USDT", "UNI/USDT",
    "XEC/USDT", "CHZ/USDT", "QNT/USDT", "ALGO/USDT", "ICP/USDT", "EOS/USDT",
    "ZEC/USDT", "MANA/USDT", "THETA/USDT", "AXS/USDT", "KLAY/USDT", "FLOW/USDT",
    "CRV/USDT", "BAT/USDT", "ZIL/USDT", "KSM/USDT", "DASH/USDT",
    "WIF/USDT", "TRUMP/USDT", "LINK/USDT", "BFUSD/USDT", "BNSOL/USDT",
    "BANANAS31/USDT", "ME/USDT", "RAY/USDT", "VIRTUAL/USDT", "BOME/USDT", "STRK/USDT",
    "AAVE/USDT"
]))

# >>> ATENÇÃO:
# Binance nem sempre possui todos os símbolos acima no par /USDT spot.
# Se algum não existir, o script só irá logar o erro e seguir.

TIMEFRAME = "1d"              # Ex.: "1m","5m","15m","1h","4h","1d","1w","1M"
EMA_PERIODS = [9, 21, 50]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ENABLE_LOOP = True
SLEEP_TIME = 60               # segundos entre iterações

# Modo de cálculo do preço semanal ideal:
#   "low"  -> mínima da última semana FECHADA (suporte)
#   "fib"  -> retração 0.618 entre low e close da última semana FECHADA
WEEKLY_PRICE_MODE = "low"     # "low" (padrão) ou "fib"

# ========== Utilidades ==========

def safe_fetch_ohlcv(symbol: str, timeframe: str, limit: int = 100, retries: int = 3, wait: float = 1.0):
    """
    Faz fetch_ohlcv com retentativas simples.
    """
    last_err = None
    for i in range(retries):
        try:
            return EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            last_err = e
            time.sleep(wait * (i + 1))
    logging.error(f"Erro ao buscar OHLCV para {symbol} {timeframe}: {last_err}")
    return None

def validate_timeframe(tf: str):
    """
    Valida se o timeframe é suportado pela exchange (quando disponível).
    Caso não seja possível validar, apenas segue.
    """
    try:
        tfs = getattr(EXCHANGE, "timeframes", None)
        if isinstance(tfs, dict) and len(tfs) > 0:
            if tf not in tfs:
                raise ValueError(f"TIMEFRAME '{tf}' não suportado por esta exchange. Suportados: {list(tfs.keys())}")
    except Exception as e:
        logging.warning(f"Não foi possível validar TIMEFRAME automaticamente: {e}")

# ========== Indicadores ==========

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula EMAs, RSI (média simples) e MACD.
    """
    close = df["close"]

    # EMAs
    for p in EMA_PERIODS:
        df[f"EMA_{p}"] = close.ewm(span=p, adjust=False).mean()

    # RSI (cálculo simples)
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain, index=df.index).rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["Signal"] = df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    return df

# ========== Lógica Semanal ==========

def get_weekly_buy_price(symbol: str, mode: str = WEEKLY_PRICE_MODE):
    """
    Retorna o preço ideal de compra com base no gráfico semanal.
    Usa SEMPRE a última semana FECHADA (evita semana em formação).
      - mode="low": mínima da semana passada
      - mode="fib": retração 0.618 entre low e close da semana passada
    """
    ohlcv = safe_fetch_ohlcv(symbol, timeframe="1w", limit=3)
    if not ohlcv:
        return None

    dfw = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if len(dfw) < 2:
        return None

    # última semana FECHADA -> penúltima vela ([-2])
    last_closed = dfw.iloc[-2]
    low = float(last_closed["low"])
    close = float(last_closed["close"])

    if mode == "fib":
        # retração 0.618 entre close (topo da zona) e low (fundo)
        # entry = low + 0.618*(close - low)
        return low + 0.618 * (close - low)
    # padrão: suporte (mínima)
    return low

# ========== Análise por símbolo ==========

def analyze_symbol(symbol: str, timeframe: str):
    ohlcv = safe_fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
    if not ohlcv:
        return None

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = calculate_indicators(df)
    last = df.iloc[-1]

    # Sinal
    signal = "HOLD"
    reasons = []
    try:
        if (last["EMA_9"] > last["EMA_21"]) and (last["RSI"] < 70) and (last["MACD"] > last["Signal"]):
            signal = "BUY"
            reasons.append("Tendência de alta confirmada (EMA9>EMA21, RSI<70, MACD>Signal)")
        elif (last["EMA_9"] < last["EMA_21"]) and (last["RSI"] > 30) and (last["MACD"] < last["Signal"]):
            signal = "SELL"
            reasons.append("Tendência de baixa confirmada (EMA9<EMA21, RSI>30, MACD<Signal)")
    except Exception as e:
        logging.error(f"Erro na lógica de sinal para {symbol}: {e}")

    # Preço ideal semanal
    weekly_buy_price = get_weekly_buy_price(symbol, mode=WEEKLY_PRICE_MODE)

    return {
        "symbol": symbol,
        "price": float(last["close"]),
        "signal": signal,
        "rsi": float(last["RSI"]) if pd.notna(last["RSI"]) else None,
        "reasons": reasons,
        "weekly_buy_price": float(weekly_buy_price) if weekly_buy_price is not None else None,
    }

# ========== Execução principal ==========

def run_analysis():
    validate_timeframe(TIMEFRAME)

    while True:
        buy_list = []
        sell_list = []

        for s in SYMBOLS:
            result = analyze_symbol(s, TIMEFRAME)
            if not result:
                continue

            if result["signal"] == "BUY":
                buy_list.append(result)
            elif result["signal"] == "SELL":
                sell_list.append(result)

        # Ordenações (BUY: RSI menor primeiro | SELL: RSI maior primeiro)
        buy_list = [r for r in buy_list if r["rsi"] is not None]
        sell_list = [r for r in sell_list if r["rsi"] is not None]
        buy_list.sort(key=lambda x: x["rsi"])
        sell_list.sort(key=lambda x: x["rsi"], reverse=True)

        # Exibição
        if buy_list:
            logging.info("\n=== MOEDAS PARA COMPRA (BUY) ===")
            for r in buy_list:
                ideal = f"{r['weekly_buy_price']:.4f}" if r["weekly_buy_price"] is not None else "-"
                logging.info(f"{r['symbol']} | Preço: {r['price']:.4f} | RSI: {r['rsi']:.2f} | Ideal(1w:{WEEKLY_PRICE_MODE}): {ideal} | {', '.join(r['reasons'])}")

        if sell_list:
            logging.info("\n=== MOEDAS PARA VENDA (SELL) ===")
            for r in sell_list:
                ideal = f"{r['weekly_buy_price']:.4f}" if r["weekly_buy_price"] is not None else "-"
                logging.info(f"{r['symbol']} | Preço: {r['price']:.4f} | RSI: {r['rsi']:.2f} | Ideal(1w:{WEEKLY_PRICE_MODE}): {ideal} | {', '.join(r['reasons'])}")

        if not ENABLE_LOOP:
            break
        time.sleep(SLEEP_TIME)

# Executar
if __name__ == "__main__":
    run_analysis()
