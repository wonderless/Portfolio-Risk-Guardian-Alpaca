"""
Backtest de breadth + gestión de riesgo dinámica (ATR sizing + stops)

Sobre la misma cesta de 16 tickers de `run_breadth_backtest.py`, agrega las
dos mejoras de mayor prioridad según la investigación (ver README):
dimensionamiento de posición por volatilidad y stops dinámicos, ambos
basados en ATR(14). Ver `core/backtester.run_technical_backtest_atr` para
el detalle de la simulación.

Uso:
    python run_breadth_atr_backtest.py
"""

from core.alpaca_client import AlpacaClient
from agents.technical_agent import TechnicalAgent
from core.backtester import run_basket_backtest_atr

LOOKBACK_DAYS = 1825  # ~5 años, igual que run_breadth_backtest.py

UNIVERSE = {
    "Tecnología": ["AAPL", "MSFT"],
    "Consumo discrecional": ["TSLA"],
    "Consumo defensivo": ["KO", "PG", "WMT"],
    "Energía": ["XOM", "CVX"],
    "Salud": ["JNJ", "UNH"],
    "Industrial": ["CAT", "HON"],
    "Financiero": ["JPM", "BAC"],
    "Utilities": ["NEE"],
    "Comunicaciones": ["DIS"],
}


def main():
    client = AlpacaClient()
    agent = TechnicalAgent(alpaca_client=client)  # parámetros de señal sin tocar

    tickers = [t for group in UNIVERSE.values() for t in group]

    print("=" * 70)
    print("Backtest de breadth + ATR sizing/stops")
    print("=" * 70)
    print(f"Cesta: {len(tickers)} tickers en {len(UNIVERSE)} sectores. Ventana: ~{LOOKBACK_DAYS // 365} años.")
    print("Sizing: target_daily_vol=1% por trade, tope 3x. Stop: 2x ATR(14) desde la entrada, fijo (no trailing).")

    bars_by_symbol = {}
    for symbol in tickers:
        try:
            bars_by_symbol[symbol] = client.get_recent_bars(symbol, lookback_days=LOOKBACK_DAYS)
        except Exception as e:
            print(f"  AVISO: no se pudo traer historial de {symbol}: {e}")

    result = run_basket_backtest_atr(agent, bars_by_symbol)

    print(f"\nPeríodo combinado: {result.start_date} -> {result.end_date} ({result.num_days} días)")
    print(f"Tickers incluidos:  {result.num_tickers}/{len(tickers)}: {', '.join(result.tickers)}")
    print(f"\nRetorno del portafolio (estrategia + ATR):  {result.portfolio_total_return:+.1%}")
    print(f"Retorno de la cesta (buy & hold, equal-weight): {result.basket_buy_and_hold_return:+.1%}")
    print(f"Máximo drawdown:    {result.portfolio_max_drawdown:.1%}")
    print(f"Sharpe / Sortino:   {result.sharpe_ratio:.2f} / {result.sortino_ratio:.2f}")
    print(f"Calmar:             {result.calmar_ratio:.2f}")
    print(f"Win rate:           {result.win_rate:.1%}")
    print(f"Profit factor:      {result.profit_factor:.2f}")


if __name__ == "__main__":
    main()
