"""
Backtest de "breadth" (Fase 5 — exploración post-diagnóstico)

Prueba la hipótesis de que la señal del Agente Técnico, aunque débil por
ticker, puede volverse una ventaja estadísticamente visible al combinarla
sobre una cesta diversificada y no correlacionada de activos (ver README,
sección "Por qué el Agente Técnico no muestra una ventaja generalizable" —
la Ley Fundamental de gestión activa: resultado ≈ calidad de la señal ×
raíz cuadrada del número de apuestas independientes).

Usa los mismos 16 tickers y el mismo período de 5 años que
`run_universe_test.py`, sin tocar ningún parámetro del agente — la
diferencia es que aquí se evalúa el resultado a nivel de PORTAFOLIO
(combinando las 16 señales, ponderadas inversamente por volatilidad) en
vez de ticker por ticker.

Uso:
    python run_breadth_backtest.py
"""

from core.alpaca_client import AlpacaClient
from agents.technical_agent import TechnicalAgent
from core.backtester import run_basket_backtest

LOOKBACK_DAYS = 1825  # ~5 años, igual que run_universe_test.py

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
    agent = TechnicalAgent(alpaca_client=client)  # parámetros por defecto, sin tocar

    tickers = [t for group in UNIVERSE.values() for t in group]

    print("=" * 70)
    print("Backtest de breadth — misma señal, cesta diversificada como portafolio")
    print("=" * 70)
    print(f"Cesta: {len(tickers)} tickers en {len(UNIVERSE)} sectores. Ventana: ~{LOOKBACK_DAYS // 365} años.")
    print("Ponderación: inversa a la volatilidad reciente de cada activo (rolling, sin mirar al futuro).")

    bars_by_symbol = {}
    for symbol in tickers:
        try:
            bars_by_symbol[symbol] = client.get_recent_bars(symbol, lookback_days=LOOKBACK_DAYS)
        except Exception as e:
            print(f"  AVISO: no se pudo traer historial de {symbol}: {e}")

    result = run_basket_backtest(agent, bars_by_symbol, allow_short=False)

    print(f"\nPeríodo combinado: {result.start_date} -> {result.end_date} ({result.num_days} días)")
    print(f"Tickers incluidos:  {result.num_tickers}/{len(tickers)}: {', '.join(result.tickers)}")
    print(f"\nRetorno del portafolio (estrategia):  {result.portfolio_total_return:+.1%}")
    print(f"Retorno de la cesta (buy & hold, mismos pesos): {result.basket_buy_and_hold_return:+.1%}")
    print(f"Máximo drawdown:    {result.portfolio_max_drawdown:.1%}")
    print(f"Sharpe / Sortino:   {result.sharpe_ratio:.2f} / {result.sortino_ratio:.2f}")
    print(f"Calmar:             {result.calmar_ratio:.2f}")
    print(f"Win rate:           {result.win_rate:.1%}")
    print(f"Profit factor:      {result.profit_factor:.2f}")


if __name__ == "__main__":
    main()
