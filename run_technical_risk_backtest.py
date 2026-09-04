"""
Backtest del consenso Técnico+Riesgo con cartera simulada real (Fase 6,
punto pendiente del README: "backtestear ... Riesgo para poder validar la
regla de consenso 2-de-3 contra alternativas").

Corre, sobre el mismo universo de 16 tickers usado en
`run_breadth_atr_backtest.py`, dos variantes lado a lado:
1. Técnico solo con ATR sizing (`run_basket_backtest_atr`, ya existente).
2. Técnico + veto del Agente de Riesgo sobre una cartera simulada
   (`run_technical_risk_backtest`, nuevo).

Uso:
    python run_technical_risk_backtest.py
"""

from agents.risk_agent import RiskAgent
from agents.technical_agent import TechnicalAgent
from core.alpaca_client import AlpacaClient
from core.backtester import run_basket_backtest_atr
from core.portfolio_backtester import run_technical_risk_backtest

LOOKBACK_DAYS = 1825  # ~5 años, igual que run_breadth_atr_backtest.py

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
    agent = TechnicalAgent(alpaca_client=client)
    risk_agent = RiskAgent(alpaca_client=client)

    tickers = [t for group in UNIVERSE.values() for t in group]

    print("=" * 70)
    print("Backtest de consenso Técnico+Riesgo (cartera simulada)")
    print("=" * 70)
    print(f"Cesta: {len(tickers)} tickers en {len(UNIVERSE)} sectores. Ventana: ~{LOOKBACK_DAYS // 365} años.")
    print(f"Riesgo: max_position_pct={risk_agent.max_position_pct:.0%}, "
          f"high_portfolio_vol_threshold={risk_agent.high_portfolio_vol_threshold:.0%}.")

    bars_by_symbol = {}
    for symbol in tickers:
        try:
            bars_by_symbol[symbol] = client.get_recent_bars(symbol, lookback_days=LOOKBACK_DAYS)
        except Exception as e:
            print(f"  AVISO: no se pudo traer historial de {symbol}: {e}")

    technical_only = run_basket_backtest_atr(agent, bars_by_symbol)
    technical_risk = run_technical_risk_backtest(agent, risk_agent, bars_by_symbol)

    print(f"\n{'Métrica':<28}{'Técnico solo (ATR)':>22}{'Técnico + Riesgo':>22}")
    print("-" * 72)
    print(f"{'Período':<28}{technical_only.start_date + ' -> ' + technical_only.end_date:>22}"
          f"{technical_risk.start_date + ' -> ' + technical_risk.end_date:>22}")
    print(f"{'Retorno total':<28}{technical_only.portfolio_total_return:>21.1%} "
          f"{technical_risk.portfolio_total_return:>21.1%}")
    print(f"{'Máximo drawdown':<28}{technical_only.portfolio_max_drawdown:>21.1%} "
          f"{technical_risk.portfolio_max_drawdown:>21.1%}")
    print(f"{'Sharpe':<28}{technical_only.sharpe_ratio:>22.2f}{technical_risk.sharpe_ratio:>22.2f}")
    print(f"{'Sortino':<28}{technical_only.sortino_ratio:>22.2f}{technical_risk.sortino_ratio:>22.2f}")
    print(f"{'Calmar':<28}{technical_only.calmar_ratio:>22.2f}{technical_risk.calmar_ratio:>22.2f}")
    print(f"{'Win rate':<28}{technical_only.win_rate:>21.1%} {technical_risk.win_rate:>21.1%}")
    print(f"{'Profit factor':<28}{technical_only.profit_factor:>22.2f}{technical_risk.profit_factor:>22.2f}")

    print(f"\nCartera simulada (Técnico + Riesgo):")
    print(f"  Trades ejecutados:        {technical_risk.num_trades}")
    print(f"  Entradas vetadas por Riesgo: {technical_risk.num_risk_vetoes}")
    print(f"  Concentración promedio por entrada: {technical_risk.avg_concentration:.1%}")


if __name__ == "__main__":
    main()
