"""
Script de prueba de la Fase 2: Agente de Sentimiento + Agente de Riesgo,
corriendo junto al Agente Técnico (Fase 1) para ver los tres votos lado a
lado. Todavía no hay lógica de consenso ni ejecución — eso es la Fase 3.

Uso:
    1. Completa .env con ALPACA_*, ANTHROPIC_API_KEY y NEWSAPI_KEY
    2. pip install -r requirements.txt
    3. python run_phase2.py
"""

from core.alpaca_client import AlpacaClient
from agents.technical_agent import TechnicalAgent
from agents.sentiment_agent import SentimentAgent
from agents.risk_agent import RiskAgent


def print_vote(vote):
    print(f"  [{vote.agent_name}]")
    print(f"    Señal:        {vote.signal.value.upper()}")
    print(f"    Confianza:    {vote.confidence:.0%}")
    print(f"    Razonamiento: {vote.reasoning}")
    print(f"    Métricas:     {vote.raw_metrics}")


def main():
    print("=" * 60)
    print("PORTFOLIO RISK GUARDIAN — Fase 2: Sentimiento + Riesgo")
    print("=" * 60)

    client = AlpacaClient()
    account = client.get_account_info()
    print(f"\nOK - Conectado a Alpaca Paper Trading")
    print(f"  Cash disponible: ${account['cash']:,.2f}")
    print(f"  Equity total:    ${account['equity']:,.2f}")

    technical_agent = TechnicalAgent(alpaca_client=client)
    sentiment_agent = SentimentAgent()
    risk_agent = RiskAgent(alpaca_client=client)

    tickers = ["AAPL", "MSFT", "TSLA"]

    for symbol in tickers:
        print(f"\n{'-' * 60}")
        print(f"Analizando {symbol}...")

        for agent in (technical_agent, sentiment_agent, risk_agent):
            try:
                vote = agent.analyze(symbol)
                print_vote(vote)
            except Exception as e:
                print(f"  AVISO: [{agent.AGENT_NAME}] Error analizando {symbol}: {e}")


if __name__ == "__main__":
    main()
