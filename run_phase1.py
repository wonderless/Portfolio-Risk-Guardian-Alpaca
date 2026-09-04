"""
Script de prueba de la Fase 1: conexión a Alpaca + Agente Técnico.

Uso:
    1. Copia .env.example a .env y agrega tus credenciales de Alpaca Paper Trading
    2. pip install -r requirements.txt
    3. python run_phase1.py
"""

from core.alpaca_client import AlpacaClient
from agents.technical_agent import TechnicalAgent


def main():
    print("=" * 60)
    print("PORTFOLIO RISK GUARDIAN — Fase 1: Agente Técnico")
    print("=" * 60)

    client = AlpacaClient()

    # Verificar conexión mostrando info de la cuenta paper
    account = client.get_account_info()
    print(f"\nOK - Conectado a Alpaca Paper Trading")
    print(f"  Cash disponible: ${account['cash']:,.2f}")
    print(f"  Equity total:    ${account['equity']:,.2f}")

    # Correr el agente técnico sobre algunos tickers de prueba
    tickers = ["AAPL", "MSFT", "TSLA"]
    agent = TechnicalAgent(alpaca_client=client)

    for symbol in tickers:
        print(f"\n{'-' * 60}")
        print(f"Analizando {symbol}...")
        try:
            vote = agent.analyze(symbol)
            print(f"  Señal:      {vote.signal.value.upper()}")
            print(f"  Confianza:  {vote.confidence:.0%}")
            print(f"  Razonamiento: {vote.reasoning}")
            print(f"  Métricas:   {vote.raw_metrics}")
        except Exception as e:
            print(f"  AVISO: Error analizando {symbol}: {e}")


if __name__ == "__main__":
    main()
