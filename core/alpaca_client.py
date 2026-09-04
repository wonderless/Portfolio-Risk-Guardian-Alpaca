"""
Cliente central de Alpaca: envuelve tanto el acceso a datos de mercado
(histórico, para calcular indicadores) como el trading (paper trading).

Todos los agentes usan este mismo cliente para no duplicar lógica de
conexión ni credenciales.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv

load_dotenv()


class AlpacaClient:
    """Envoltorio simple sobre alpaca-py para datos + trading (paper)."""

    def __init__(self):
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "Faltan ALPACA_API_KEY / ALPACA_SECRET_KEY en el archivo .env. "
                "Copia .env.example a .env y completa tus credenciales de paper trading."
            )

        # Cliente de datos históricos (velas OHLCV)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

        # Cliente de trading (paper=True es crítico: dinero simulado, no real)
        self.trading_client = TradingClient(api_key, secret_key, paper=True)

    def get_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Trae velas diarias entre `start` y `end` para un ticker, ajustadas
        por splits y dividendos (si no, un split como el de WMT en feb-2024
        aparece como una caída de precio de un día para otro que nunca
        ocurrió, y arruina cualquier cálculo de retornos o indicadores).
        Devuelve un DataFrame ordenado por fecha ascendente con columnas
        open, high, low, close, volume.
        """
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment=Adjustment.ALL,
        )
        bars = self.data_client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            raise ValueError(f"No se encontraron datos para el ticker '{symbol}'.")

        # El índice viene multi-nivel (symbol, timestamp); nos quedamos con
        # el nivel de timestamp para un solo símbolo.
        df = df.reset_index(level=0, drop=True)
        return df

    def get_recent_bars(self, symbol: str, lookback_days: int = 60) -> pd.DataFrame:
        """Trae velas diarias de los últimos `lookback_days` días para un ticker."""
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        return self.get_bars(symbol, start, end)

    def get_account_info(self) -> dict:
        """Info básica de la cuenta paper: efectivo disponible, equity total."""
        account = self.trading_client.get_account()
        return {
            "cash": float(account.cash),
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
        }

    def get_open_positions(self) -> list[dict]:
        """Lista las posiciones abiertas actuales en la cuenta paper."""
        positions = self.trading_client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "current_price": float(p.current_price),
            }
            for p in positions
        ]

    def submit_market_order(self, symbol: str, qty: float, side: str) -> dict:
        """
        Envía una orden de mercado en la cuenta paper.
        side debe ser 'buy' o 'sell'.
        """
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading_client.submit_order(order_request)
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": str(order.side),
            "status": str(order.status),
        }
