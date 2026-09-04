"""
Agente 2 — Analista de Sentimiento

Responsabilidad única: traer noticias recientes de un ticker (NewsAPI) y
usar Claude para clasificar el sentimiento del mercado hacia ese activo,
con una justificación en lenguaje natural.

Este agente NO sabe nada de indicadores técnicos, riesgo de portafolio,
ni ejecución. Solo emite un AgentVote estandarizado.
"""

import os
from typing import Literal

import anthropic
import requests
from pydantic import BaseModel, Field

from core.schemas import AgentVote, Signal

NEWSAPI_URL = "https://newsapi.org/v2/everything"


class NewsSentimentAnalysis(BaseModel):
    """Esquema que le pedimos a Claude que devuelva (structured output)."""
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Explicación en lenguaje natural, citando las noticias relevantes")


class SentimentAgent:
    AGENT_NAME = "sentiment_agent"
    CLAUDE_MODEL = "claude-sonnet-4-6"

    def __init__(self, max_articles: int = 10):
        newsapi_key = os.getenv("NEWSAPI_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        if not newsapi_key:
            raise ValueError(
                "Falta NEWSAPI_KEY en el archivo .env. "
                "Obtén una gratis en https://newsapi.org/"
            )
        if not anthropic_key:
            raise ValueError(
                "Falta ANTHROPIC_API_KEY en el archivo .env. "
                "Obtén una en https://console.anthropic.com/"
            )

        self.newsapi_key = newsapi_key
        self.claude = anthropic.Anthropic(api_key=anthropic_key)
        self.max_articles = max_articles

    def _fetch_news(self, symbol: str) -> list[dict]:
        """
        Trae los artículos más recientes y relevantes sobre el ticker.
        Usamos el endpoint 'everything' de NewsAPI, ordenado por relevancia.
        """
        params = {
            "qInTitle": symbol,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": self.max_articles,
            "apiKey": self.newsapi_key,
        }
        response = requests.get(NEWSAPI_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        articles = data.get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description") or "",
                "source": (a.get("source") or {}).get("name", "desconocida"),
                "published_at": a.get("publishedAt", ""),
            }
            for a in articles
            if a.get("title")
        ]

    def _classify_sentiment(self, symbol: str, articles: list[dict]) -> NewsSentimentAnalysis:
        """Le pasa las noticias a Claude y le pide una clasificación estructurada."""
        headlines_block = "\n".join(
            f"- [{a['source']}, {a['published_at']}] {a['title']}: {a['description']}"
            for a in articles
        )

        prompt = f"""Eres un analista financiero. Analiza el sentimiento de mercado hacia
el ticker {symbol} a partir de estas noticias recientes:

{headlines_block}

Clasifica el sentimiento general como positive, negative o neutral, indica
tu nivel de confianza (0 a 1), y explica tu razonamiento citando las
noticias más relevantes.

Formato del campo reasoning: un punto por línea (separado por saltos de
línea reales, "\\n"), no un párrafo corrido. Cada línea debe cubrir una
noticia o factor distinto, en 1-2 oraciones cortas."""

        response = self.claude.messages.parse(
            model=self.CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=NewsSentimentAnalysis,
        )
        return response.parsed_output

    def analyze(self, symbol: str) -> AgentVote:
        """
        Punto de entrada principal: trae noticias, las clasifica con Claude,
        y devuelve un AgentVote con la señal, confianza y razonamiento.
        """
        articles = self._fetch_news(symbol)

        if not articles:
            return AgentVote(
                agent_name=self.AGENT_NAME,
                symbol=symbol,
                signal=Signal.HOLD,
                confidence=0.0,
                reasoning=f"No se encontraron noticias recientes sobre {symbol}. Sin datos suficientes para opinar.",
                raw_metrics={"articles_found": 0},
            )

        analysis = self._classify_sentiment(symbol, articles)

        signal_map = {
            "positive": Signal.BUY,
            "negative": Signal.SELL,
            "neutral": Signal.HOLD,
        }

        return AgentVote(
            agent_name=self.AGENT_NAME,
            symbol=symbol,
            signal=signal_map[analysis.sentiment],
            confidence=round(analysis.confidence, 2),
            reasoning=analysis.reasoning,
            raw_metrics={
                "sentiment": analysis.sentiment,
                "articles_analyzed": len(articles),
            },
        )
