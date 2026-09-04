"""
Dashboard — Portfolio Risk Guardian (Fase 4)

Panel en tiempo real que muestra qué "piensa" cada agente (técnico,
sentimiento, riesgo) y por qué, además de la decisión de consenso del
orquestador. La transparencia del razonamiento es el requisito central
de este dashboard: nunca se muestra solo la señal final sin el detalle
de cada voto que la originó.

Uso:
    streamlit run dashboard/app.py
"""

import os
import sys

import streamlit as st

# Al correr `streamlit run dashboard/app.py`, Python solo agrega dashboard/
# a sys.path, no la raíz del proyecto. La agregamos a mano para poder
# importar core/ y agents/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.alpaca_client import AlpacaClient
from core.schemas import AgentVote, ConsensusDecision, Signal
from agents.technical_agent import TechnicalAgent
from agents.sentiment_agent import SentimentAgent
from agents.risk_agent import RiskAgent
from agents.orchestrator import build_orchestrator_graph

st.set_page_config(page_title="Portfolio Risk Guardian", layout="wide")


def load_secrets_into_env():
    """
    `AlpacaClient`/`SentimentAgent` leen credenciales con `os.getenv(...)`
    (pensado para `.env` local). En Streamlit Community Cloud las
    credenciales se configuran en su gestor de secrets, expuesto como
    `st.secrets`, no como variables de entorno — este puente copia lo que
    haya en `st.secrets` a `os.environ` ANTES de instanciar esos clientes,
    sin pisar un `.env` local que ya las tenga.
    """
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ANTHROPIC_API_KEY", "NEWSAPI_KEY", "APP_PASSWORD"):
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])


load_secrets_into_env()


def require_password():
    """
    Gate simple por contraseña compartida (`st.secrets["APP_PASSWORD"]`).
    Necesario porque cada clic en 'Analizar' dispara llamadas reales a
    Claude y NewsAPI (con costo/cuota real) y puede colocar órdenes en la
    cuenta paper — sin esto, cualquier visitante del dashboard público
    podría gastar esas cuotas o tocar la cuenta.

    En local, sin `APP_PASSWORD` configurado en `.streamlit/secrets.toml`,
    no bloquea nada (para no romper el flujo de desarrollo).
    """
    expected = st.secrets.get("APP_PASSWORD")
    if not expected:
        return

    if st.session_state.get("authenticated"):
        return

    st.title("Portfolio Risk Guardian")
    password = st.text_input("Contraseña", type="password")
    if st.button("Entrar"):
        if password == expected:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()


AGENT_DISPLAY_NAMES = {
    "technical_agent": "Agente Técnico",
    "sentiment_agent": "Agente de Sentimiento",
    "risk_agent": "Agente de Riesgo",
}

SIGNAL_COLOR = {
    Signal.BUY: "#16a34a",
    Signal.SELL: "#dc2626",
    Signal.HOLD: "#6b7280",
}


@st.cache_resource(show_spinner="Conectando a Alpaca y preparando agentes...")
def get_resources():
    """
    Crea el cliente de Alpaca, los 3 agentes analistas y el grafo del
    orquestador una sola vez por sesión (evita reconectar en cada rerun
    de Streamlit, que ocurre en cada interacción del usuario).
    """
    client = AlpacaClient()
    technical_agent = TechnicalAgent(alpaca_client=client)
    sentiment_agent = SentimentAgent()
    risk_agent = RiskAgent(alpaca_client=client)
    graph = build_orchestrator_graph(technical_agent, sentiment_agent, risk_agent, client)
    return client, graph


def signal_badge(signal: Signal) -> str:
    color = SIGNAL_COLOR[signal]
    return (
        f'<span style="background-color:{color}; color:white; padding:2px 10px; '
        f'border-radius:12px; font-weight:600; font-size:0.85rem;">{signal.value.upper()}</span>'
    )


def format_reasoning(text: str) -> str:
    """
    Prepara el texto de razonamiento para st.markdown():
    - Escapa '$' para que precios como "$328.21" no se interpreten como LaTeX.
    - Convierte saltos de línea simples en saltos de línea de Markdown
      (que requiere dos espacios al final de la línea, si no los junta en un párrafo).
    """
    escaped = text.replace("$", r"\$")
    return escaped.replace("\n", "  \n")


def render_vote_card(vote: AgentVote):
    with st.container(border=True):
        st.markdown(f"**{AGENT_DISPLAY_NAMES.get(vote.agent_name, vote.agent_name)}**")
        st.markdown(signal_badge(vote.signal), unsafe_allow_html=True)
        st.progress(vote.confidence, text=f"Confianza: {vote.confidence:.0%}")
        st.markdown(format_reasoning(vote.reasoning))
        with st.expander("Métricas"):
            st.json(vote.raw_metrics)


def render_consensus_card(consensus: ConsensusDecision):
    with st.container(border=True):
        st.markdown("#### Decisión de consenso")
        st.markdown(signal_badge(consensus.final_signal), unsafe_allow_html=True)
        st.markdown(f"**Consenso alcanzado:** {'Sí' if consensus.consensus_reached else 'No'}")
        st.markdown(format_reasoning(consensus.explanation))
        if consensus.executed:
            st.success(f"Orden ejecutada en paper trading — order_id: {consensus.order_id}")
        elif consensus.final_signal != Signal.HOLD:
            st.info("Consenso no-HOLD alcanzado, pero no se ejecutó (auto-ejecución desactivada).")


def main():
    require_password()

    st.title("Portfolio Risk Guardian")
    st.caption(
        "Sistema multi-agente de gestión de riesgo — paper trading. "
        "Cada tarjeta muestra la señal, la confianza y el razonamiento del agente, no solo la decisión final."
    )

    with st.sidebar:
        st.header("Configuración")
        default_tickers = os.getenv("TICKERS_TO_WATCH", "AAPL,MSFT,TSLA")
        tickers_input = st.text_input("Tickers a analizar (separados por coma)", value=default_tickers)
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        order_qty = st.number_input("Cantidad por orden (acciones)", min_value=1.0, value=1.0, step=1.0)

        auto_execute = st.checkbox("Auto-ejecutar órdenes en paper trading", value=False)
        if auto_execute:
            st.warning(
                "Con esta opción activada, un consenso BUY/SELL colocará una orden real "
                "en tu cuenta paper al presionar 'Analizar'."
            )

        analyze_clicked = st.button("Analizar", type="primary", use_container_width=True)

    try:
        client, graph = get_resources()
    except ValueError as e:
        st.error(str(e))
        st.stop()

    account = client.get_account_info()
    col1, col2, col3 = st.columns(3)
    col1.metric("Cash disponible", f"${account['cash']:,.2f}")
    col2.metric("Equity total", f"${account['equity']:,.2f}")
    col3.metric("Buying power", f"${account['buying_power']:,.2f}")

    if "results" not in st.session_state:
        st.session_state.results = {}

    if analyze_clicked:
        if not tickers:
            st.warning("Agrega al menos un ticker.")
        for symbol in tickers:
            with st.spinner(f"Orquestando {symbol}..."):
                result = graph.invoke({
                    "symbol": symbol,
                    "votes": [],
                    "consensus": None,
                    "auto_execute": auto_execute,
                    "order_qty": order_qty,
                })
                st.session_state.results[symbol] = result

    if not st.session_state.results:
        st.info("Configura los tickers en la barra lateral y presiona 'Analizar' para ver el razonamiento de cada agente.")
        return

    for symbol, result in st.session_state.results.items():
        st.divider()
        st.subheader(symbol)

        votes_by_agent = {v.agent_name: v for v in result["votes"]}
        cols = st.columns(3)
        for col, agent_name in zip(cols, ["technical_agent", "sentiment_agent", "risk_agent"]):
            vote = votes_by_agent.get(agent_name)
            with col:
                if vote:
                    render_vote_card(vote)
                else:
                    st.warning(f"{AGENT_DISPLAY_NAMES.get(agent_name, agent_name)} no emitió voto.")

        render_consensus_card(result["consensus"])


if __name__ == "__main__":
    main()
