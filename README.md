# Portfolio Risk Guardian

Sistema multi-agente de gestión de riesgo para trading algorítmico, construido
sobre la Alpaca Trading API (paper trading) y Claude como motor de razonamiento.

## Resumen del proyecto

Portfolio Risk Guardian es un proyecto de portafolio personal para aprender a
construir sistemas multi-agente coordinados, no una entrega de hackathon.
Simula un equipo de analistas de inversión: tres agentes independientes miran
un ticker desde ángulos distintos (indicadores técnicos, noticias, riesgo del
portafolio), un cuarto agente junta sus votos por consenso, y — solo si hay
mayoría clara — ejecuta la orden resultante en una cuenta de **paper trading**
(dinero simulado, nunca real).

El objetivo de diseño explícito es la **transparencia**: el sistema nunca
muestra solo "compra" o "vende" sin explicar por qué, en lenguaje natural,
desde cada agente. Eso se ve tanto en el dashboard como en cada `AgentVote`
que producen los agentes.

El proyecto ya pasó por un ciclo completo de "construir → medir → mejorar":
el Agente Técnico se backtesteó contra 2 años de datos reales, el resultado
inicial perdía dinero, se investigó qué usan sistemas más serios, se aplicó
un cambio concreto (filtro ADX + eliminar posiciones cortas) y se volvió a
medir para confirmar la mejora — ver [Resultados del backtest](#resultados-del-backtest-fase-5)
más abajo.

## Arquitectura

```
Agente 1 (Técnico)      → indicadores: RSI, medias móviles, volatilidad
Agente 2 (Sentimiento)  → noticias + Claude → sentimiento de mercado
Agente 3 (Riesgo)       → exposición del portafolio, sugiere hedging
Agente 4 (Orquestador)  → consenso entre agentes → ejecuta en Alpaca
Dashboard               → visualiza el razonamiento de cada agente en vivo
```

## Setup

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar credenciales
cp .env.example .env
# Editar .env con tus API keys reales:
#   - Alpaca: https://app.alpaca.markets/paper/dashboard/overview
#   - Anthropic: https://console.anthropic.com/

# 4. Probar la Fase 1 (Agente Técnico + conexión a Alpaca)
python run_phase1.py

# 5. Probar la Fase 2 (+ Agente de Sentimiento + Agente de Riesgo)
python run_phase2.py

# 6. Probar la Fase 3 (orquestador LangGraph, sin ejecutar órdenes)
python run_phase3.py
# Para que sí ejecute las órdenes de consenso en paper trading:
AUTO_EXECUTE=true python run_phase3.py
# Para que además dimensione la posición por volatilidad (ATR) en vez de tamaño fijo:
AUTO_EXECUTE=true USE_ATR_SIZING=true python run_phase3.py

# 7. Levantar el dashboard (Fase 4)
streamlit run dashboard/app.py

# 8. Backtest del Agente Técnico sobre datos históricos (Fase 5)
python run_backtest.py

# 9. Impacto de comisión + slippage en el backtest
python run_costs_backtest.py

# 10. Validación walk-forward del umbral de ADX (fuera de muestra real)
python run_walk_forward.py

# 11. Tamaño fijo vs. escalado por confianza propia del Agente Técnico
python run_confidence_sizing_backtest.py

# 12. Backtest del consenso Técnico+Riesgo sobre una cartera simulada
python run_technical_risk_backtest.py

# 13. Exploración: momentum cruzado (cross-sectional)
python run_momentum_backtest.py

# 14. Validación walk-forward del momentum cruzado (top_n fuera de muestra)
python run_momentum_walk_forward.py

# 15. Impacto de comisión + slippage en el momentum cruzado
python run_momentum_costs_backtest.py

# 16. Momentum cruzado fuera de universo (tickers distintos, mismos parámetros)
python run_momentum_universe_test.py

# 17. Tests unitarios (schemas, consenso, motor de backtest — sin llamar a Alpaca ni a Claude)
pytest tests/
```

## Roadmap de fases

- [x] **Fase 1** — Conexión a Alpaca + Agente Técnico funcional
- [x] **Fase 2** — Agente de Sentimiento (noticias + Claude) + Agente de Riesgo
- [x] **Fase 3** — Orquestador con lógica de consenso (LangGraph) + ejecución real
- [x] **Fase 4** — Dashboard en tiempo real (Streamlit) mostrando el razonamiento de cada agente
- [x] **Fase 5** — Backtesting del Agente Técnico con datos históricos (solo agente técnico —
      ver `core/backtester.py` para por qué sentimiento/riesgo quedan fuera de esta fase)
- [x] **Fase 6** — Rigor de validación: costos de transacción, walk-forward, position sizing
      por ATR portado al orquestador real, y tests automatizados (ver
      [Fase 6: rigor de validación](#fase-6-rigor-de-validación-post-diagnóstico-de-sobreajuste) más abajo)
- [x] **Fase 7** — Cierre de los dos pendientes de Fase 6: simulador de cartera para
      backtestear el Agente de Riesgo, y momentum cruzado como mejora de forma de
      estrategia para el Técnico, validado con el mismo rigor que el resto del
      proyecto (walk-forward, costos, fuera de universo) — ver
      [Backtest del consenso Técnico+Riesgo](#backtest-del-consenso-técnicoriesgo-cartera-simulada)
      y [Exploración: momentum cruzado](#exploración-momentum-cruzado) más abajo

**Estado del proyecto**: con la Fase 7 cerrada, el único pendiente que
queda es una limitación externa real, no de código — el Agente de
Sentimiento sigue bloqueado por el límite de historial del plan gratuito
de NewsAPI (no cubre más de ~1 mes hacia atrás, así que backtestear años de
"sentimiento histórico" no es viable). Todo lo demás que estaba al alcance
del proyecto se corrió, se probó con 48 tests automatizados y se documentó
con lectura honesta, resultado ganador o perdedor.

### Resultados del backtest (Fase 5)

El backtest inicial del Agente Técnico (RSI + cruce de SMA, sin filtro de tendencia,
permitiendo posiciones cortas) se corrió sobre ~2 años de datos diarios reales
(AAPL, MSFT, TSLA, vía `run_backtest.py`) y **perdía dinero en los tres tickers**,
por debajo incluso de comprar y mantener (buy & hold):

| Ticker | Retorno estrategia (baseline) | Buy & hold | Máx. drawdown |
|---|---|---|---|
| AAPL | -16.2% | +44.3% | -27.2% |
| MSFT | -16.5% | +16.8% | -42.9% |
| TSLA | -43.3% | +8.1% | -60.5% |

Se investigó qué usan sistemas de trading algorítmico más serios (ver el prompt
de investigación usado, abajo) y se aplicaron dos cambios concretos, cada uno
validado corriendo `run_backtest.py` antes de aplicarse al sistema real:

1. **Filtro ADX** (`agents/technical_agent.py`): el cruce de medias móviles solo
   cuenta como señal de tendencia si el ADX confirma que hay una tendencia real
   (≥25). Antes, el agente cambiaba de señal casi a diario en mercados laterales
   — puro ruido, no señal.
2. **Sin posiciones cortas** (`agents/orchestrator.py` + `core/backtester.py`):
   una señal SELL sin posición existente ya no abre un corto, ni en el backtest
   ni en el sistema real — simplemente no hace nada.

Resultado después de aplicar ambos cambios (mismo período, mismos tickers):

| Ticker | Retorno estrategia (con ADX, sin cortos) | Buy & hold | Máx. drawdown | Sharpe |
|---|---|---|---|---|
| AAPL | -6.9% | +44.3% | -12.9% | -0.26 |
| MSFT | **+13.5%** | +16.8% | -5.1% | 1.32 |
| TSLA | **+15.8%** | +8.1% | -23.3% | 0.49 |

Lectura honesta: mejoró de forma medible en los tres casos (TSLA pasó de perder
-43.3% a ganar +15.8%, MSFT de perder a casi igualar buy & hold con un drawdown
mucho menor), pero AAPL sigue sin ganarle a comprar y mantener — la estrategia
está pensada para detectar reversiones y tendencias confirmadas, no para un
activo con una subida sostenida y sin sobresaltos como tuvo AAPL en este período.
Ver `agents/technical_agent.py` y `core/backtester.py` para el detalle.

Para repetir esta comparación tras un cambio futuro: anota los números actuales,
haz el cambio, corre `python run_backtest.py` de nuevo y compara contra esta tabla.

### Diagnóstico de sobreajuste (post-Fase 5)

La mejora de la tabla anterior generó una duda razonable: ¿el filtro ADX capta
algo real, o solo "acertó" en esos 3 tickers y ese período específico? Se hizo
un diagnóstico de dos partes, sin tocar ningún parámetro entre medio.

**1. Barrido de sensibilidad del ADX** (`run_adx_sweep.py`) — correr el backtest
variando solo el umbral de ADX entre 15 y 35, para ver si el rendimiento forma
una "meseta" estable (señal real) o un "pico" aislado (suerte estadística):

- **MSFT y TSLA**: meseta estable y amplia (Sharpe positivo en un rango ancho
  de umbrales) — indicio de que el filtro captura algo real en esos activos.
- **AAPL**: el umbral elegido (25) cae justo al borde de un precipicio — en
  ADX=23 el retorno era +15.9%, y en ADX=25 ya es -6.9%, empeorando hasta
  -25.8% en ADX=28. Es el patrón de "pico junto a un precipicio" que indica
  sobreajuste, no una propiedad real del activo.

**2. Prueba fuera de universo** (`run_universe_test.py`) — correr la misma
lógica (RSI 14 + SMA 20/50 + ADX≥25, sin cortos), **sin tocar ningún
parámetro**, sobre 13 tickers de 7 sectores distintos a los usados para
desarrollarla, sobre 5 años en vez de 2:

| Ticker | Sector | Sharpe | Retorno estrategia | Buy & Hold | ¿Le gana? |
|---|---|---|---|---|---|
| MSFT* | Tecnología | 1.45 | +15.0% | +18.7% | No |
| NEE | Utilities | 0.50 | +21.3% | +9.3% | Sí |
| TSLA* | Consumo discrecional (EV) | 0.49 | +15.8% | +8.1% | Sí |
| JNJ | Salud | 0.46 | +16.1% | +94.5% | No |
| CVX | Energía | 0.45 | +24.9% | +120.7% | No |
| XOM | Energía | 0.33 | +15.8% | +196.8% | No |
| CAT | Industrial | 0.32 | +18.6% | +318.0% | No |
| JPM | Financiero | -0.04 | -2.9% | +142.7% | No |
| WMT | Consumo defensivo | -0.00 | -3.0% | +136.6% | No |
| UNH | Salud | -0.03 | -4.0% | -3.3% | No |
| KO | Consumo defensivo | -0.01 | -1.1% | +80.6% | No |
| BAC | Financiero | -0.26 | -11.5% | +49.5% | No |
| AAPL* | Tecnología | -0.31 | -8.0% | +45.4% | No |
| HON | Industrial | -0.31 | -12.1% | -6.9% | No |
| DIS | Comunicaciones | -0.32 | -11.9% | -29.9% | Sí |
| PG | Consumo defensivo | -0.32 | -8.6% | +13.4% | No |

*(\* = los 3 tickers originales sobre los que se desarrolló y ajustó el filtro)*

**Resumen agregado**: solo 2/13 tickers nuevos le ganaron a buy & hold, Sharpe
promedio ≈0.06 (indistinguible de no tener ninguna ventaja real). No hay un
patrón limpio por sector (Salud tiene un caso arriba y uno abajo; Tecnología
igual; Industrial igual) — con 16 tickers no alcanza para aislar un subgrupo
donde el sistema funcione de forma confiable sin caer en el mismo sobreajuste
que se está diagnosticando.

**Efecto colateral útil**: este diagnóstico encontró un bug real de datos —
`AlpacaClient` traía precios sin ajustar por splits (`adjustment=Adjustment.ALL`
ahora corrige esto en `core/alpaca_client.py`). Sin el fix, WMT mostraba un
"crash" falso de -66% en un día por su split 3-por-1 de feb-2024, que no
ocurrió en la realidad. Cualquier backtest futuro sobre un ticker con split
en el rango de fechas se hubiera visto afectado igual.

### Por qué el Agente Técnico no muestra una ventaja generalizable (investigación)

Con el hallazgo de arriba, la duda siguiente fue: si esto no generaliza,
¿estamos omitiendo algo, o es esperable? Investigación con fuentes académicas
reales (no otro prompt genérico) dio una respuesta concreta:

1. **Las reglas técnicas simples (RSI, cruces de medias) tienen evidencia
   débil e inconsistente específicamente en acciones líquidas de EE.UU.** —
   más soporte académico existe en forex y futuros/commodities que en acciones
   individuales, que están entre los mercados más eficientemente cubiertos del
   mundo (Park & Irwin 2004; Sullivan et al. 1999).
2. **Que una estrategia backtesteada no generalice es lo típico, no lo raro**:
   ~44% de estrategias publicadas en papers académicos no replican en datos
   nuevos; en la industria, >90% de estrategias fallan al pasar de backtest a
   capital real.
3. **La pieza clave que faltaba — por qué los fondos que usan reglas parecidas
   sí ganan dinero**: no es que su señal sea muchísimo mejor, es la **amplitud
   (breadth)**. La "Ley Fundamental" de gestión activa (Grinold) dice que el
   resultado ajustado por riesgo de una estrategia ≈ (calidad de la señal) ×
   **raíz cuadrada del número de apuestas independientes**. Los fondos CTA
   aplican reglas de tendencia simples sobre decenas o cientos de mercados no
   correlacionados simultáneamente — una señal débil en UN solo activo nunca
   va a mostrar un Sharpe robusto por diseño; necesita muchas apuestas
   independientes para que el ruido se cancele y el promedio salga positivo.
4. **Diferencia de forma de estrategia**: lo que sí tiene evidencia académica
   sólida es el *momentum cruzado* (Jegadeesh-Titman) — comparar muchas
   acciones entre sí y apostar por las relativamente mejores contra las
   peores — no el *momentum de un solo activo contra su propio pasado*, que es
   lo que hace nuestro RSI+SMA. Son dos estrategias con evidencia muy distinta.

**Conclusión**: el Agente Técnico, evaluando un ticker a la vez, está
estructuralmente en la peor posición posible para mostrar una ventaja
estadística robusta — no es un error de implementación, es una limitación de
diseño conocida en la literatura. El camino con más soporte real para obtener
una ventaja genuina sería aplicar la misma lógica sobre una cesta diversificada
de forma simultánea (breadth) en vez de analizar tickers de forma aislada —
explorado en la siguiente sección.

Fuentes consultadas:
- [The predictive ability of technical trading rules: an empirical study](https://link.springer.com/article/10.1007/s11408-023-00433-2)
- [Why 90% of Profitable Backtests Are Statistically Invalid](https://daviddtech.medium.com/the-three-deadly-sins-of-backtesting-overfitting-look-ahead-bias-and-p-hacking-a68c6345e668)
- [A Trend Following Deep Dive: The Optimal Market Mix for a Trend Following Strategy](https://www.man.com/insights/trend-following-optimal-market-mix)
- [The Fundamental Law of Active Management: Redux](https://www.sciencedirect.com/science/article/pii/S0927539817300543)
- [Cross-Sectional and Time-Series Momentum Returns and Market Dynamics](https://mpra.ub.uni-muenchen.de/78989/1/MPRA_paper_78989.pdf)

### Exploración de breadth: cesta diversificada + gestión de riesgo (ATR)

Se probó la hipótesis de breadth directamente: en vez de evaluar cada ticker
por separado, se combinó la señal del Agente Técnico sobre los mismos 16
tickers **simultáneamente, como un solo portafolio**, en dos pasos
incrementales (código en `core/backtester.py`: `run_basket_backtest` y
`run_basket_backtest_atr`; scripts `run_breadth_backtest.py` y
`run_breadth_atr_backtest.py`):

1. **Combinar los 16 tickers, ponderados de forma inversa a su volatilidad**
   (para que TSLA no domine el riesgo del conjunto).
2. **Sumarle sizing por volatilidad y stops dinámicos basados en ATR(14)**
   por trade — las mejoras de mayor prioridad según la investigación (bajo
   riesgo de sobreajuste, no agregan señales de entrada nuevas).

| Enfoque | Sharpe | Máx. drawdown | Retorno estrategia | Buy & hold de la cesta |
|---|---|---|---|---|
| Ticker individual (Sharpe promedio de la tabla anterior) | ~0.06 | -5% a -30% según ticker | mixto | 2/13 le ganan |
| + Breadth (16 tickers, peso por volatilidad) | 0.28 | -5.6% | +4.2% | +71.3% |
| + Breadth + ATR sizing/stops | **0.49** | **-2.6%** | +3.8% | +89.3% |

**Lectura honesta**: cada capa de gestión de riesgo mejoró la calidad
ajustada por riesgo de forma consistente — el Sharpe casi se duplicó dos
veces seguidas (0.06 → 0.28 → 0.49) y el drawdown máximo se redujo a una
fracción de lo que tenía cualquier ticker individual. Es justo lo que predice
la Ley Fundamental de Grinold y lo que recomendaba la investigación sobre ATR.
Pero el retorno absoluto se queda prácticamente plano en las tres variantes,
muy por debajo de simplemente comprar y mantener la misma cesta. Un Sharpe de
0.49 es una mejora real, pero sigue lejos del ~1.0 considerado sólido.

**Conclusión de esta exploración**: la gestión de riesgo (breadth, ATR) puede
pulir una señal débil — reduce drawdown y mejora el Sharpe de forma medible y
repetible — pero no la convierte en una señal fuerte. El cuello de botella
sigue siendo la calidad de la señal de entrada (RSI+SMA+ADX en un ticker
contra su propio pasado), no la falta de diversificación o de gestión de
riesgo. Mejorar eso requeriría un cambio de forma de estrategia más profundo
(ej. momentum cruzado en vez de momentum de un solo activo — ver fuentes
arriba), que queda fuera del alcance actual del proyecto.

## Fase 6: rigor de validación (post-diagnóstico de sobreajuste)

Sesión de seguimiento después de la Fase 5: los diagnósticos de
sobreajuste (sweep de ADX, prueba fuera de universo) ya habían encontrado
el problema — el Agente Técnico no generaliza — pero quedaban huecos de
rigor sin cerrar. Esta fase los cierra, con el mismo criterio de "probar
con datos reales, documentar el resultado sea cual sea" que el resto del
proyecto. Resumen de lo agregado (detalle de cada uno en sus propias
secciones):

1. **Costos de transacción** (comisión + slippage) en el backtest —
   antes asumía ejecución gratuita y perfecta.
2. **Walk-forward rodante** del umbral de ADX — la validación fuera de
   muestra que faltaba; confirmó el sobreajuste de forma más contundente
   que los diagnósticos previos.
3. **Position sizing por confianza propia** del Agente Técnico — probado
   como la versión ejecutable de "validar el consenso 2-de-3 contra
   alternativas" (la versión completa con Sentimiento/Riesgo sigue
   bloqueada por falta de historial real).
4. **Position sizing por ATR portado al orquestador real**
   (ver [Position sizing por ATR en el orquestador real](#position-sizing-por-atr-en-el-orquestador-real))
   — con un bug real de sobre-concentración encontrado y corregido en el
   camino (ver esa sección).
5. **33 tests automatizados** en `tests/` (antes vacío) — schemas, la
   regla de consenso 2-de-3, el motor de backtest y el sizing por ATR,
   todos sobre datos sintéticos o con clientes falsos, sin llamar a
   Alpaca ni a Claude.

Scripts nuevos: `run_costs_backtest.py`, `run_walk_forward.py`,
`run_confidence_sizing_backtest.py`. Módulo nuevo: `core/position_sizing.py`
(fuente única de la fórmula ATR, usada por el backtester y por
`TechnicalAgent.calculate_position_size`). `requirements.txt` ahora incluye
`pytest`.

**Actualización (Fase 7)**: este pendiente quedó cerrado del lado que
dependía del proyecto. Se construyó un simulador de cartera y se
backtesteó el consenso Técnico+Riesgo (ver
[Backtest del consenso Técnico+Riesgo](#backtest-del-consenso-técnicoriesgo-cartera-simulada)
más abajo), y se exploró momentum cruzado como mejora de forma de
estrategia para el Técnico — validado con el mismo rigor que el resto del
proyecto: walk-forward, costos de transacción, y prueba fuera de universo
(ver [Exploración: momentum cruzado](#exploración-momentum-cruzado) y sus
subsecciones). Sentimiento sigue bloqueado por el límite de historial de
NewsAPI — esa parte es una limitación externa real, no de código, y sigue
sin ser viable barato validar la regla de consenso 2-de-3 completa con los
3 agentes reales.

### Costos de transacción y validación walk-forward (post-diagnóstico)

Los backtests anteriores tenían dos huecos de rigor: asumían ejecución
gratuita y perfecta (sin comisión ni slippage), y el umbral ADX=25 se
"eligió" mirando cómo funcionaba en todo el período evaluado — un
look-ahead implícito, aunque nunca se ajustó buscando el mejor número.
Se agregaron ambas correcciones a `core/backtester.py`.

**1. Comisión + slippage** (`run_costs_backtest.py`) — Alpaca no cobra
comisión en acciones, pero se asumió un slippage conservador de 5bps
(0.05%) por cambio de posición, sobre el mismo backtest de Fase 5:

| Ticker | Retorno sin costos | Retorno con costos (slippage 5bps) | Costo absorbido |
|---|---|---|---|
| AAPL | -8.0% | -9.4% | 1.46 puntos |
| MSFT | +15.0% | +13.8% | 1.20 puntos |
| TSLA | +15.8% | +15.0% | 0.81 puntos |

El efecto es real pero modesto (~1 punto porcentual) porque la estrategia
ya opera poco (filtro ADX). No cambia la conclusión por sí solo, pero
confirma que los números de Fase 5 eran ligeramente optimistas.

**2. Walk-forward rodante del umbral de ADX** (`run_walk_forward.py`) —
train 2 años / test 6 meses, reoptimizando el umbral (barrido 15-35)
usando SOLO el pasado en cada ventana, sin tocarlo una vez fijado para el
período de test siguiente:

| Ticker | Retorno OOS walk-forward | Buy & hold | Sharpe OOS | Retorno con ADX=25 fijo (referencia, con look-ahead) | Sharpe fijo |
|---|---|---|---|---|---|
| AAPL | -0.4% | +42.5% | 0.04 | +27.3% | 0.45 |
| MSFT | +3.7% | +22.2% | 0.27 | +13.6% | 0.34 |
| TSLA | -11.0% | +55.6% | -0.13 | +17.0% | 0.26 |

**Lectura honesta**: esto es más contundente que el barrido de
sensibilidad o la prueba fuera de universo. Cuando el umbral se
reoptimiza cada 6 meses sin ver el período que se está evaluando, el
resultado colapsa frente al backtest de "ADX=25 fijo sobre todo el
período" en los tres tickers — TSLA pasa de ganar +17% a **perder**
-11% fuera de muestra, y AAPL y MSFT quedan muy por debajo de buy & hold
con Sharpe cercano a cero. Confirma, con el estándar de validación que
usa la industria (no solo con datos de otros tickers), que la ventaja
reportada en la Fase 5 era en gran parte ajuste al ruido del período
específico sobre el que se desarrolló, no una propiedad real y estable
del umbral elegido.

### Position sizing por confianza propia (¿el agente "sabe" cuándo está más seguro?)

La regla de consenso 2-de-3 del orquestador (ver [Notas de diseño](#notas-de-diseño))
nunca se validó contra alternativas (ponderar por confianza, exigir
unanimidad) porque esas reglas requieren backtestear también Sentimiento y
Riesgo, y ninguno de los dos tiene historial real disponible (Sentimiento
por límite de NewsAPI, Riesgo porque depende del estado del portafolio en
el tiempo — ver el alcance deliberado al inicio de este archivo). Lo que sí
se puede probar con datos reales, sin inventar nada, es una versión más
chica de la misma pregunta: el Agente Técnico ya calcula una `confidence`
(0-1) por señal, más alta cuando RSI y tendencia coinciden (ver
`agents/technical_agent.py::_decide`) — ¿esos días de alta confianza
producen mejor resultado que los de baja confianza?

`run_confidence_sizing_backtest.py` compara el tamaño de posición fijo
(±1, usado en toda la Fase 5) contra escalar la posición por esa
confianza en cada trade:

| Ticker | Retorno fijo | Retorno por confianza | Sharpe fijo | Sharpe por confianza | MaxDD fijo | MaxDD por confianza |
|---|---|---|---|---|---|---|
| AAPL | -8.0% | -4.2% | -0.31 | -0.31 | -12.9% | -8.1% |
| MSFT | +15.0% | +6.3% | 1.45 | 1.28 | -5.0% | -2.8% |
| TSLA | +15.8% | +5.2% | 0.49 | 0.45 | -23.3% | -7.8% |

**Lectura honesta**: escalar por confianza reduce el retorno absoluto y el
drawdown máximo de forma consistente en los tres tickers (tiene sentido:
la confianza rara vez llega a 1.0, así que la exposición promedio baja),
pero el **Sharpe se mantiene igual o levemente peor** en los tres casos —
no hay evidencia de que la confianza que el agente calcula para sí mismo
prediga mejor calidad de señal fuera de reducir el tamaño de la apuesta.
Es un resultado negativo, pero uno real: confirma que la regla 2-de-3 por
conteo simple (no ponderada) no está dejando una mejora obvia sobre la
mesa, al menos por este camino — no se puede decir lo mismo de las
alternativas que involucran a Sentimiento y Riesgo, que siguen sin
validar por falta de historial.

### Backtest del consenso Técnico+Riesgo (cartera simulada)

El pendiente de arriba señalaba que Riesgo no se podía backtestear porque
depende de la evolución de una cartera real en el tiempo (`cash`,
posiciones, `$` de cada una), no de histórico de noticias como Sentimiento.
`core/portfolio_backtester.py` cierra ese hueco: simula una cartera
compartida ($100,000 iniciales) sobre la misma cesta de 16 tickers, día por
día, y antes de cada entrada BUY del Agente Técnico llama directamente a
`RiskAgent._decide` (la misma función que usa el sistema real, sin
reimplementar sus umbrales) con la concentración y volatilidad de cartera
resultantes, calculadas de forma causal (solo datos hasta ese día).

Un detalle que apareció al implementarlo: `RiskAgent._decide` nunca emite
BUY (por diseño — ver `agents/risk_agent.py`, es un agente que solo frena,
nunca empuja). Así que la regla de consenso no puede ser "ambos agentes
votan BUY" — es "la entrada se ejecuta salvo que Riesgo marque un problema
real" (SELL por sobre-concentración, o HOLD de alta confianza por
volatilidad de cartera alta).

`run_technical_risk_backtest.py` compara, sobre el mismo universo y período
de `run_breadth_atr_backtest.py`, el Técnico solo (con ATR sizing) contra
el Técnico+Riesgo (cartera simulada, mismos umbrales de Riesgo que el
sistema real: `max_position_pct=25%`, `high_portfolio_vol_threshold=40%`):

| Métrica | Técnico solo (ATR) | Técnico + Riesgo |
|---|---|---|
| Retorno total | +3.8% | +62.5% |
| Máximo drawdown | -2.6% | -26.2% |
| Sharpe | 0.49 | 0.68 |
| Sortino | 0.75 | 1.11 |
| Calmar | 0.30 | 0.41 |
| Win rate | 53.8% | 53.2% |
| Profit factor | 1.10 | 1.12 |

Cartera simulada: 44 trades ejecutados, **0 entradas vetadas por Riesgo**,
concentración promedio por entrada 19.3% (por debajo del tope de 25%).

**Lectura honesta**: estos dos números no son directamente comparables en
metodología — `run_basket_backtest_atr` promedia retornos ATR-sizeados por
ticker (espacio de porcentajes, sin cartera real), mientras que el
backtest de cartera compone sobre un solo pool de $100,000 compartido
entre 16 tickers, lo que le permite capitalizar ganadores de forma
distinta. La comparación de Sharpe/drawdown es más informativa que la de
retorno absoluto. El hallazgo más claro es que **el Agente de Riesgo, con
sus umbrales actuales, no vetó ni una sola entrada en 5 años de historial
sobre esta cesta diversificada** — la concentración promedio (19.3%) se
mantuvo cómodamente bajo el tope (25%) porque diversificar entre 16
tickers ya evita naturalmente la sobre-concentración que el agente está
diseñado para frenar. Es un resultado negativo honesto sobre el *valor
añadido* del veto en este escenario concreto (no sobre si la regla en sí
es razonable): con una cesta amplia y bien diversificada, el Agente de
Riesgo actúa como una red de seguridad que casi nunca hace falta activar
— su utilidad esperada crece en escenarios más concentrados (pocos
tickers, posiciones más grandes), que no es el caso probado acá.

### Exploración: momentum cruzado

La investigación previa (ver "Por qué el Agente Técnico no muestra una
ventaja generalizable" más arriba) señaló que la única mejora de *forma*
de estrategia con evidencia académica sólida es el momentum cruzado
(Jegadeesh-Titman): rankear muchos tickers entre sí y apostar por los
relativamente mejores, en vez de mirar un solo activo contra su propio
pasado (lo que hace RSI+SMA). `core/momentum_backtester.py` lo implementa
como exploración de backtest (no integrado al orquestador real — ver esa
sección para por qué): retorno de formación de 12 meses menos el último
mes (convención estándar que evita el efecto de reversión de muy corto
plazo), rebalanceo mensual, top 5 tickers equal-weight, sin cortos.

`run_momentum_backtest.py` corre esto sobre el mismo universo de 16
tickers de `run_universe_test.py`, ~5 años:

| Métrica | Momentum cruzado |
|---|---|
| Retorno total | +85.6% |
| Buy & hold de la cesta | +102.7% |
| Máximo drawdown | -25.3% |
| Sharpe | 0.97 |
| Sortino | 1.60 |
| Calmar | 0.68 |
| Win rate | 54.8% |
| Profit factor | 1.19 |

**Lectura honesta**: no le gana a comprar y mantener la cesta en retorno
absoluto (+85.6% vs. +102.7%, en un período con mercado fuertemente
alcista donde superar buy & hold en términos absolutos es difícil para
cualquier estrategia con gestión de riesgo), pero el Sharpe (0.97) casi
duplica la mejor referencia previa del proyecto (breadth+ATR, Sharpe 0.49
— ver la sección de esa exploración) y el drawdown es comparable al de esa
variante. Es la señal más fuerte que ha producido cualquier variante
probada en este proyecto de que la *forma* de estrategia (cross-sectional,
no un solo activo contra su pasado) importa más que agregar gestión de
riesgo sobre una señal débil — consistente con lo que predecía la
investigación académica citada arriba. Con un Sharpe de 0.97, sigue sin
llegar al ~1.0+ considerado sólido en la industria, así que esto es
evidencia alentadora, no una conclusión definitiva — el resultado depende
del universo de 16 tickers específico y del período de ~5 años probado. El
mismo tipo de rigor que expuso el sobreajuste del Agente Técnico original
(walk-forward) se aplicó también acá — ver a continuación.

#### Walk-forward del momentum cruzado

El Sharpe de 0.97 de arriba se calculó sobre una sola ventana de ~5 años —
la misma forma en que se calculó originalmente el ADX=25 que después
resultó sobreajustado. `run_momentum_walk_forward_backtest`
(`core/momentum_backtester.py`) aplica el mismo pipeline de
`run_walk_forward_backtest` (Fase 6): reoptimiza `top_n` (cuántos tickers
sostener — el único parámetro "libre" acá; `lookback_months=12` y
`skip_months=1` son la convención académica estándar, no algo ajustado a
este universo, así que barrerlos no diagnosticaría sobreajuste, lo
introduciría) sobre una ventana de entrenamiento de 3 años, y evalúa ese
`top_n` fuera de muestra en los 6 meses siguientes que la optimización
nunca vio, rodando hacia adelante (`run_momentum_walk_forward.py`):

| Fold | Train | top_n elegido | Train Sharpe | Test | Test Sharpe | Test retorno |
|---|---|---|---|---|---|---|
| 1 | 2021-09-07 → 2024-09-09 | 5 | 0.83 | 2024-09-10 → 2025-03-12 | -0.83 | -8.9% |
| 2 | 2022-03-08 → 2025-03-12 | 6 | 0.49 | 2025-03-13 → 2025-09-11 | 0.96 | +12.1% |
| 3 | 2022-09-07 → 2025-09-11 | 5 | 0.80 | 2025-09-12 → 2026-03-13 | 2.13 | +18.6% |

| Métrica | Ventana única (referencia) | Walk-forward (fuera de muestra) |
|---|---|---|
| Sharpe | 0.97 | **0.69** |
| Retorno | +85.6% | +21.1% (período de test es más corto: solo empieza tras 3 años de warmup de train) |
| Máximo drawdown | -25.3% | -27.0% |

**Lectura honesta**: el Sharpe cae de 0.97 a 0.69 fuera de muestra — hay
sobreajuste real, como es de esperar (nunca desaparece del todo al pasar a
validación out-of-sample). Pero la caída es **mucho más moderada** que la
que sufrió el ADX del Agente Técnico (ahí el Sharpe colapsaba a negativo
en TSLA y cerca de cero en los demás tickers, ver la sección de walk-forward
de Fase 6 más arriba). El `top_n` elegido se mantuvo estable entre folds
(5, 6, 5) en vez de saltar erráticamente entre valores del grid — otra
señal de que no es puro ruido. Solo uno de los tres folds de test salió
negativo (-8.9%); los otros dos fueron sólidamente positivos (+12.1%,
+18.6%). Conclusión: el momentum cruzado retiene *parte* de su ventaja
fuera de muestra — no es sobreajuste puro como el ADX, pero el número
creíble para este sistema es **Sharpe ≈0.69, no 0.97** — y sigue sin
probarse fuera del universo de 16 tickers usado (el otro diagnóstico que sí
se le aplicó al Agente Técnico, `run_universe_test.py`, pendiente para una
sesión futura si se decide seguir esta línea).

#### Costos de transacción del momentum cruzado

Con más turnover que el Agente Técnico (rebalanceo mensual sobre una cesta
de 16 tickers, en vez de operar solo cuando ADX confirma tendencia), valía
la pena verificar si el Sharpe walk-forward de 0.69 sobrevivía a una
ejecución realista. `run_momentum_costs_backtest.py` corre tanto la
ventana única como el walk-forward con el mismo supuesto conservador que
`run_costs_backtest.py` usó para el Técnico (slippage 5bps por cambio de
posición, comisión 0 — Alpaca no cobra en acciones):

| Variante | Sharpe sin costos | Sharpe con costos | Retorno absorbido |
|---|---|---|---|
| Ventana única | 0.97 | 0.95 | 1.94 puntos |
| Walk-forward (OOS) | 0.69 | 0.68 | 0.41 puntos |

**Lectura honesta**: el efecto es mínimo — más chico incluso, en términos
relativos, que el ~1 punto que absorbió el Agente Técnico original a pesar
de que este rebalancea con más frecuencia. Tiene sentido: cada rebalanceo
mensual típicamente reemplaza 1-2 de los 5 tickers sostenidos, no la
cesta entera, así que el turnover real por mes es bajo. El `top_n` elegido
por fold del walk-forward ni siquiera cambió con costos (`[5, 6, 5]` en
ambos casos). Conclusión: el número que importa reportar como resultado
final de esta exploración sigue siendo **Sharpe walk-forward ≈0.68-0.69**
— los costos no lo infravaloraban ni lo estaban inflando de forma
relevante.

#### Prueba fuera de universo del momentum cruzado

El último diagnóstico que le faltaba al momentum cruzado, el mismo que
expuso el sobreajuste del ADX del Agente Técnico en Fase 5
(`run_universe_test.py`): correr la estrategia, **sin tocar ningún
parámetro** (formación 12m-1m, top 5 equal-weight, rebalance mensual, sin
cortos), sobre una cesta de 16 tickers completamente distinta a la
original — sectores similares, cero superposición de tickers
(`run_momentum_universe_test.py`):

| Universo | Sharpe (ventana única) | Retorno | Buy & hold | ¿Le gana? |
|---|---|---|---|---|
| Original (16 tickers, con el que se desarrolló) | 0.97 | +85.6% | +102.7% | No |
| Nuevo (16 tickers, distinto, mismo período) | **1.54** | **+254.0%** | +122.9% | **Sí** |

**Lectura honesta**: a diferencia del ADX (que colapsaba al cambiar de
universo), acá el resultado se **sostiene y hasta mejora** — Sharpe más
alto y, por primera vez en todo el proyecto, la estrategia le gana a
comprar y mantener en retorno absoluto. Es la evidencia más favorable que
ha producido cualquier variante probada. Pero hay una salvedad real que no
corresponde ocultar: el universo nuevo incluye NVDA, uno de los mayores
ganadores del boom de IA 2022-2026 — parte de este resultado puede ser el
sector/período (cualquier estrategia de momentum captura ganadores fuertes
casi por definición cuando el mercado tiene un líder tan dominante), no
solo una propiedad robusta de la estrategia en sí. Tampoco se corrió
walk-forward sobre este universo nuevo (solo ventana única) — con el mismo
rigor aplicado al universo original, el número walk-forward de este
universo probablemente sería menor a 1.54, igual que pasó con el original
(0.97 → 0.69). Con esto, los dos diagnósticos pendientes del momentum
cruzado (costos, fuera de universo) quedan cerrados: el resultado más
creíble y ya validado con ambos tipos de rigor sigue siendo el del universo
original, **Sharpe walk-forward ≈0.68-0.69**; el universo nuevo es una
señal adicional alentadora, no una confirmación con el mismo nivel de
validación.

## Estructura del proyecto

```
portfolio-risk-guardian/
├── agents/
│   ├── technical_agent.py    # Agente 1: análisis técnico
│   ├── sentiment_agent.py    # Agente 2: noticias + Claude
│   ├── risk_agent.py         # Agente 3: exposición del portafolio + hedging
│   └── orchestrator.py       # Agente 4: consenso (LangGraph) + ejecución
├── core/
│   ├── alpaca_client.py      # Cliente central Alpaca (datos + trading)
│   ├── schemas.py            # Modelos de datos compartidos (AgentVote, etc.)
│   ├── position_sizing.py    # ATR + sizing por volatilidad (compartido backtest/real)
│   ├── backtester.py         # Backtester del Agente Técnico (Fase 5)
│   ├── portfolio_backtester.py  # Simulador de cartera + backtest Técnico+Riesgo
│   └── momentum_backtester.py   # Backtest de momentum cruzado (exploración)
├── dashboard/
│   └── app.py                 # Dashboard Streamlit (Fase 4)
├── data/                     # (opcional) cache de datos históricos
├── tests/
│   ├── test_schemas.py             # validación de AgentVote/ConsensusDecision
│   ├── test_orchestrator_consensus.py  # regla de consenso 2-de-3
│   ├── test_backtester.py          # motor de backtest sobre datos sintéticos
│   ├── test_position_sizing.py     # ATR + sizing (backtest y orquestador real)
│   ├── test_portfolio_backtest.py  # simulador de cartera + veto de Riesgo
│   ├── test_momentum_backtest.py   # ranking de momentum cruzado, sin look-ahead
│   └── test_momentum_walk_forward.py  # walk-forward de top_n, sin look-ahead, costos
├── run_phase1.py             # script de prueba de la Fase 1
├── run_phase2.py             # script de prueba de la Fase 2
├── run_phase3.py             # script de prueba de la Fase 3
├── run_backtest.py           # script de la Fase 5 (backtest Agente Técnico)
├── run_adx_sweep.py          # diagnóstico: sensibilidad del umbral ADX
├── run_universe_test.py      # diagnóstico: generalización fuera de universo
├── run_breadth_backtest.py   # exploración: cesta diversificada como portafolio
├── run_breadth_atr_backtest.py  # exploración: + ATR sizing/stops
├── run_costs_backtest.py     # impacto de comisión + slippage en el backtest
├── run_walk_forward.py       # validación walk-forward del umbral de ADX
├── run_confidence_sizing_backtest.py  # tamaño fijo vs. escalado por confianza propia
├── run_technical_risk_backtest.py  # backtest del consenso Técnico+Riesgo (cartera simulada)
├── run_momentum_backtest.py  # exploración: momentum cruzado
├── run_momentum_walk_forward.py  # validación walk-forward de top_n (momentum cruzado)
├── run_momentum_costs_backtest.py  # impacto de comisión + slippage en el momentum cruzado
├── run_momentum_universe_test.py  # diagnóstico: generalización fuera de universo (momentum)
├── requirements.txt
└── .env.example
```

## Notas de diseño

- **Separación de responsabilidades**: cada agente analista (1, 2, 3) NO conoce
  la lógica de los demás. Solo emiten un `AgentVote` estandarizado (ver
  `core/schemas.py`). Esto permite testear y mejorar cada agente de forma
  independiente, y es clave para el diseño de sistemas multi-agente serios.
- **Transparencia por diseño**: cada `AgentVote` incluye un campo `reasoning`
  en lenguaje natural. Esto no es cosmético — es lo que después alimenta el
  dashboard y hace que el sistema sea auditable ("¿por qué se tomó esta
  decisión?"), algo crítico en cualquier sistema que toque dinero real o
  simulado.
- **Paper trading siempre**: `AlpacaClient` se inicializa con `paper=True`
  de forma fija. Cambiar esto a producción debe ser una decisión explícita
  y consciente, no un accidente de configuración.

### Position sizing por ATR en el orquestador real

El sizing por volatilidad (ATR) que se validó en el backtest de Fase 5
(`run_technical_backtest_atr` / `run_basket_backtest_atr`) se portó al
orquestador real (`core/position_sizing.py`, usado tanto por el backtester
como por `TechnicalAgent.calculate_position_size`) para que una orden BUY
en paper trading no compre una cantidad fija de acciones, sino dimensionada
por qué tan volátil está el ticker — activable con `USE_ATR_SIZING=true`
en `run_phase3.py` (opt-in explícito, igual que `AUTO_EXECUTE`; el default
sigue siendo tamaño fijo).

Al portar la fórmula apareció un problema real que el backtest no exponía:
`max_weight=3.0` era seguro en el backtest porque `weight` se promediaba
entre una cesta de 16 tickers (`run_basket_backtest_atr`) — ningún ticker
dominaba el libro. Aplicado a una sola orden real para un solo ticker, sin
ese contexto de cesta, el peso "crudo" resultó en **30-50% de la cuenta en
un solo nombre** (verificado con la cuenta paper real: AAPL pedía 46%,
MSFT 42%). Eso contradice directamente el límite de concentración que el
propio Agente de Riesgo recomienda (`RiskAgent.max_position_pct=0.25`) —
el sistema se hubiera auto-contradicho, exponiéndose de más con una mano
mientras la otra lo señala como riesgo. Se corrigió agregando el mismo
tope (`max_position_pct=0.25` por default) a
`TechnicalAgent.calculate_position_size`, verificado con la cuenta paper
real (los tres tickers quedan topados en exactamente 25% en vez de 25-50%
sin tope).

## Apéndice: prompt de investigación usado antes de ajustar el Agente Técnico

Antes de tocar `technical_agent.py` tras el primer backtest (que perdía dinero
en los tres tickers), se usó este prompt en una conversación aparte para
investigar qué usan sistemas de trading algorítmico más serios, antes de asumir
que había que agregar más indicadores sin evidencia de que ayudaran:

```
Estoy construyendo un agente técnico de trading algorítmico en Python
(parte de un sistema multi-agente más grande, con Alpaca paper trading)
que actualmente usa solo RSI (14 períodos) y cruce de medias móviles
simples (SMA 20/50) para generar señales BUY/SELL/HOLD. Backtesteé esta
lógica sobre ~2 años de datos diarios de AAPL, MSFT y TSLA y perdió
dinero en los tres casos, además de perder claramente contra un simple
buy & hold. Sospecho que los indicadores que uso son demasiado básicos
y/o generan señales con mucho ruido (cambia de opinión casi a diario).

Quiero investigar qué usan en la práctica sistemas de trading
algorítmico más serios (incluyendo los que usan IA/ML), para evaluar si
debería reemplazar o complementar mi enfoque actual. Específicamente:

1. ¿Qué indicadores técnicos y features usan modelos cuantitativos
   reales (no solo tutoriales) más allá de RSI y cruces de medias
   simples? Por ejemplo: MACD, Bollinger Bands, ATR, momentum de
   varios timeframes, volumen, microestructura, etc. ¿Cuáles tienen
   evidencia real de funcionar mejor y en qué contextos?

2. ¿Cómo evitan estos sistemas el problema de "whipsaw" (señales que
   cambian demasiado seguido y generan pérdidas por sobre-operar)?
   ¿Qué técnicas de filtrado, confirmación multi-indicador, o
   suavizado se usan?

3. ¿Qué métricas de evaluación usan para saber si una estrategia es
   buena, más allá del retorno total? (Sharpe ratio, Sortino, Calmar,
   win rate, profit factor, etc.) ¿Cuáles son estándar en la industria
   y cuáles deberían aplicar a un backtest simple como el mío?

4. En sistemas que combinan reglas técnicas con modelos de IA/ML
   (no solo LLMs para sentimiento, sino modelos predictivos de
   precio/dirección), ¿qué enfoques son realmente usados con éxito
   documentado (papers, casos de estudio, proyectos open source
   serios) versus cuáles son más marketing/hype sin evidencia?

5. Dado que mi sistema prioriza explícitamente la interpretabilidad
   (cada decisión debe tener una explicación en lenguaje natural, no
   caja negra), ¿qué mejoras razonables podría hacer a mi agente
   técnico actual sin caer en un modelo de ML complejo que pierda esa
   transparencia?

Dame una respuesta estructurada y honesta, priorizando fuentes con
evidencia real (papers, backtests documentados, proyectos open source
con historial) sobre afirmaciones sin respaldo. No necesito que me
generes código todavía, solo el research.
```

De esa investigación salieron los dos cambios aplicados (filtro ADX y eliminar
cortos) — ver [Resultados del backtest](#resultados-del-backtest-fase-5) arriba.
Las demás recomendaciones (MACD como confirmación adicional, stops basados en
ATR, análisis multitemporal) quedan como mejoras futuras posibles, no aplicadas
todavía.
