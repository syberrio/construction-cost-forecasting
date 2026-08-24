# 🏗️ Construction Cost Forecasting

Sistema de estimación y proyección de costos de equipos de construcción
basado en precios de materias primas, con agente de IA conversacional
para consultas enriquecidas con contexto de mercado.

---

## 📋 Descripción del proyecto

Una empresa constructora necesita anticipar los costos de adquisición
de dos equipos críticos antes del inicio de cada fase del proyecto.
Este sistema resuelve el problema en dos fases:

**Fase 1 — Modelado y Forecast:**
Identifica qué materias primas explican el comportamiento de cada equipo
y proyecta sus costos futuros con intervalos de confianza correctamente
propagados via simulación Monte Carlo.

**Fase 2 — Agente de IA:**
Agente conversacional ReAct que combina los resultados del modelo con
contexto externo de mercado para responder preguntas en lenguaje natural.

---

## 🏆 Resultados clave

| | Equipo 1 | Equipo 2 |
|---|---|---|
| **Predictor principal** | Price_Y (r=0.997) | Price_Z (r=0.983) |
| **Modelo** | ARIMAX(1,1,2) | ARIMAX(0,1,1) |
| **MAPE (test)** | 1.55% | 1.75% |
| **R² (test)** | 0.928 | 0.829 |
| **Forecast sep 2023** | ~451 [414, 488] | ~933 [866, 1000] |
| **Horizonte confiable** | 1 mes | 1 mes |

> El error del modelo es **10x menor** que la volatilidad natural
> de los precios (CV 24.71% y 19.11% respectivamente).

---

## 🛠️ Stack tecnológico

### Fase 1 — ML
| Componente | Tecnología |
|---|---|
| Análisis de datos | pandas, numpy, matplotlib, seaborn, plotly |
| Modelado estadístico | statsmodels (ARIMAX), pmdarima (auto_arima) |
| Modelado ML | XGBoost, scikit-learn |
| Forecast | Monte Carlo (numpy), ARIMA propio |
| Gestión de entorno | uv |

### Fase 2 — Agente IA
| Componente | Tecnología |
|---|---|
| Orquestador agente | LangGraph (ReAct pattern) |
| Gateway LLM | LiteLLM Proxy → OpenRouter |
| Modelo LLM | NVIDIA Nemotron Ultra (free) |
| Observabilidad | LangFuse v4 (self-hosted) |
| MCP Server | FastMCP (Data Access Layer) |
| Búsqueda web | DuckDuckGo (ddgs) |
| UI | Streamlit (Dashboard + Chat) |
| Infraestructura local | Docker Compose |

### Arquitectura de código
| Patrón | Implementación |
|---|---|
| DDD | domain / application / infrastructure / presentation |
| SOLID | SRP, DIP via interfaces abstractas (ABC) |
| Clean Code | type hints, docstrings Google style, logging centralizado |
| Adapter | MCP Server desacoplado del storage backend |

---

## 📁 Estructura del proyecto

```
construction-cost-forecasting/
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Análisis exploratorio completo
│   ├── 02_feature_selection.ipynb    # Selección formal de variables
│   ├── 03_modeling.ipynb             # OLS, ARIMAX, XGBoost
│   └── 04_forecasting.ipynb          # Monte Carlo + proyecciones
│
├── src/
│   ├── domain/                       # Entidades, value objects, interfaces
│   ├── application/                  # Casos de uso
│   └── infrastructure/
│       ├── ml/                       # Feature engineering, ARIMAX, pipeline
│       ├── data/                     # CSV adapters, MCP Server
│       └── agent/                    # LangGraph, tools, LangFuse
│
├── data/
│   ├── raw/                          # CSVs originales
│   └── processed/                    # Datos limpios + forecasts
│
├── models/                           # Modelos entrenados (.pkl)
├── docs/
│   ├── informe_fase_ml.md            # Informe técnico completo
│   └── arquitectura_cloud_azure.png  # Diagrama arquitectura cloud
│
├── litellm/
│   └── config.yaml                   # Configuración LiteLLM Gateway
├── docker-compose.yml                # LiteLLM + LangFuse self-hosted
├── main.py                           # Entry point pipeline ML
└── pyproject.toml                    # Dependencias (uv)
```

---

## 🚀 Instalación y uso

### Prerrequisitos
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) instalado
- Docker Desktop (para LiteLLM + LangFuse)
- API key de [OpenRouter](https://openrouter.ai)

### 1. Clonar y configurar entorno

```bash
git clone https://github.com/syberrio/construction-cost-forecasting
cd construction-cost-forecasting

# Instalar dependencias
uv sync

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys
```

### 2. Ejecutar pipeline de ML

```bash
uv run python main.py
```

Esto ejecuta el pipeline completo:
- Carga datos históricos
- Proyecta predictoras (ARIMA)
- Genera forecast Monte Carlo (1000 simulaciones)
- Guarda resultados en `data/processed/`

### 3. Levantar infraestructura (LiteLLM + LangFuse)

```bash
docker compose up -d
```

Servicios disponibles:
- **LiteLLM Gateway:** `http://localhost:4000`
- **LangFuse UI:** `http://localhost:3000`

### 4. Levantar la aplicación Streamlit

```bash
uv run streamlit run src/presentation/streamlit_app.py
```

Abre `http://localhost:8501` en el navegador.

---

## 🔑 Variables de entorno requeridas

```bash
# OpenRouter (LLM)
OPENROUTER_API_KEY=          # API key de openrouter.ai

# LiteLLM Gateway
LITELLM_MASTER_KEY=          # Key maestra del gateway
LITELLM_BASE_URL=http://localhost:4000
LITELLM_MODEL_NAME=nvidia-nemotron-ultra-free
OPENROUTER_MODEL=openrouter/nvidia/nemotron-3-ultra-550b-a55b:free

# LangFuse (self-hosted)
LANGFUSE_PUBLIC_KEY=         # Generada localmente
LANGFUSE_SECRET_KEY=         # Generada localmente
LANGFUSE_HOST=http://localhost:3000

# Infraestructura Docker
LANGFUSE_NEXTAUTH_SECRET=
LANGFUSE_SALT=
LANGFUSE_ENCRYPTION_KEY=     # 64 caracteres hex
LANGFUSE_INIT_USER_EMAIL=
LANGFUSE_INIT_USER_PASSWORD=
CLICKHOUSE_PASSWORD=
MINIO_ROOT_USER=
MINIO_ROOT_PASSWORD=
REDIS_PASSWORD=
```

---

## 📊 Notebooks — Guía de análisis

| Notebook | Contenido | Hallazgos clave |
|---|---|---|
| **01_eda** | Calidad datos, series temporales, correlaciones, outliers, STL, estacionariedad | Price_X es ruido; Y→Eq1 r=0.997; Z→Eq2 r=0.983 |
| **02_feature_selection** | Cointegración, CCF, VIF, veredicto señal vs ruido | Cointegración confirmada; Price_X descartada formalmente |
| **03_modeling** | OLS, ARIMAX, XGBoost con variable dummy pandemia | ARIMAX ganador; dummy mejora XGBoost Eq2 (R² 0.356→0.560) |
| **04_forecasting** | ARIMA predictoras, Monte Carlo, IC corregido, horizonte | IC original subestimado 10x; horizonte confiable: 1 mes |

---

## 🤖 Agente de IA — Capacidades

El agente ReAct tiene acceso a 4 herramientas:

| Tool | Descripción |
|---|---|
| `get_forecast_data` | Forecast mensual con IC 95% |
| `get_historical_data` | Precios históricos de equipos y materias primas |
| `get_model_metrics` | Métricas OLS / ARIMAX / XGBoost comparadas |
| `web_search` | Contexto externo de mercado (DuckDuckGo) |

**Ejemplo de consultas:**
```
¿Cuánto costará el Equipo1 el próximo mes?
¿Qué materias primas explican el precio del Equipo2?
¿Cuál fue el precio histórico del Equipo1 en 2022?
¿Qué tendencias actuales del mercado de commodities pueden afectar el forecast?
```

---

## ☁️ Arquitectura Cloud (Producción)

La solución está diseñada para desplegarse en **Microsoft Azure**
aprovechando el ecosistema de la empresa:

```
Ingesta:        Microsoft Fabric Data Factory + OneLake
Procesamiento:  Azure Databricks + MLflow
Agente:         Azure Container Apps + Azure API Management
LLM:            Azure OpenAI Service (GPT-4o)
Canales:        Microsoft Teams Bot + Power Apps
Observabilidad: LangFuse + Azure Monitor + Power BI
Seguridad:      Azure AD + Key Vault + Defender for AI
CI/CD:          Azure DevOps + Terraform/Bicep (IaC)
```

Ver diagrama completo: `docs/img/arquitectura_cloud_azure_propuesta.png`
Ver justificación detallada: `docs/INFORME_TECNICO.md` sesión Arquitectura Cloud Propuesta

---

## 📄 Documentación

- **Informe técnico completo:** `docs/INFORME_TECNICO.md`
- **Diagrama arquitectura cloud:** `docs/img/arquitectura_cloud_azure_propuesta.png`


---

## 👩‍💻 Desarrollo

```bash

# Linting
uv run ruff check src/

# Kernel Jupyter para los notebooks
uv run python -m ipykernel install --user \
  --name construction-cost-forecasting \
  --display-name "Python (construction-cost-forecasting)"
