# Tech Interview AI

Simulador de entrevistas técnicas con IA local. El sistema presenta 3 preguntas adaptativas (junior · mid · senior), evalúa las respuestas con embeddings semánticos y un LLM local (Ollama), y entrega retroalimentación detallada al finalizar la entrevista.

Proyecto académico — Universidad ECCI.

---

## Tecnologías

| Componente | Tecnología |
|---|---|
| Backend | FastAPI + Uvicorn |
| LLM local | Ollama (`qwen2.5:3b`) |
| Embeddings | Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Base de datos | SQLite (SQLAlchemy ORM) |
| Frontend | Jinja2 Templates + CSS personalizado |
| Autenticación | Sesiones con cookie + bcrypt |

---

## Requisitos previos

- Python 3.10+
- [Ollama](https://ollama.com) instalado

---

## Inicio rápido

```bash
# 1. Descargar el modelo LLM (solo la primera vez)
ollama pull qwen2.5:3b

# 2. Iniciar Ollama (en una terminal aparte)
ollama serve

# 3. Lanzar la aplicación
./run.sh
```

Abre `http://localhost:2026` en tu navegador.

> `run.sh` crea el entorno virtual e instala todas las dependencias automáticamente en la primera ejecución.

---

## Instalación manual

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 2026
```

---

## Flujo de la entrevista

1. El usuario se registra o inicia sesión
2. Selecciona un área de enfoque (opcional)
3. Responde **3 preguntas** en secuencia: 1 junior → 1 mid → 1 senior
4. Al finalizar, recibe el **resumen completo** con:
   - Nivel detectado según el rendimiento por dificultad
   - Retroalimentación del LLM por cada pregunta
   - Fortalezas y áreas de mejora
   - Keywords cubiertas y faltantes
   - Comparación respuesta del usuario vs respuesta ideal

---

## Cómo funciona la evaluación

Para cada respuesta el sistema calcula:

1. **Similitud semántica** — Sentence-BERT convierte la respuesta y la respuesta ideal en vectores y calcula la distancia coseno
2. **Cobertura de keywords** — Compara las 5 keywords más importantes de la pregunta contra la respuesta del usuario (normalizada: cubrir el 60% da score completo)
3. **Feedback LLM** — `qwen2.5:3b` genera feedback estructurado en JSON con score, fortalezas, mejoras y nivel detectado
4. **Score de calidad compuesta**:

```
Calidad = 0.25×Relevancia + 0.20×Coherencia + 0.30×Similitud + 0.25×Exactitud
```

---

## Estructura del proyecto

```
tech-interview-ai-uni/
├── main.py                        # FastAPI — rutas y controladores
├── run.sh                         # Script de inicio con venv automático
├── requirements.txt
├── benchmark.py                   # Benchmark comparativo de modelos Ollama
├── .env.example                   # Variables de entorno de ejemplo
├── app/
│   ├── ai/
│   │   ├── ollama_client.py       # Cliente HTTP para Ollama API
│   │   ├── evaluator.py           # Motor de evaluación principal
│   │   ├── embeddings.py          # Similitud semántica (Sentence-BERT)
│   │   ├── metrics.py             # Métricas ponderadas de calidad
│   │   └── prompts.py             # Prompts del sistema
│   ├── auth/
│   │   └── auth.py                # Hash de contraseñas (bcrypt)
│   ├── core/
│   │   └── logging.py             # Configuración de logs
│   ├── data/
│   │   └── questions.json         # Dataset: 93 preguntas técnicas
│   ├── db/
│   │   ├── database.py            # Conexión SQLite + SQLAlchemy
│   │   ├── models.py              # Modelos ORM: User, Evaluation
│   │   └── crud.py                # Operaciones de base de datos
│   ├── static/css/style.css       # Tema oscuro
│   └── templates/                 # Vistas Jinja2
│       ├── index.html             # Pantalla de inicio / configuración
│       ├── question.html          # Pregunta activa con progreso
│       ├── summary.html           # Resultado final con feedback completo
│       ├── history.html           # Historial de evaluaciones
│       ├── result.html            # Resultado individual (modo libre)
│       ├── generate.html          # Generador de preguntas con IA
│       ├── login.html
│       └── register.html
└── services/
    ├── questions.py               # Selección balanceada del dataset
    └── generator.py               # Generación dinámica de preguntas
```

---

## Dataset

93 preguntas técnicas distribuidas en:

| Dificultad | Cantidad |
|---|---|
| Junior | 30 |
| Mid | 33 |
| Senior | 30 |

| Área | Cantidad |
|---|---|
| Backend | 16 |
| Frontend | 16 |
| Data / ML | 14 |
| Sistemas | 13 |
| Seguridad | 10 |
| DevOps | 9 |
| Arquitectura | 9 |
| Mobile | 6 |

---

## Benchmark de modelos

Compara la precisión, velocidad y calidad de respuesta de distintos modelos Ollama:

```bash
python benchmark.py
```

Con exportación a CSV y JSON:

```bash
python benchmark.py --export
```

Modelos comparados por defecto: `qwen2.5:3b`, `llama3.2:3b`, `gemma3:4b`, `qwen2.5:7b`

Métricas evaluadas: exactitud, alucinaciones, consistencia, velocidad, JSON válido.

---

## Variables de entorno

Copia `.env.example` a `.env` para personalizar:

```bash
cp .env.example .env
```

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | URL de la API de Ollama |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Modelo a usar para evaluación |
| `SECRET_KEY` | *(generado)* | Clave de sesión (cambiar en producción) |

---

## Endpoints principales

| Ruta | Método | Descripción |
|---|---|---|
| `/` | GET | Pantalla de inicio |
| `/interview/start` | POST | Inicia una nueva entrevista |
| `/question` | GET | Pregunta activa |
| `/evaluate` | POST | Evalúa la respuesta |
| `/next` | GET | Avanza a la siguiente pregunta |
| `/summary` | GET | Resultado final con feedback |
| `/history` | GET | Historial de evaluaciones |
| `/generate` | GET/POST | Generador libre de preguntas |
| `/health` | GET | Estado de Ollama (JSON) |
| `/register` | GET/POST | Registro de usuario |
| `/login` | GET/POST | Inicio de sesión |
| `/logout` | GET | Cierre de sesión |
