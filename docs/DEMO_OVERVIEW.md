## Materials Ontology Expansion – Demo Overview

### What this system does

A knowledge-graph powered platform for materials discovery. It connects curated data and LLM-generated hypotheses into a Neo4j graph, exposes fast query/analytics via a Streamlit UI, and provides interactive 3D/2D visualizations plus an assistant for natural-language questions.

### High-level architecture

- Data layer: Neo4j stores nodes and relationships
  - Nodes: Material, Property, Application
  - Relationships: USED_IN (Material→Application), HAS_PROPERTY (Material→Property)
- App layer: Streamlit UI (tabs for chat, queries, discovery, visualization, analytics)
- AI layer: LLM-driven hypothesis generation (default: local model through Ollama; can be configured)

### Key features (tabs)

- AI Chat Assistant
  - MatBot answers material/application questions, suggests candidates, validates ideas
  - Uses dynamic KG-driven routing (not hardcoded keywords)
- Intelligent Query
  - Materials by Application
  - Properties by Material
  - Similar Materials (shared applications/properties)
  - Discovery Gaps (under-served applications)
- Ensemble Discovery
  - Basic fast suggestions via the configured model
- 3D Visualization
  - Interactive 3D network if drawable; automatic fallback to 2D interactive graph
- Analytics Dashboard
  - Clusters (with fallback grouping if numeric features are sparse)
  - Trends and performance views
- Discovery Insights
  - Auto-generated insights and recommended actions

### Typical demo flow (5–7 minutes)

1) System status: confirm Neo4j and model availability in the sidebar
2) Load data: Seed Loader (or MatKG Builder if needed)
3) Intelligent Query:
   - Materials by Application (e.g., capacitor)
   - Properties by Material (e.g., BaTiO3)
   - Similar Materials and Discovery Gaps
4) 3D Visualization: refresh; demonstrate automatic 2D fallback if graph is sparse
5) AI Chat Assistant:
   - “What materials are used for capacitors?”
   - “Tell me about BaTiO3”
   - “Suggest materials for thermoelectric devices”
6) Analytics Dashboard: show clusters and trends
7) (Optional) Ensemble Discovery and add validated hypotheses to the KG

### Setup and run

1) Create and activate a Python virtual environment
   - macOS/Linux
     - python3 -m venv venv
     - source venv/bin/activate
   - Windows (PowerShell)
     - python -m venv venv
     - venv\\Scripts\\Activate.ps1
2) Install requirements
   - pip install -r materials_ontology_expansion/requirements_working.txt
3) Ensure Neo4j is running and set environment variables if needed
   - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
4) Start the app
   - streamlit run materials_ontology_expansion/src/enhanced_app.py --server.port 8501

### Data ingestion paths

- Seed Loader (sidebar): loads materials/properties/applications and relationships from materials_ontology_expansion/data/seed_data.json
- MatKG Builder (sidebar): optional, ingests relationships from a large CSV into a term graph

### Configuration

- materials_ontology_expansion/src/config.py reads env vars (Neo4j, LLM, MatKG options)
- Default LLM is a local model via Ollama; you can swap providers by adjusting env/config

### Important files

- Streamlit app: materials_ontology_expansion/src/enhanced_app.py
- Knowledge graph API: materials_ontology_expansion/src/knowledge_graph.py
- Visualization: materials_ontology_expansion/src/advanced_visualization.py
- Analytics and insights: materials_ontology_expansion/src/discovery_analytics.py
- Chat assistant: materials_ontology_expansion/src/chat_interface.py

### Troubleshooting

- Visualization shows a legend but no nodes
  - Use Seed Loader and then refresh visualization; the app auto-falls back to 2D if 3D has no drawable nodes
- Analytics shows zeros
  - Ensure data was ingested; run Seed Loader and revisit Analytics
- LLM suggestions are empty
  - Confirm model is available and accessible; adjust configuration or use Seed context first


