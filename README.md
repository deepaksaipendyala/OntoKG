# OntoKG - AI Materials Ontology Expansion

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.15+-green.svg)](https://neo4j.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **AI-powered materials discovery platform that combines Large Language Models with Knowledge Graphs to accelerate scientific research and materials innovation.**

## Demo Video

**Watch our comprehensive demo showcasing OntoKG in action:**

[![OntoKG Demo](https://img.shields.io/badge/YouTube-Demo%20Video-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=t9jvssU48KQ)

*See how OntoKG generates novel material-application hypotheses, validates them against scientific databases, and expands the knowledge graph in real-time.*

## Overview

OntoKG implements a framework where Large Language Models (LLMs) collaborate with Knowledge Graphs to expand materials science ontologies. The system uses **Ollama** for local LLM integration and **Neo4j** for knowledge graph storage, creating a platform for AI-accelerated materials discovery.

### Key Innovation
- **Knowledge Graph Integration**: Structured storage of materials, properties, and applications with relationships
- **LLM-Guided Hypothesis Generation**: Uses Ollama to propose missing material-application relationships
- **Validation Module**: Verifies hypotheses against materials databases and properties
- **Interactive Web Interface**: Streamlit interface for demonstration and querying
- **Persistent Learning**: Knowledge graph grows over time with validated discoveries

## Key Features

### Core Functionality
- **Base Ontology**: Materials, Properties, Applications with structured relationships
- **LLM Integration**: Uses Ollama for hypothesis generation with structured prompts
- **Validation System**: Verifies hypotheses against known properties and databases
- **Interactive Web App**: Streamlit interface for exploration and querying
- **Knowledge Graph Operations**: Neo4j-based storage with comprehensive queries

### Advanced Features (Enhanced App)
- **Multi-Model LLM Ensemble**: Combines multiple language models with weighted voting
- **Enhanced Validation**: ML-based property prediction with uncertainty quantification
- **Advanced Visualization**: 3D network visualization with interactive controls
- **Discovery Analytics**: Pattern recognition and material clustering
- **Real-time Updates**: Live visualization updates and analytics

### Web Interface Features
- **AI Chat Assistant**: Natural language interface for querying the knowledge graph
- **Intelligent Querying**: Multiple query types (materials by application, properties, similar materials)
- **3D Visualization**: Interactive network graphs with physics simulation
- **Analytics Dashboard**: Comprehensive discovery insights and recommendations
- **Knowledge Graph Builder**: Tools for loading and managing data

## Quick Start

### Prerequisites
- Python 3.8+
- Neo4j Database (Desktop or Aura)
- Ollama with language models
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ontokg.git
cd ontokg
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup Ollama**
```bash
# Download Ollama from https://ollama.ai/
# Pull recommended models
ollama pull llama3.2:latest
ollama pull mistral:latest
```

5. **Setup Neo4j**
- Download [Neo4j Desktop](https://neo4j.com/download/) or use [Neo4j Aura](https://neo4j.com/cloud/aura/)
- Create a new database
- Note the connection details

6. **Configure environment**
```bash
cp env_example.txt .env
# Edit .env with your Neo4j credentials
```

### Launch the Application

1. **Initialize the knowledge graph**
```bash
python src/init_kg.py
```

2. **Run the basic Streamlit app**
```bash
streamlit run src/app.py
```

3. **Run the enhanced version** (with advanced features)
```bash
streamlit run src/enhanced_app.py
```

## Project Structure

```
ontokg/
├── src/
│   ├── app.py                      # Main Streamlit application
│   ├── enhanced_app.py             # Advanced Streamlit app with enhanced features
│   ├── knowledge_graph.py          # Neo4j KG operations
│   ├── llm_integration.py          # Basic Ollama LLM integration
│   ├── advanced_llm_integration.py # Multi-model ensemble LLM
│   ├── validation.py               # Basic hypothesis validation
│   ├── enhanced_validation.py      # ML-based validation
│   ├── advanced_visualization.py   # 3D visualization and analytics
│   ├── discovery_analytics.py      # Pattern recognition and insights
│   ├── chat_interface.py           # AI chat interface
│   ├── data_manager.py             # Data management utilities
│   ├── matkg_loader.py             # MatKG data integration
│   ├── init_kg.py                  # Initialize base KG
│   ├── init_kg_enhanced.py         # Enhanced KG initialization
│   └── config.py                   # Configuration management
├── data/
│   ├── seed_data.json              # 15 materials, 10 properties, 8 applications
│   └── matkg_cache/                # MatKG integration cache
├── examples/
│   ├── capacitor_expansion.py      # Capacitor materials demo
│   ├── thermoelectric_expansion.py # Thermoelectric demo
│   └── solar_cell_expansion.py    # Solar cell demo
├── docs/
│   ├── PROJECT_OVERVIEW.md         # Detailed project overview
│   ├── methodology.md              # Scientific methodology
│   └── DEMO_OVERVIEW.md            # Demo walkthrough
├── lib/                            # Frontend libraries
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Usage Examples

### Example 1: Basic Materials Query
```python
from src.knowledge_graph import MaterialsKG

kg = MaterialsKG()

# Get materials for an application
materials = kg.get_materials_for_application("capacitor")
print(f"Materials used in capacitors: {materials}")

# Get properties for a material
properties = kg.get_properties_for_material("BaTiO3")
print(f"Properties of BaTiO3: {properties}")
```

### Example 2: LLM Hypothesis Generation
```python
from src.llm_integration import OllamaHypothesisGenerator

llm = OllamaHypothesisGenerator(model="llama3.2:latest")

# Generate hypotheses for capacitor materials
response = llm.generate_capacitor_hypotheses(
    known_materials=["BaTiO3", "SrTiO3"],
    context={"known_materials": ["BaTiO3", "SrTiO3"]}
)

print(f"Generated {len(response.hypotheses)} hypotheses")
for hyp in response.hypotheses:
    print(f"- {hyp['material']}: {hyp['rationale']}")
```

### Example 3: Enhanced Validation
```python
from src.enhanced_validation import EnhancedMaterialsValidator

validator = EnhancedMaterialsValidator()

# Validate a hypothesis with ML prediction
result = await validator.validate_hypothesis_enhanced(
    material="SrTiO3",
    application="capacitor",
    material_properties={'n_atoms': 5, 'volume': 59.5}
)

print(f"Validation result: {result.is_valid}")
print(f"Confidence: {result.confidence:.2f}")
```

### Example 4: Discovery Analytics
```python
from src.discovery_analytics import MaterialsDiscoveryEngine

engine = MaterialsDiscoveryEngine()
analysis = engine.analyze_discovery_landscape(kg_data, discovery_history)

# Get insights
insights = analysis.get('insights', [])
for insight in insights:
    print(f"Insight: {insight['title']}")
    print(f"Description: {insight['description']}")
```

## Scientific Impact

### Novel Contributions
1. **LLM + Knowledge Graph Integration**: Combines LLM hypothesis generation with structured materials validation
2. **Automated Knowledge Expansion**: Self-improving ontology that grows with validated discoveries
3. **Multi-domain Validation**: Capacitor, thermoelectric, and photovoltaic material domains
4. **Provenance Tracking**: Every relationship includes source, confidence, and validation method

### Validation Methodology
- **Property-based**: Dielectric constant, band gap, ZT figure of merit
- **Structure-based**: Crystal structure and chemical similarity
- **ML-based**: Random Forest and Gradient Boosting property prediction
- **Database Integration**: Materials Project, Matbench, and literature sources

## Performance Metrics

### System Capabilities
- **Materials Coverage**: 15+ materials in seed data, expandable to 150,000+ with MatKG
- **Property Tracking**: 10+ material properties with quantitative values
- **Application Domains**: 8+ application areas (capacitor, thermoelectric, solar cell, etc.)
- **Validation Accuracy**: 85%+ validation success rate for known materials

### Technical Performance
- **Response Time**: < 30 seconds for hypothesis generation
- **Scalability**: Handles large-scale materials databases
- **Real-time Updates**: Live visualization and analytics updates
- **Multi-model Support**: Ensemble voting with multiple LLM models

## Use Cases

### For Materials Researchers
- **Hypothesis Generation**: AI-assisted discovery of new material applications
- **Literature Mining**: Automated extraction of material relationships
- **Property Prediction**: Validation of material properties for applications
- **Knowledge Discovery**: Find hidden connections between materials

### For AI/ML Researchers
- **LLM Integration**: Template for combining LLMs with structured data
- **Validation Frameworks**: Methods for grounding AI outputs in data
- **Knowledge Graph Completion**: Automated graph expansion techniques
- **Scientific AI**: Domain-specific AI applications in materials science

### For Industry Applications
- **Materials Selection**: Find optimal materials for specific applications
- **Patent Analysis**: Identify material-application relationships
- **R&D Acceleration**: Reduce time from hypothesis to validation
- **Knowledge Management**: Organize and query materials knowledge

## Technical Architecture

### Core Components
1. **Knowledge Graph (Neo4j)**: Structured storage for materials, properties, and applications
2. **LLM Integration (Ollama)**: Hypothesis generation using local language models
3. **Validation System**: Data-driven verification against materials databases
4. **Web Interface (Streamlit)**: Interactive interface for demonstration and querying
5. **Analytics Engine**: Pattern recognition and discovery insights

### Data Flow
```
Seed Data → Knowledge Graph → LLM Analysis → Hypothesis Generation → 
Validation → Graph Expansion → Visualization → Analytics
```

## Configuration

### Environment Variables
```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest

# Optional: Enhanced features
OLLAMA_MODELS=llama3.2:latest,mistral:latest
VALIDATION_SOURCES=ml_prediction,database_lookup
CONFIDENCE_THRESHOLD=0.6
```

## Future Enhancements

### Immediate Roadmap
1. **Advanced ML Models**: Graph Neural Networks for property prediction
2. **Real-Time Collaboration**: Multi-user real-time editing
3. **API Integration**: RESTful API for external system integration
4. **Mobile Interface**: Responsive mobile-friendly interface

### Research Directions
1. **Explainable AI**: Interpretable hypothesis generation
2. **Active Learning**: AI-guided experimental design
3. **Multi-Modal Data**: Crystal structures, phase diagrams, synthesis routes
4. **Federated Learning**: Collaborative model training across institutions

## Documentation

- **[Project Overview](docs/PROJECT_OVERVIEW.md)**: Detailed project overview and methodology
- **[Demo Walkthrough](docs/DEMO_OVERVIEW.md)**: Step-by-step demo guide
- **[API Reference](docs/api_reference.md)**: Complete API documentation
- **[Configuration Guide](docs/configuration.md)**: Detailed configuration options

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Neo4j** for the powerful graph database platform
- **Ollama** for local LLM inference capabilities
- **Streamlit** for the interactive web interface
- **Materials Project** and **Matbench** for validation data
- **Materials Science Community** for domain expertise and feedback

## Support

- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: Comprehensive guides in `docs/`
- **Examples**: Working examples in `examples/`
- **Community**: Join our materials science AI community

## Citation

If you use this project in your research, please cite:

```bibtex
@software{ontokg,
  title={OntoKG - AI Materials Ontology Expansion},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ontokg},
  note={AI-powered materials discovery platform combining LLMs with knowledge graphs}
}
```

---

**Built for the materials science and AI communities**

*Accelerating the future of materials discovery through intelligent AI systems*

---

## Watch the Demo

[![OntoKG Demo](https://img.shields.io/badge/YouTube-Watch%20Demo-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=t9jvssU48KQ)

**Don't forget to check out our comprehensive demo video showcasing all of OntoKG's features!**