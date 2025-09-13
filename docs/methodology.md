# Methodology: Chemical Knowledge Graph-Guided Materials Ontology Expansion

## Overview

This document describes the detailed methodology for our LLM-guided materials ontology expansion system. The approach combines the pattern recognition capabilities of Large Language Models with the rigor of structured knowledge graphs and data validation.

## System Architecture

### Core Components

1. **Knowledge Graph (Neo4j)**: Stores structured materials knowledge with nodes (materials, properties, applications) and relationships (HAS_PROPERTY, USED_IN)
2. **LLM Integration (Ollama)**: Generates hypotheses about missing material-application relationships
3. **Validation Module**: Verifies hypotheses against materials databases and property thresholds
4. **Web Interface (Streamlit)**: Provides interactive access to the system

### Data Flow

```
Base Knowledge Graph → LLM Analysis → Hypothesis Generation → Validation → Graph Expansion
```

## Detailed Process

### 1. Base Ontology Construction

#### Node Types
- **Materials**: Chemical compounds with formula and type (perovskite, chalcogenide, etc.)
- **Properties**: Physical/chemical properties (dielectric constant, band gap, ZT, etc.)
- **Applications**: End-use applications (capacitor, solar cell, thermoelectric device, etc.)

#### Relationship Types
- **HAS_PROPERTY**: Material → Property (with value, unit, confidence, source)
- **USED_IN**: Material → Application (with confidence, source, validated_by)

#### Seed Data Sources
- Curated literature data for well-known materials
- Materials Project database integration
- Matbench property datasets
- Domain expert knowledge

### 2. LLM-Guided Hypothesis Generation

#### Prompt Engineering Strategy

**System Prompt Structure**:
```
You are a materials science expert. Analyze patterns in materials used for specific applications and suggest new materials based on scientific reasoning. Return your response in JSON format with:
{
  "hypotheses": [
    {
      "material": "Material Name",
      "application": "Application Name", 
      "relationship": "USED_IN",
      "rationale": "Scientific reasoning",
      "confidence": 0.8
    }
  ],
  "reasoning": "Overall analysis",
  "confidence": 0.7
}
```

**Context Injection**:
- Known materials for the target application
- Related properties and their values
- Chemical/structural similarities
- Processing considerations

#### Application-Specific Prompts

**Capacitor Materials**:
- Focus on high dielectric constant materials
- Perovskite structure patterns
- Ferroelectric properties
- Chemical similarity to known capacitor materials

**Thermoelectric Materials**:
- Low thermal conductivity requirements
- Chalcogenide family patterns
- ZT figure of merit considerations
- Complex crystal structures for phonon scattering

**Solar Cell Materials**:
- Band gap optimization (1.0-1.8 eV)
- Direct band gap preference
- Absorption coefficient requirements
- Stability under solar irradiation

### 3. Validation Framework

#### Validation Criteria

**Capacitor Materials**:
- Dielectric constant ≥ 50 (relative permittivity)
- Perovskite or ferroelectric structure preferred
- High breakdown voltage capability

**Thermoelectric Materials**:
- ZT ≥ 0.5 (dimensionless figure of merit)
- Low thermal conductivity (< 2 W/m·K)
- Chalcogenide or skutterudite structure preferred

**Solar Cell Materials**:
- Band gap between 1.0-1.8 eV
- Direct band gap semiconductor
- High absorption coefficient (> 10⁴ cm⁻¹)

#### Validation Sources

**Primary Sources**:
- Matbench datasets (dielectric, band gap, formation energy)
- Materials Project API
- AFLOW database
- Literature references with DOI tracking

**Secondary Sources**:
- Domain expert validation
- Cross-reference with multiple databases
- Property consistency checks

#### Confidence Scoring

**Confidence Calculation**:
```
confidence = (property_match_score + structure_score + literature_score) / 3
```

Where:
- `property_match_score`: How well properties match criteria (0-1)
- `structure_score`: Structural similarity to known materials (0-1)
- `literature_score`: Literature evidence strength (0-1)

### 4. Knowledge Graph Expansion

#### Edge Addition Process

1. **Validation Check**: Verify hypothesis meets criteria
2. **Confidence Assessment**: Calculate combined LLM + validation confidence
3. **Provenance Tracking**: Record source, validation method, timestamp
4. **Graph Update**: Add edge with metadata
5. **Consistency Check**: Verify no conflicts with existing knowledge

#### Metadata Schema

**Edge Properties**:
```cypher
{
  confidence: Float (0.0-1.0),
  source: String ("LLM", "curated", "literature"),
  validated_by: String ("Matbench", "MaterialsProject", "literature"),
  created_at: DateTime,
  rationale: String (LLM reasoning),
  evidence: Object (validation evidence)
}
```

### 5. Quality Assurance

#### Validation Metrics

**Precision**: Percentage of added edges that are correct
**Recall**: Percentage of valid relationships discovered
**Coverage**: Expansion in number of materials per application
**Consistency**: Absence of contradictory relationships

#### Error Detection

**Conflicting Relationships**:
- Same material with contradictory properties
- Impossible property combinations
- Inconsistent confidence scores

**Hallucination Detection**:
- Cross-validation across multiple sources
- Property value sanity checks
- Literature DOI verification

### 6. Iterative Improvement

#### Active Learning Loop

1. **Pattern Analysis**: Identify gaps in current knowledge
2. **Targeted Prompting**: Focus LLM on specific application domains
3. **Validation Feedback**: Use validation results to improve prompts
4. **Knowledge Refinement**: Update validation criteria based on results

#### Continuous Learning

**Prompt Refinement**:
- Analyze successful vs failed hypotheses
- Update prompt templates based on performance
- Incorporate new validation criteria

**Validation Enhancement**:
- Expand materials property database
- Add new validation sources
- Refine confidence scoring algorithms

## Implementation Details

### Database Schema

```cypher
// Node Labels
(:Material {name, formula, type})
(:Property {name, description})
(:Application {name, description})

// Relationship Types
(:Material)-[:HAS_PROPERTY {value, unit, confidence, source}]->(:Property)
(:Material)-[:USED_IN {confidence, source, validated_by}]->(:Application)
```

### API Integration

**Ollama Integration**:
- REST API calls to local Ollama instance
- Model selection (llama3.1, mistral, etc.)
- Temperature and top-p parameter tuning
- Response parsing and error handling

**Neo4j Integration**:
- Bolt protocol connection
- Cypher query execution
- Transaction management
- Constraint creation and maintenance

### Performance Optimization

**Caching Strategy**:
- Validation result caching
- LLM response caching for similar queries
- Graph statistics caching

**Batch Processing**:
- Bulk hypothesis generation
- Batch validation operations
- Bulk graph updates

## Evaluation Methodology

### Benchmark Datasets

**Test Cases**:
- Known material-application pairs (positive examples)
- Invalid material-application pairs (negative examples)
- Edge cases and boundary conditions

**Metrics**:
- Accuracy: Correct predictions / Total predictions
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1-Score: 2 × (Precision × Recall) / (Precision + Recall)

### Comparison Baselines

**Baseline Methods**:
- Random material-application assignment
- Rule-based matching (property thresholds only)
- Pure LLM predictions without validation
- Human expert curation

**Evaluation Domains**:
- Capacitor materials (dielectric constant focus)
- Thermoelectric materials (ZT optimization)
- Solar cell materials (band gap optimization)

## Limitations and Future Work

### Current Limitations

1. **Limited Material Coverage**: Database contains ~50 materials vs thousands in real databases
2. **Simplified Validation**: Basic property threshold checking vs complex multi-criteria validation
3. **Static Validation Criteria**: Fixed thresholds vs adaptive criteria
4. **Limited LLM Models**: Primarily tested with local Ollama models

### Future Enhancements

1. **Database Expansion**: Integration with full Materials Project, AFLOW, and JARVIS databases
2. **Advanced Validation**: Machine learning-based validation models
3. **Multi-Modal Integration**: Include crystal structure, phase diagrams, and synthesis routes
4. **Real-Time Updates**: Continuous learning from new literature and databases
5. **Domain Extension**: Apply methodology to other materials science domains

### Research Directions

1. **Explainable AI**: Develop interpretable hypothesis generation and validation
2. **Uncertainty Quantification**: Better confidence estimation and uncertainty propagation
3. **Active Learning**: Intelligent selection of materials for experimental validation
4. **Knowledge Graph Embeddings**: Use graph neural networks for better pattern recognition

## Conclusion

This methodology provides a systematic approach to expanding materials science knowledge graphs using LLM-guided hypothesis generation and rigorous validation. The combination of statistical pattern recognition (LLM) with symbolic reasoning (knowledge graph) and empirical validation creates a robust framework for scientific knowledge discovery.

The iterative nature of the system allows for continuous improvement, making it a valuable tool for materials researchers and a stepping stone toward more sophisticated AI-assisted materials discovery systems.

