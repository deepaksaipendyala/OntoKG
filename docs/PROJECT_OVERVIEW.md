## Project Overview: AI-Accelerated Materials Ontology Expansion

### Vision

Accelerate materials discovery by unifying structured knowledge graphs with AI reasoning. The system turns scattered facts from papers, databases, and expert prompts into a continuously evolving knowledge graph (KG) that can be queried, visualized, and expanded with validated AI suggestions.

### Problem Addressed

Materials R&D faces three persistent bottlenecks:
- Knowledge fragmentation: properties and applications are buried across heterogeneous sources.
- Slow hypothesis generation: experts iterate manually and narrowly around known compounds.
- Limited reuse of prior knowledge: insights often remain siloed and are hard to integrate.

### What the System Does

- Builds and maintains a Neo4j-based KG of Materials, Properties, and Applications, with relationships such as HAS_PROPERTY and USED_IN.
- Answers natural-language questions via an assistant that routes dynamically to KG queries or AI generation without brittle keyword rules.
- Generates new material–application hypotheses using LLMs, then filters them through validation checks before adding to the KG.
- Provides interactive 3D/2D network views and analytics (clusters, trends, discovery gaps) to reveal patterns and opportunities.

### Why This Matters for Materials Science

This stack shortens the cycle from “idea” to “testable hypothesis” by:
- Making prior knowledge computable: Scientists query a live graph instead of scanning papers.
- Guiding exploration: Clusters, gaps, and similarity reveal where to look next.
- Scaling ideation: LLMs propose candidates across chemical families; validation emphasizes plausible suggestions.
- Reinforcing knowledge: Validated discoveries are re-inserted into the KG, compounding value over time.

### Technical Approach

1) Knowledge Graph Backbone
- Nodes: Material, Property, Application
- Edges: HAS_PROPERTY(Material→Property), USED_IN(Material→Application)
- Import pipelines: a curated seed and a builder for larger extractions; schema enforces unique names and consistent types.

2) AI Hypothesis Engine
- Prompts encode domain constraints (thermoelectrics, solar cells, dielectrics) and request structured outputs.
- Multi-model or single-model generation (configurable), designed to be provider-agnostic.
- Validation layer cross-checks properties, plausibility, and known patterns before KG insertion.

3) Dynamic Assistant and Queries
- Entity-driven routing finds materials/applications mentioned in a question and runs the right KG query or suggestion routine.
- No hardcoded keyword lists, reducing maintenance and improving robustness to phrasing.

4) Visualization and Analytics
- 3D network view with automatic fallback to interactive 2D to ensure a render even with sparse data.
- Clustering with numerical-feature extraction; fallback grouping when features are limited.
- Discovery gap detection for under-served applications; similarity search to find analogues.

### Novelty

- Closed-loop KG + LLMs: Hypotheses are generated from the KG’s context and, once validated, fed back to expand the KG. This creates a virtuous cycle of knowledge accumulation.
- Dynamic semantic routing: The assistant maps user queries to graph entities on the fly, eliminating brittle intent taxonomies and enabling domain growth without rule rewrites.
- Schema-consistent ingest and visualization: The system normalizes diverse inputs into a stable graph schema and ensures visualizations render even with heterogeneous data quality.
- Practical validation emphasis: Instead of purely generative suggestions, the pipeline prioritizes plausibility, confidence, and provenance, aligning with scientific workflows.

### Impact Metrics (Indicative)

- Literature triage time: Reduced by enabling targeted KG queries and similarity search.
- Hypothesis throughput: More candidate materials tested per week due to AI assistance.
- Reuse of knowledge: Validated results persist in the KG and inform future AI runs, lifting overall precision over time.

### Example User Journeys

- Application-first: “What materials are used for capacitors?” → ranked list from the KG → AI suggests analogues → validate top candidates → add back to the KG.
- Material-first: “Tell me about BaTiO3” → properties and application edges → find similar materials and gaps → propose new used-in hypotheses for close applications.

### Extensibility

- Data sources: Add loaders for Materials Project, Matbench, literature-mined edges; they normalize into the same schema.
- Models: Swap LLM backends by configuration; prompts and validators stay consistent.
- Schema: Introduce new node/edge types (e.g., Process, Instrument, Dataset) while preserving existing queries.

### Responsible Use and Limitations

- LLM suggestions are hypotheses, not facts; the system surfaces confidence and provenance and encourages validation.
- Data incompleteness can bias clusters and trends; dashboards indicate when fallbacks are used.
- The approach complements, not replaces, high-fidelity simulation and experimentation.

### Summary

By coupling a living knowledge graph with AI-driven proposal and validation loops, the platform converts fragmented materials knowledge into actionable discovery workflows. This lowers the barrier to exploring new chemical spaces, improves reusability of prior insights, and accelerates the path from hypothesis to experimentally testable candidates.


