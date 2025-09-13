"""
Knowledge Graph operations for Materials Ontology Expansion
Handles Neo4j database operations, entity management, and graph queries
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
import pandas as pd
from dataclasses import dataclass
from dotenv import load_dotenv
import logging

from config import SystemConfig

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class Hypothesis:
    """Represents a proposed relationship between entities"""
    material: str
    application: str
    relationship: str
    confidence: float = 0.0
    source: str = "LLM"
    validated_by: Optional[str] = None
    rationale: Optional[str] = None


class MaterialsKG:
    """Materials Knowledge Graph using Neo4j"""
    
    def __init__(self, config: SystemConfig = None, uri: str = None, user: str = None, password: str = None):
        if config:
            self.config = config
            self.uri = config.neo4j_uri
            self.user = config.neo4j_user
            self.password = config.neo4j_password
            # Import DataManager here to avoid circular import
            from data_manager import DataManager
            self.data_manager = DataManager(config)
        else:
            # Legacy initialization for backward compatibility
            self.config = None
            self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            self.user = user or os.getenv('NEO4J_USER', 'neo4j')
            self.password = password or os.getenv('NEO4J_PASSWORD', '123123123')
            self.data_manager = None
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def initialize_from_data_sources(self):
        """Initialize knowledge graph from all configured data sources"""
        if not self.data_manager:
            logger.warning("No data manager available. Use legacy initialization.")
            return
        
        logger.info("Initializing knowledge graph from data sources...")
        
        # Create constraints first
        self.create_constraints()
        
        # Load entities
        entity_count = 0
        for entity_data in self.data_manager.load_all_entities():
            try:
                self._add_entity_from_data(entity_data)
                entity_count += 1
                
                if entity_count % 1000 == 0:
                    logger.info(f"Loaded {entity_count} entities...")
                    
            except Exception as e:
                logger.error(f"Error adding entity {entity_data}: {e}")
        
        # Load relationships
        relationship_count = 0
        for rel_data in self.data_manager.load_all_relationships():
            try:
                self._add_relationship_from_data(rel_data)
                relationship_count += 1
                
                if relationship_count % 1000 == 0:
                    logger.info(f"Loaded {relationship_count} relationships...")
                    
            except Exception as e:
                logger.error(f"Error adding relationship {rel_data}: {e}")
        
        logger.info(f"Initialization complete: {entity_count} entities, {relationship_count} relationships")
    
    def _add_entity_from_data(self, entity_data: Dict[str, Any]):
        """Add entity from data source"""
        entity_type = entity_data.get('type', 'Unknown')
        name = entity_data.get('name', '')
        
        if not name:
            return
        
        with self.driver.session() as session:
            if entity_type == 'Material':
                self.add_material(
                    name=name,
                    formula=entity_data.get('formula'),
                    properties=entity_data.get('metadata', {})
                )
            elif entity_type == 'Property':
                self.add_property(
                    name=name,
                    description=entity_data.get('description')
                )
            elif entity_type == 'Application':
                self.add_application(
                    name=name,
                    description=entity_data.get('description')
                )
    
    def _add_relationship_from_data(self, rel_data: Dict[str, Any]):
        """Add relationship from data source"""
        subject = rel_data.get('subject', '')
        predicate = rel_data.get('predicate', '')
        object_uri = rel_data.get('object', '')
        confidence = rel_data.get('confidence', 1.0)
        source = rel_data.get('source', 'Unknown')
        
        if not all([subject, predicate, object_uri]):
            return
        
        with self.driver.session() as session:
            if predicate == 'USED_IN':
                self.add_used_in_relationship(
                    material=subject,
                    application=object_uri,
                    confidence=confidence,
                    source=source,
                    validated_by=source
                )
            elif predicate == 'HAS_PROPERTY':
                # Extract property value from metadata if available
                metadata = rel_data.get('metadata', {})
                value = metadata.get('value')
                unit = metadata.get('unit')
                
                self.add_has_property_relationship(
                    material=subject,
                    property_name=object_uri,
                    value=value,
                    unit=unit,
                    source=source,
                    confidence=confidence
                )
    
    def create_constraints(self):
        """Create unique constraints for entities"""
        constraints = [
            "CREATE CONSTRAINT material_name IF NOT EXISTS FOR (m:Material) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT property_name IF NOT EXISTS FOR (p:Property) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT application_name IF NOT EXISTS FOR (a:Application) REQUIRE a.name IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"Constraint creation warning: {e}")
    
    def add_material(self, name: str, formula: str = None, properties: Dict[str, Any] = None):
        """Add a material node to the knowledge graph"""
        with self.driver.session() as session:
            query = """
            MERGE (m:Material {name: $name})
            SET m.formula = $formula
            """
            if properties:
                for key, value in properties.items():
                    query += f"SET m.{key} = ${key}\n"
            
            params = {"name": name, "formula": formula}
            if properties:
                params.update(properties)
                
            session.run(query, params)
    
    def add_property(self, name: str, description: str = None):
        """Add a property node to the knowledge graph"""
        with self.driver.session() as session:
            query = """
            MERGE (p:Property {name: $name})
            SET p.description = $description
            """
            session.run(query, {"name": name, "description": description})
    
    def add_application(self, name: str, description: str = None):
        """Add an application node to the knowledge graph"""
        with self.driver.session() as session:
            query = """
            MERGE (a:Application {name: $name})
            SET a.description = $description
            """
            session.run(query, {"name": name, "description": description})
    
    def add_has_property_relationship(self, material: str, property_name: str, 
                                    value: float = None, unit: str = None, 
                                    source: str = "curated", confidence: float = 1.0):
        """Add a HAS_PROPERTY relationship between material and property"""
        with self.driver.session() as session:
            query = """
            MATCH (m:Material {name: $material})
            MATCH (p:Property {name: $property})
            MERGE (m)-[r:HAS_PROPERTY]->(p)
            SET r.value = $value, r.unit = $unit, r.source = $source, 
                r.confidence = $confidence, r.created_at = datetime()
            """
            session.run(query, {
                "material": material, "property": property_name,
                "value": value, "unit": unit, "source": source, "confidence": confidence
            })
    
    def add_used_in_relationship(self, material: str, application: str,
                               confidence: float = 1.0, source: str = "curated",
                               validated_by: str = None):
        """Add a USED_IN relationship between material and application"""
        with self.driver.session() as session:
            query = """
            MATCH (m:Material {name: $material})
            MATCH (a:Application {name: $application})
            MERGE (m)-[r:USED_IN]->(a)
            SET r.confidence = $confidence, r.source = $source,
                r.validated_by = $validated_by, r.created_at = datetime()
            """
            session.run(query, {
                "material": material, "application": application,
                "confidence": confidence, "source": source, "validated_by": validated_by
            })
    
    def add_hypothesis(self, hypothesis: Hypothesis):
        """Add a validated hypothesis as a relationship"""
        if hypothesis.relationship == "USED_IN":
            self.add_used_in_relationship(
                hypothesis.material, hypothesis.application,
                hypothesis.confidence, hypothesis.source, hypothesis.validated_by
            )
        elif hypothesis.relationship == "HAS_PROPERTY":
            self.add_has_property_relationship(
                hypothesis.material, hypothesis.application,  # application is actually property name
                source=hypothesis.source, confidence=hypothesis.confidence
            )
    
    def get_materials_for_application(self, application: str) -> List[Dict[str, Any]]:
        """Get all materials used in a specific application"""
        with self.driver.session() as session:
            query = """
            MATCH (m:Material)-[r:USED_IN]->(a:Application)
            WHERE toLower(a.name) = toLower($app)
            RETURN m.name as material, r.confidence, r.source, r.validated_by,
                   toString(r.created_at) as created_at
            ORDER BY coalesce(r.confidence, 0) DESC
            """
            result = session.run(query, {"app": application})
            data = [record.data() for record in result]
            if data:
                return data

            # Fallback 1: reverse direction with explicit labels
            reverse_query = """
            MATCH (a:Application)-[r:USED_IN]->(m:Material)
            WHERE toLower(a.name) = toLower($app)
            RETURN m.name as material,
                   coalesce(r.confidence, 1.0) as confidence,
                   coalesce(r.source, 'KG') as source,
                   r.validated_by as validated_by,
                   toString(coalesce(r.created_at, datetime())) as created_at
            ORDER BY confidence DESC
            """
            result_rev = session.run(reverse_query, {"app": application})
            data_rev = [record.data() for record in result_rev]
            if data_rev:
                return data_rev

            # Fallback 2: generic Term graph created by build_matkg_neo4j (forward)
            fallback_query = """
            MATCH (m:Term)-[r]->(a:Term)
            WHERE toLower(a.name) = toLower($app)
            RETURN m.name as material,
                   coalesce(r.confidence, 1.0) as confidence,
                   coalesce(r.original_type, type(r)) as source,
                   coalesce(r.validated_by, 'TermGraph') as validated_by,
                   toString(coalesce(r.created_at, datetime())) as created_at
            ORDER BY confidence DESC
            LIMIT 100
            """
            result2 = session.run(fallback_query, {"app": application})
            data2 = [record.data() for record in result2]
            if data2:
                return data2

            # Fallback 3: generic Term graph reverse direction
            fallback_reverse = """
            MATCH (a:Term)-[r]->(m:Term)
            WHERE toLower(a.name) = toLower($app)
            RETURN m.name as material,
                   coalesce(r.confidence, 1.0) as confidence,
                   coalesce(r.original_type, type(r)) as source,
                   coalesce(r.validated_by, 'TermGraph') as validated_by,
                   toString(coalesce(r.created_at, datetime())) as created_at
            ORDER BY confidence DESC
            LIMIT 100
            """
            result3 = session.run(fallback_reverse, {"app": application})
            return [record.data() for record in result3]
    
    def get_properties_for_material(self, material: str) -> List[Dict[str, Any]]:
        """Get all properties of a material"""
        with self.driver.session() as session:
            query = """
            MATCH (m:Material {name: $material})-[r:HAS_PROPERTY]->(p:Property)
            RETURN p.name as property, r.value, r.unit, r.confidence, r.source
            ORDER BY r.confidence DESC
            """
            result = session.run(query, {"material": material})
            return [record.data() for record in result]
    
    def get_neighbors(self, entity_name: str, depth: int = 1) -> Dict[str, List[str]]:
        """Get neighboring entities for context"""
        with self.driver.session() as session:
            query = f"""
            MATCH (n {{name: $name}})-[r*1..{depth}]-(neighbor)
            RETURN DISTINCT labels(neighbor)[0] as type, neighbor.name as name
            ORDER BY type, name
            """
            result = session.run(query, {"name": entity_name})
            
            neighbors = {}
            for record in result:
                entity_type = record["type"]
                name = record["name"]
                if entity_type not in neighbors:
                    neighbors[entity_type] = []
                neighbors[entity_type].append(name)
            
            return neighbors
    
    def get_graph_stats(self) -> Dict[str, int]:
        """Get basic statistics about the knowledge graph"""
        with self.driver.session() as session:
            stats = {}
            
            # Count nodes by type
            node_types = ["Material", "Property", "Application"]
            for node_type in node_types:
                query = f"MATCH (n:{node_type}) RETURN count(n) as count"
                result = session.run(query)
                stats[f"{node_type.lower()}s"] = result.single()["count"]
            
            # Count relationships
            rel_query = "MATCH ()-[r]->() RETURN count(r) as count"
            result = session.run(rel_query)
            stats["relationships"] = result.single()["count"]
            
            return stats
    
    def search_materials(self, query: str) -> List[Dict[str, Any]]:
        """Search for materials by name or formula"""
        with self.driver.session() as session:
            cypher = """
            MATCH (m:Material)
            WHERE toLower(m.name) CONTAINS toLower($query) 
               OR toLower(m.formula) CONTAINS toLower($query)
            RETURN m.name as name, m.formula as formula
            LIMIT 20
            """
            result = session.run(cypher, {"query": query})
            return [record.data() for record in result]

    def get_similar_materials(self, material: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find materials that share applications or properties with the given material."""
        with self.driver.session() as session:
            cypher = """
            MATCH (m:Material {name: $material})
            WITH m
            MATCH (m)-[:USED_IN]->(:Application)<-[:USED_IN]-(sim:Material)
            WHERE sim.name <> m.name
            WITH m, sim, 1.0 AS app_score
            OPTIONAL MATCH (m)-[:HAS_PROPERTY]->(p:Property)<-[:HAS_PROPERTY]-(sim)
            WITH sim, app_score, count(DISTINCT p) AS shared_props
            RETURN sim.name AS material, (app_score + shared_props * 0.1) AS score
            ORDER BY score DESC
            LIMIT $limit
            """
            result = session.run(cypher, {"material": material, "limit": limit})
            return [record.data() for record in result]

    def get_discovery_gaps(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Identify applications with relatively few associated materials (gaps)."""
        with self.driver.session() as session:
            cypher = """
            MATCH (a:Application)
            OPTIONAL MATCH (m:Material)-[:USED_IN]->(a)
            WITH a, count(m) as num_materials
            WITH collect({app:a.name, count:num_materials}) as rows,
                 apoc.agg.average([x in collect(num_materials) | toFloat(x)]) as avg_count
            UNWIND rows as r
            WITH r.app as application, r.count as count, avg_count
            WHERE count < avg_count * 0.5 OR avg_count = 0
            RETURN application, count
            ORDER BY count ASC
            LIMIT $k
            """
            try:
                result = session.run(cypher, {"k": top_k})
                return [record.data() for record in result]
            except Exception:
                # Fallback without APOC: approximate average using Cypher
                cypher2 = """
                MATCH (a:Application)
                OPTIONAL MATCH (m:Material)-[:USED_IN]->(a)
                WITH a, count(m) as num_materials
                WITH collect(num_materials) as counts, collect(a.name) as apps
                WITH counts, apps, reduce(s=0.0, x in counts | s + toFloat(x)) / case when size(counts)=0 then 1 else size(counts) end as avg_count
                UNWIND range(0, size(apps)-1) as i
                WITH apps[i] as application, counts[i] as count, avg_count
                WHERE count < avg_count * 0.5 OR avg_count = 0
                RETURN application, count
                ORDER BY count ASC
                LIMIT $k
                """
                result2 = session.run(cypher2, {"k": top_k})
                return [record.data() for record in result2]
    
    def validate_hypothesis(self, hypothesis: Hypothesis) -> bool:
        """Basic validation logic - can be extended with database lookups"""
        # For now, simple validation based on material name patterns
        # In a real implementation, this would query Matbench, Materials Project, etc.
        
        if hypothesis.relationship == "USED_IN":
            # Check if material exists in our KG
            with self.driver.session() as session:
                query = "MATCH (m:Material {name: $name}) RETURN m"
                result = session.run(query, {"name": hypothesis.material})
                return result.single() is not None
        
        return True
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export graph data for visualization"""
        with self.driver.session() as session:
            # Get all nodes
            try:
                nodes_query = """
                MATCH (n)
                RETURN labels(n)[0] as type, n.name as name, n.formula as formula
                """
                nodes = [record.data() for record in session.run(nodes_query)]
            except Exception:
                nodes = []
            
            # Get all relationships
            try:
                edges_query = """
                MATCH (a)-[r]->(b)
                RETURN a.name as source, b.name as target, 
                       type(r) as relationship, r.confidence as confidence
                """
                edges = [record.data() for record in session.run(edges_query)]
            except Exception:
                edges = []

            if nodes and edges:
                return {"nodes": nodes, "edges": edges}

            # Fallback normalization for Term graph
            term_nodes_query = """
            MATCH (n:Term)
            RETURN 'Term' as type, n.name as name, null as formula
            """
            term_edges_query = """
            MATCH (a:Term)-[r]->(b:Term)
            RETURN a.name as source, b.name as target,
                   coalesce(r.original_type, type(r)) as relationship,
                   coalesce(r.confidence, 1.0) as confidence
            """
            term_nodes = [record.data() for record in session.run(term_nodes_query)]
            term_edges = [record.data() for record in session.run(term_edges_query)]
            return {"nodes": term_nodes, "edges": term_edges}

