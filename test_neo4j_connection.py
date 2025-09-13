#!/usr/bin/env python3
"""
Test Neo4j connection and initialize with sample data
"""

import sys
import os
sys.path.append('src')

from neo4j import GraphDatabase
import json

def test_connection_simple():
    """Test basic Neo4j connection"""
    
    # Try different password combinations
    passwords = ['123123123', 'neo4j', 'password', '']
    
    for password in passwords:
        try:
            print(f"🔍 Trying password: {'(empty)' if not password else password}")
            
            driver = GraphDatabase.driver(
                "bolt://localhost:7687", 
                auth=("neo4j", password)
            )
            
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                
            driver.close()
            
            print(f"✅ Connection successful with password: {password}")
            return password
            
        except Exception as e:
            print(f"❌ Failed with password '{password}': {e}")
            continue
    
    print("❌ All password attempts failed")
    return None

def create_sample_data(password):
    """Create sample materials data in Neo4j"""
    
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687", 
            auth=("neo4j", password)
        )
        
        with driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            print("🧹 Cleared existing data")
            
            # Create materials
            materials = [
                {"name": "BaTiO3", "formula": "BaTiO₃", "type": "perovskite"},
                {"name": "SrTiO3", "formula": "SrTiO₃", "type": "perovskite"},
                {"name": "Bi2Te3", "formula": "Bi₂Te₃", "type": "chalcogenide"},
                {"name": "SnSe", "formula": "SnSe", "type": "chalcogenide"},
                {"name": "CH3NH3PbI3", "formula": "CH₃NH₃PbI₃", "type": "perovskite"}
            ]
            
            for material in materials:
                session.run(
                    "CREATE (m:Material {name: $name, formula: $formula, type: $type})",
                    **material
                )
            print(f"✅ Created {len(materials)} materials")
            
            # Create applications
            applications = [
                {"name": "capacitor", "description": "Energy storage device"},
                {"name": "thermoelectric_device", "description": "Heat to electricity conversion"},
                {"name": "solar_cell", "description": "Photovoltaic energy conversion"}
            ]
            
            for app in applications:
                session.run(
                    "CREATE (a:Application {name: $name, description: $description})",
                    **app
                )
            print(f"✅ Created {len(applications)} applications")
            
            # Create properties
            properties = [
                {"name": "dielectric_constant", "description": "Relative permittivity"},
                {"name": "band_gap", "description": "Energy gap between bands"},
                {"name": "zt", "description": "Thermoelectric figure of merit"}
            ]
            
            for prop in properties:
                session.run(
                    "CREATE (p:Property {name: $name, description: $description})",
                    **prop
                )
            print(f"✅ Created {len(properties)} properties")
            
            # Create relationships
            relationships = [
                # Material-Application relationships
                ("BaTiO3", "capacitor", "USED_IN", 0.9),
                ("SrTiO3", "capacitor", "USED_IN", 0.8),
                ("Bi2Te3", "thermoelectric_device", "USED_IN", 0.9),
                ("SnSe", "thermoelectric_device", "USED_IN", 0.95),
                ("CH3NH3PbI3", "solar_cell", "USED_IN", 0.85),
                
                # Material-Property relationships
                ("BaTiO3", "dielectric_constant", "HAS_PROPERTY", 1500),
                ("SrTiO3", "dielectric_constant", "HAS_PROPERTY", 300),
                ("Bi2Te3", "zt", "HAS_PROPERTY", 0.8),
                ("SnSe", "zt", "HAS_PROPERTY", 2.6),
                ("CH3NH3PbI3", "band_gap", "HAS_PROPERTY", 1.6)
            ]
            
            for source, target, rel_type, value in relationships:
                if rel_type == "USED_IN":
                    session.run("""
                        MATCH (m:Material {name: $source})
                        MATCH (a:Application {name: $target})
                        CREATE (m)-[r:USED_IN {confidence: $value, source: 'seed_data'}]->(a)
                    """, source=source, target=target, value=value)
                elif rel_type == "HAS_PROPERTY":
                    session.run("""
                        MATCH (m:Material {name: $source})
                        MATCH (p:Property {name: $target})
                        CREATE (m)-[r:HAS_PROPERTY {value: $value, source: 'seed_data'}]->(p)
                    """, source=source, target=target, value=value)
            
            print(f"✅ Created {len(relationships)} relationships")
            
            # Verify data
            result = session.run("""
                MATCH (n) 
                RETURN labels(n)[0] as type, count(n) as count 
                ORDER BY type
            """)
            
            print("\n📊 Database Statistics:")
            for record in result:
                print(f"   {record['type']}: {record['count']}")
            
            # Test a query
            result = session.run("""
                MATCH (m:Material)-[r:USED_IN]->(a:Application)
                RETURN m.name as material, a.name as application, r.confidence as confidence
                ORDER BY r.confidence DESC
            """)
            
            print("\n🔗 Sample Relationships:")
            for record in result:
                print(f"   {record['material']} → {record['application']} (confidence: {record['confidence']})")
        
        driver.close()
        print("\n✅ Knowledge graph initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

def export_graph_data(password):
    """Export graph data for testing"""
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687", 
            auth=("neo4j", password)
        )
        
        with driver.session() as session:
            # Get all nodes
            nodes_result = session.run("""
                MATCH (n) 
                RETURN labels(n)[0] as type, properties(n) as props, id(n) as id
            """)
            
            nodes = {}
            for record in nodes_result:
                node_type = record['type']
                if node_type not in nodes:
                    nodes[node_type] = []
                
                node_data = dict(record['props'])
                node_data['id'] = record['id']
                nodes[node_type].append(node_data)
            
            # Get all relationships
            edges_result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN a.name as source, b.name as target, type(r) as relationship, properties(r) as props
            """)
            
            edges = []
            for record in edges_result:
                edge_data = {
                    'source': record['source'],
                    'target': record['target'], 
                    'relationship': record['relationship']
                }
                edge_data.update(dict(record['props']))
                edges.append(edge_data)
            
            graph_data = {
                'nodes': nodes,
                'edges': edges
            }
            
            print(f"\n📦 Exported graph data:")
            print(f"   Node types: {list(nodes.keys())}")
            print(f"   Total edges: {len(edges)}")
            
            return graph_data
        
    except Exception as e:
        print(f"❌ Error exporting data: {e}")
        return None

def main():
    print("🧬 Testing Neo4j Connection for Materials Ontology Expansion")
    print("=" * 60)
    
    # Test connection
    password = test_connection_simple()
    if not password:
        print("\n❌ Could not establish Neo4j connection")
        print("Please ensure Neo4j is running and accessible at http://localhost:7474")
        return
    
    # Create sample data
    print(f"\n🚀 Initializing knowledge graph with password: {password}")
    success = create_sample_data(password)
    
    if success:
        print("\n📦 Exporting graph data for application use...")
        graph_data = export_graph_data(password)
        
        if graph_data:
            # Save to file for debugging
            with open('kg_export.json', 'w') as f:
                json.dump(graph_data, f, indent=2)
            print("💾 Graph data saved to kg_export.json")
    
    print("\n🎯 Next steps:")
    print("1. Neo4j browser: http://localhost:7474")
    print("2. Run Streamlit app: streamlit run src/app.py")
    print("3. Or enhanced app: streamlit run src/enhanced_app.py")

if __name__ == "__main__":
    main()
