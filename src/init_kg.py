"""
Initialize the Materials Knowledge Graph with seed data
Creates the base ontology with known materials, properties, and applications
"""

import json
import os
from knowledge_graph import MaterialsKG, Hypothesis

def create_seed_data():
    """Create seed data for the knowledge graph"""
    
    # Materials data
    materials = [
        {"name": "BaTiO3", "formula": "BaTiO₃", "type": "perovskite"},
        {"name": "PbTiO3", "formula": "PbTiO₃", "type": "perovskite"},
        {"name": "SrTiO3", "formula": "SrTiO₃", "type": "perovskite"},
        {"name": "Bi2Te3", "formula": "Bi₂Te₃", "type": "chalcogenide"},
        {"name": "PbTe", "formula": "PbTe", "type": "chalcogenide"},
        {"name": "SnSe", "formula": "SnSe", "type": "chalcogenide"},
        {"name": "CH3NH3PbI3", "formula": "CH₃NH₃PbI₃", "type": "perovskite"},
        {"name": "Si", "formula": "Si", "type": "elemental"},
        {"name": "CdTe", "formula": "CdTe", "type": "compound"},
        {"name": "CsSnI3", "formula": "CsSnI₃", "type": "perovskite"},
        {"name": "CaTiO3", "formula": "CaTiO₃", "type": "perovskite"},
        {"name": "KNbO3", "formula": "KNbO₃", "type": "perovskite"},
        {"name": "Bi2Se3", "formula": "Bi₂Se₃", "type": "chalcogenide"},
        {"name": "AgSbTe2", "formula": "AgSbTe₂", "type": "chalcogenide"},
        {"name": "CIGS", "formula": "CuInGaSe₂", "type": "compound"}
    ]
    
    # Properties data
    properties = [
        {"name": "dielectric_constant", "description": "Relative permittivity of the material"},
        {"name": "band_gap", "description": "Energy gap between valence and conduction bands"},
        {"name": "thermal_conductivity", "description": "Ability to conduct heat"},
        {"name": "electrical_conductivity", "description": "Ability to conduct electricity"},
        {"name": "zt", "description": "Thermoelectric figure of merit"},
        {"name": "absorption_coefficient", "description": "Light absorption ability"},
        {"name": "formation_energy", "description": "Energy required to form the compound"},
        {"name": "melting_point", "description": "Temperature at which material melts"},
        {"name": "crystal_structure", "description": "Crystalline arrangement of atoms"},
        {"name": "magnetic_properties", "description": "Magnetic behavior of the material"}
    ]
    
    # Applications data
    applications = [
        {"name": "capacitor", "description": "Energy storage device"},
        {"name": "thermoelectric_device", "description": "Heat to electricity conversion"},
        {"name": "solar_cell", "description": "Photovoltaic energy conversion"},
        {"name": "battery", "description": "Electrochemical energy storage"},
        {"name": "sensor", "description": "Environmental or physical sensing"},
        {"name": "catalyst", "description": "Chemical reaction acceleration"},
        {"name": "superconductor", "description": "Zero electrical resistance material"},
        {"name": "semiconductor", "description": "Electronic device component"}
    ]
    
    # Known relationships
    has_property_relations = [
        # Capacitor materials
        {"material": "BaTiO3", "property": "dielectric_constant", "value": 1500, "unit": "relative", "confidence": 1.0},
        {"material": "PbTiO3", "property": "dielectric_constant", "value": 200, "unit": "relative", "confidence": 1.0},
        {"material": "SrTiO3", "property": "dielectric_constant", "value": 300, "unit": "relative", "confidence": 1.0},
        {"material": "CaTiO3", "property": "dielectric_constant", "value": 150, "unit": "relative", "confidence": 1.0},
        {"material": "KNbO3", "property": "dielectric_constant", "value": 700, "unit": "relative", "confidence": 1.0},
        
        # Thermoelectric materials
        {"material": "Bi2Te3", "property": "zt", "value": 0.8, "unit": "dimensionless", "confidence": 1.0},
        {"material": "PbTe", "property": "zt", "value": 0.8, "unit": "dimensionless", "confidence": 1.0},
        {"material": "SnSe", "property": "zt", "value": 2.6, "unit": "dimensionless", "confidence": 1.0},
        {"material": "Bi2Se3", "property": "zt", "value": 0.4, "unit": "dimensionless", "confidence": 1.0},
        {"material": "AgSbTe2", "property": "zt", "value": 1.4, "unit": "dimensionless", "confidence": 1.0},
        
        # Solar cell materials
        {"material": "CH3NH3PbI3", "property": "band_gap", "value": 1.6, "unit": "eV", "confidence": 1.0},
        {"material": "CsSnI3", "property": "band_gap", "value": 1.3, "unit": "eV", "confidence": 1.0},
        {"material": "Si", "property": "band_gap", "value": 1.1, "unit": "eV", "confidence": 1.0},
        {"material": "CdTe", "property": "band_gap", "value": 1.5, "unit": "eV", "confidence": 1.0},
        {"material": "CIGS", "property": "band_gap", "value": 1.2, "unit": "eV", "confidence": 1.0},
        
        # Additional properties
        {"material": "BaTiO3", "property": "band_gap", "value": 3.2, "unit": "eV", "confidence": 1.0},
        {"material": "SrTiO3", "property": "band_gap", "value": 3.2, "unit": "eV", "confidence": 1.0},
        {"material": "CH3NH3PbI3", "property": "absorption_coefficient", "value": 1e5, "unit": "cm⁻¹", "confidence": 1.0},
        {"material": "CsSnI3", "property": "absorption_coefficient", "value": 8e4, "unit": "cm⁻¹", "confidence": 1.0},
    ]
    
    used_in_relations = [
        # Capacitor applications
        {"material": "BaTiO3", "application": "capacitor", "confidence": 1.0, "source": "curated", "validated_by": "literature"},
        {"material": "PbTiO3", "application": "capacitor", "confidence": 1.0, "source": "curated", "validated_by": "literature"},
        
        # Thermoelectric applications
        {"material": "Bi2Te3", "application": "thermoelectric_device", "confidence": 1.0, "source": "curated", "validated_by": "literature"},
        {"material": "PbTe", "application": "thermoelectric_device", "confidence": 1.0, "source": "curated", "validated_by": "literature"},
        
        # Solar cell applications
        {"material": "CH3NH3PbI3", "application": "solar_cell", "confidence": 1.0, "source": "curated", "validated_by": "literature"},
        {"material": "Si", "application": "solar_cell", "confidence": 1.0, "source": "curated", "validated_by": "literature"},
        {"material": "CdTe", "application": "solar_cell", "confidence": 1.0, "source": "curated", "validated_by": "literature"},
        {"material": "CIGS", "application": "solar_cell", "confidence": 1.0, "source": "curated", "validated_by": "literature"},
    ]
    
    return {
        "materials": materials,
        "properties": properties,
        "applications": applications,
        "has_property_relations": has_property_relations,
        "used_in_relations": used_in_relations
    }

def initialize_knowledge_graph():
    """Initialize the knowledge graph with seed data"""
    
    print("Initializing Materials Knowledge Graph...")
    
    # Create knowledge graph instance
    kg = MaterialsKG()
    
    try:
        # Clear existing data
        print("Clearing existing data...")
        kg.clear_database()
        
        # Create constraints
        print("Creating constraints...")
        kg.create_constraints()
        
        # Get seed data
        seed_data = create_seed_data()
        
        # Add materials
        print("Adding materials...")
        for material in seed_data["materials"]:
            kg.add_material(
                material["name"], 
                material["formula"], 
                {"type": material["type"]}
            )
        
        # Add properties
        print("Adding properties...")
        for prop in seed_data["properties"]:
            kg.add_property(prop["name"], prop["description"])
        
        # Add applications
        print("Adding applications...")
        for app in seed_data["applications"]:
            kg.add_application(app["name"], app["description"])
        
        # Add HAS_PROPERTY relationships
        print("Adding material-property relationships...")
        for rel in seed_data["has_property_relations"]:
            kg.add_has_property_relationship(
                rel["material"], rel["property"], 
                rel["value"], rel["unit"], 
                "curated", rel["confidence"]
            )
        
        # Add USED_IN relationships
        print("Adding material-application relationships...")
        for rel in seed_data["used_in_relations"]:
            kg.add_used_in_relationship(
                rel["material"], rel["application"],
                rel["confidence"], rel["source"], rel["validated_by"]
            )
        
        # Print statistics
        stats = kg.get_graph_stats()
        print("\nKnowledge Graph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Save seed data to file
        os.makedirs("../data", exist_ok=True)
        with open("../data/seed_data.json", "w") as f:
            json.dump(seed_data, f, indent=2)
        
        print("\nKnowledge graph initialization completed successfully!")
        print("Seed data saved to data/seed_data.json")
        
        return True
        
    except Exception as e:
        print(f"Error initializing knowledge graph: {e}")
        return False
    
    finally:
        kg.close()

def main():
    """Main function to initialize the knowledge graph"""
    success = initialize_knowledge_graph()
    
    if success:
        print("\n✅ Knowledge graph ready for use!")
        print("You can now run the Streamlit app: streamlit run app.py")
    else:
        print("\n❌ Failed to initialize knowledge graph")
        print("Please check your Neo4j connection and try again")

if __name__ == "__main__":
    main()

