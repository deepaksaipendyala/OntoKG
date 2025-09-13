"""
Quick Demo - Shows the Materials Ontology Expansion results without Neo4j
This will create a visual representation of what the system does
"""

import sys
import os
sys.path.append('src')

def create_demo_graph():
    """Create a simple text-based visualization of the knowledge graph"""
    print("🧬 Materials Ontology Expansion - Quick Demo")
    print("=" * 60)
    
    # Simulate the knowledge graph data
    materials = {
        "BaTiO3": {"type": "perovskite", "dielectric_constant": 1500, "band_gap": 3.2},
        "SrTiO3": {"type": "perovskite", "dielectric_constant": 300, "band_gap": 3.2},
        "CaTiO3": {"type": "perovskite", "dielectric_constant": 150, "band_gap": 3.5},
        "Bi2Te3": {"type": "chalcogenide", "zt": 0.8, "thermal_conductivity": 1.5},
        "SnSe": {"type": "chalcogenide", "zt": 2.6, "thermal_conductivity": 0.7},
        "CH3NH3PbI3": {"type": "perovskite", "band_gap": 1.6, "absorption_coefficient": 1e5},
        "Si": {"type": "elemental", "band_gap": 1.1, "absorption_coefficient": 1e4}
    }
    
    applications = {
        "capacitor": ["BaTiO3", "SrTiO3", "CaTiO3"],
        "thermoelectric_device": ["Bi2Te3", "SnSe"],
        "solar_cell": ["CH3NH3PbI3", "Si"]
    }
    
    print("\n📊 INITIAL KNOWLEDGE GRAPH")
    print("-" * 40)
    
    for app, mat_list in applications.items():
        print(f"\n🔹 {app.replace('_', ' ').title()}:")
        for material in mat_list:
            props = materials[material]
            print(f"   • {material:15s} ({props['type']})")
            for prop, value in props.items():
                if prop != 'type':
                    print(f"     - {prop}: {value}")
    
    print("\n🤖 LLM HYPOTHESIS GENERATION")
    print("-" * 40)
    
    # Simulate LLM hypotheses
    hypotheses = [
        {"material": "KNbO3", "application": "capacitor", "rationale": "Perovskite with high dielectric constant", "confidence": 0.8},
        {"material": "PbTe", "application": "thermoelectric_device", "rationale": "Chalcogenide with good ZT", "confidence": 0.9},
        {"material": "CsSnI3", "application": "solar_cell", "rationale": "Lead-free perovskite with optimal band gap", "confidence": 0.85}
    ]
    
    print("LLM proposes new material-application relationships:")
    for i, hyp in enumerate(hypotheses, 1):
        print(f"   {i}. {hyp['material']} → {hyp['application']}")
        print(f"      Rationale: {hyp['rationale']}")
        print(f"      Confidence: {hyp['confidence']}")
    
    print("\n🔍 VALIDATION PROCESS")
    print("-" * 40)
    
    # Simulate validation
    validation_results = {
        "KNbO3": {"is_valid": True, "confidence": 0.75, "evidence": {"dielectric_constant": 700}},
        "PbTe": {"is_valid": True, "confidence": 0.85, "evidence": {"zt": 0.8}},
        "CsSnI3": {"is_valid": True, "confidence": 0.80, "evidence": {"band_gap": 1.3}}
    }
    
    print("Validating hypotheses against materials databases:")
    for material, result in validation_results.items():
        status = "✅ VALID" if result["is_valid"] else "❌ INVALID"
        print(f"   {material:12s} {status:10s} (confidence: {result['confidence']:.2f})")
        if result["evidence"]:
            evidence_str = ", ".join([f"{k}={v}" for k, v in result["evidence"].items()])
            print(f"                Evidence: {evidence_str}")
    
    print("\n📈 EXPANDED KNOWLEDGE GRAPH")
    print("-" * 40)
    
    # Show expanded graph
    expanded_applications = {
        "capacitor": ["BaTiO3", "SrTiO3", "CaTiO3", "KNbO3"],
        "thermoelectric_device": ["Bi2Te3", "SnSe", "PbTe"],
        "solar_cell": ["CH3NH3PbI3", "Si", "CsSnI3"]
    }
    
    print("After LLM expansion and validation:")
    for app, mat_list in expanded_applications.items():
        print(f"\n🔹 {app.replace('_', ' ').title()}:")
        for material in mat_list:
            if material in ["KNbO3", "PbTe", "CsSnI3"]:
                print(f"   • {material:15s} 🤖 NEW (LLM discovered)")
            else:
                print(f"   • {material:15s} 📚 Original")
    
    print("\n🎯 QUERY RESULTS")
    print("-" * 40)
    
    print("Query: Find all materials used in capacitors")
    print("MATCH (m:Material)-[:USED_IN]->(a:Application {name: 'capacitor'})")
    print("RETURN m.name, r.confidence, r.source")
    print("\nResults:")
    capacitor_results = [
        ("BaTiO3", 1.0, "curated", "📚"),
        ("SrTiO3", 1.0, "curated", "📚"),
        ("CaTiO3", 1.0, "curated", "📚"),
        ("KNbO3", 0.8, "LLM", "🤖")
    ]
    
    for material, confidence, source, icon in capacitor_results:
        print(f"   {icon} {material:12s} (confidence: {confidence:.2f}, source: {source})")
    
    print("\n📊 STATISTICS")
    print("-" * 40)
    print(f"   Original materials: 7")
    print(f"   New materials discovered: 3")
    print(f"   Total relationships: 10")
    print(f"   Expansion rate: 42.9%")
    
    print("\n🎉 DEMO COMPLETE!")
    print("This shows what the full system does:")
    print("1. ✅ LLM analyzes existing materials patterns")
    print("2. ✅ Generates hypotheses for new material-application relationships")
    print("3. ✅ Validates hypotheses against materials databases")
    print("4. ✅ Adds validated relationships to knowledge graph")
    print("5. ✅ Enables complex queries across expanded knowledge")

def main():
    create_demo_graph()
    
    print("\n" + "=" * 60)
    print("🚀 TO SEE THE FULL INTERACTIVE SYSTEM:")
    print("=" * 60)
    print("1. Set up Neo4j database")
    print("2. Run: python src/init_kg.py")
    print("3. Run: streamlit run src/app.py")
    print("4. Open: http://localhost:8501")
    print("\nOr try the working examples:")
    print("• python examples/capacitor_expansion.py")
    print("• python demo.py")

if __name__ == "__main__":
    main()

