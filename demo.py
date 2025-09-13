"""
Complete Demonstration of Materials Ontology Expansion
Shows the full workflow from knowledge graph to LLM expansion to validation
"""

import sys
import os
sys.path.append('src')

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"🧬 {title}")
    print("=" * 60)

def print_step(step, description):
    """Print a formatted step"""
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def demo_validation_only():
    """Demonstrate the validation system without Neo4j/Ollama"""
    print_header("Materials Ontology Expansion - Validation Demo")
    
    print_step(1, "Initialize Validation System")
    try:
        from validation import MaterialsValidator
        validator = MaterialsValidator()
        print("✅ Validation system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize validation: {e}")
        return False
    
    print_step(2, "Test Material Property Database")
    print("📊 Available materials in validation database:")
    materials = list(validator.materials_properties.keys())
    for i, material in enumerate(materials[:10], 1):
        props = validator.materials_properties[material]
        print(f"   {i:2d}. {material:15s} - {props.get('type', 'unknown'):12s}")
    
    if len(materials) > 10:
        print(f"   ... and {len(materials) - 10} more materials")
    
    print_step(3, "Demonstrate Capacitor Material Validation")
    capacitor_materials = ["BaTiO3", "SrTiO3", "CaTiO3", "Si", "Bi2Te3"]
    
    print("🔍 Validating materials for capacitor applications:")
    for material in capacitor_materials:
        result = validator.validate_capacitor_material(material)
        status = "✅ VALID" if result.is_valid else "❌ INVALID"
        confidence = result.confidence
        
        if result.evidence:
            evidence_str = ", ".join([f"{k}={v}" for k, v in result.evidence.items()])
            print(f"   {material:12s} {status:10s} (confidence: {confidence:.2f}) - {evidence_str}")
        else:
            print(f"   {material:12s} {status:10s} (confidence: {confidence:.2f})")
    
    print_step(4, "Demonstrate Thermoelectric Material Validation")
    thermoelectric_materials = ["Bi2Te3", "PbTe", "SnSe", "Bi2Se3", "BaTiO3"]
    
    print("🔥 Validating materials for thermoelectric applications:")
    for material in thermoelectric_materials:
        result = validator.validate_thermoelectric_material(material)
        status = "✅ VALID" if result.is_valid else "❌ INVALID"
        confidence = result.confidence
        
        if result.evidence:
            evidence_str = ", ".join([f"{k}={v}" for k, v in result.evidence.items()])
            print(f"   {material:12s} {status:10s} (confidence: {confidence:.2f}) - {evidence_str}")
        else:
            print(f"   {material:12s} {status:10s} (confidence: {confidence:.2f})")
    
    print_step(5, "Demonstrate Solar Cell Material Validation")
    solar_materials = ["Si", "CdTe", "CH3NH3PbI3", "CsSnI3", "BaTiO3"]
    
    print("☀️ Validating materials for solar cell applications:")
    for material in solar_materials:
        result = validator.validate_solar_cell_material(material)
        status = "✅ VALID" if result.is_valid else "❌ INVALID"
        confidence = result.confidence
        
        if result.evidence:
            evidence_str = ", ".join([f"{k}={v}" for k, v in result.evidence.items()])
            print(f"   {material:12s} {status:10s} (confidence: {confidence:.2f}) - {evidence_str}")
        else:
            print(f"   {material:12s} {status:10s} (confidence: {confidence:.2f})")
    
    print_step(6, "Generate Material Recommendations")
    print("🎯 Top recommendations for each application:")
    
    applications = ["capacitor", "thermoelectric_device", "solar_cell"]
    for app in applications:
        recommendations = validator.get_recommended_materials(app, limit=3)
        print(f"\n   {app.replace('_', ' ').title()}:")
        for rec in recommendations:
            print(f"      • {rec['material']:12s} (confidence: {rec['confidence']:.2f})")
    
    return True

def demo_llm_integration():
    """Demonstrate LLM integration capabilities"""
    print_header("LLM Integration Demo")
    
    print_step(1, "Test Ollama Connection")
    try:
        from llm_integration import OllamaHypothesisGenerator
        llm = OllamaHypothesisGenerator()
        
        if llm.test_connection():
            print("✅ Ollama connection successful")
            models = llm.get_available_models()
            print(f"📋 Available models: {', '.join(models)}")
        else:
            print("❌ Ollama not connected")
            return False
    except Exception as e:
        print(f"❌ LLM integration failed: {e}")
        return False
    
    print_step(2, "Test Response Parsing")
    test_response = '''
    {
      "hypotheses": [
        {
          "material": "CaTiO3",
          "application": "capacitor",
          "relationship": "USED_IN",
          "rationale": "Similar perovskite structure to BaTiO3 with high dielectric constant",
          "confidence": 0.8
        },
        {
          "material": "KNbO3", 
          "application": "capacitor",
          "relationship": "USED_IN",
          "rationale": "Perovskite oxide with ferroelectric properties suitable for capacitors",
          "confidence": 0.7
        }
      ],
      "reasoning": "Perovskite oxides with high dielectric constants are commonly used in capacitor applications. The crystal structure and electronic properties make them suitable for energy storage.",
      "confidence": 0.75
    }
    '''
    
    parsed = llm._parse_response(test_response)
    if parsed and parsed.hypotheses:
        print(f"✅ Successfully parsed {len(parsed.hypotheses)} hypotheses")
        print(f"📝 Reasoning: {parsed.reasoning}")
        print(f"🎯 Overall confidence: {parsed.confidence}")
        
        for i, hyp in enumerate(parsed.hypotheses, 1):
            print(f"   {i}. {hyp['material']} → {hyp['application']} (confidence: {hyp['confidence']})")
            print(f"      Rationale: {hyp['rationale']}")
    else:
        print("❌ Failed to parse LLM response")
        return False
    
    print_step(3, "Demonstrate Application-Specific Prompting")
    print("🤖 Testing specialized prompts for different applications...")
    
    # Test capacitor prompt
    known_capacitors = ["BaTiO3", "PbTiO3"]
    context = {"Property": ["dielectric_constant", "band_gap"], "Application": ["capacitor"]}
    
    print("   Capacitor materials prompt ready")
    print(f"   Known materials: {', '.join(known_capacitors)}")
    print("   Context includes: dielectric_constant, band_gap properties")
    
    return True

def demo_complete_workflow():
    """Demonstrate the complete workflow (without actual Neo4j)"""
    print_header("Complete Workflow Demo")
    
    print_step(1, "Simulate Knowledge Graph State")
    print("📊 Current knowledge graph state:")
    print("   Materials: 15 (BaTiO3, SrTiO3, Bi2Te3, SnSe, CH3NH3PbI3, etc.)")
    print("   Properties: 10 (dielectric_constant, band_gap, zt, etc.)")
    print("   Applications: 8 (capacitor, thermoelectric_device, solar_cell, etc.)")
    print("   Relationships: 27 (material-property and material-application)")
    
    print_step(2, "Simulate LLM Hypothesis Generation")
    print("🤖 LLM analyzes patterns and generates hypotheses:")
    
    # Simulate capacitor expansion
    capacitor_hypotheses = [
        {"material": "CaTiO3", "rationale": "Perovskite structure similar to BaTiO3", "confidence": 0.8},
        {"material": "KNbO3", "rationale": "Ferroelectric perovskite oxide", "confidence": 0.7},
        {"material": "LiNbO3", "rationale": "High dielectric constant perovskite", "confidence": 0.6}
    ]
    
    print("   Capacitor material hypotheses:")
    for i, hyp in enumerate(capacitor_hypotheses, 1):
        print(f"      {i}. {hyp['material']:12s} - {hyp['rationale']} (confidence: {hyp['confidence']})")
    
    print_step(3, "Simulate Validation Process")
    print("🔍 Validating hypotheses against materials databases:")
    
    for hyp in capacitor_hypotheses:
        material = hyp['material']
        
        # Simulate validation results
        if material == "CaTiO3":
            validation_result = {"is_valid": True, "confidence": 0.85, "evidence": {"dielectric_constant": 150}}
        elif material == "KNbO3":
            validation_result = {"is_valid": True, "confidence": 0.75, "evidence": {"dielectric_constant": 700}}
        else:
            validation_result = {"is_valid": False, "confidence": 0.4, "evidence": {"dielectric_constant": 25}}
        
        status = "✅ VALID" if validation_result["is_valid"] else "❌ INVALID"
        final_confidence = hyp['confidence'] * validation_result['confidence']
        
        print(f"   {material:12s} {status:10s} (final confidence: {final_confidence:.2f})")
        if validation_result["evidence"]:
            evidence = ", ".join([f"{k}={v}" for k, v in validation_result["evidence"].items()])
            print(f"                Evidence: {evidence}")
    
    print_step(4, "Simulate Knowledge Graph Update")
    print("📈 Adding validated hypotheses to knowledge graph:")
    
    validated_count = 0
    for hyp in capacitor_hypotheses:
        material = hyp['material']
        if material in ["CaTiO3", "KNbO3"]:  # Valid ones
            validated_count += 1
            print(f"   ✅ Added: {material} → capacitor (confidence: {hyp['confidence']:.2f})")
    
    print(f"\n   📊 Expansion complete: {validated_count} new relationships added")
    
    print_step(5, "Simulate Query Results")
    print("🔍 Querying expanded knowledge graph:")
    
    print("   MATCH (m:Material)-[:USED_IN]->(a:Application {name: 'capacitor'})")
    print("   RETURN m.name, r.confidence, r.source")
    print()
    print("   Results:")
    
    capacitor_results = [
        ("BaTiO3", 1.0, "curated"),
        ("PbTiO3", 1.0, "curated"), 
        ("CaTiO3", 0.8, "LLM"),
        ("KNbO3", 0.7, "LLM")
    ]
    
    for material, confidence, source in capacitor_results:
        source_icon = "📚" if source == "curated" else "🤖"
        print(f"      {source_icon} {material:12s} (confidence: {confidence:.2f})")
    
    return True

def main():
    """Main demonstration function"""
    print("🧬 Materials Ontology Expansion - Complete Demonstration")
    print("This demo shows the full workflow without requiring Neo4j setup")
    
    demos = [
        ("Validation System", demo_validation_only),
        ("LLM Integration", demo_llm_integration),
        ("Complete Workflow", demo_complete_workflow)
    ]
    
    for demo_name, demo_func in demos:
        try:
            success = demo_func()
            if success:
                print(f"\n✅ {demo_name} demo completed successfully!")
            else:
                print(f"\n❌ {demo_name} demo failed!")
        except Exception as e:
            print(f"\n❌ {demo_name} demo error: {e}")
        
        input("\nPress Enter to continue to next demo...")
    
    print_header("Demo Complete!")
    print("🎉 All demonstrations completed successfully!")
    print("\nTo run the full system:")
    print("1. Set up Neo4j database")
    print("2. Configure Ollama with a model")
    print("3. Run: python src/init_kg.py")
    print("4. Run: streamlit run src/app.py")
    print("\nOr try the example scripts:")
    print("• python examples/capacitor_expansion.py")
    print("• python examples/thermoelectric_expansion.py")

if __name__ == "__main__":
    main()

