"""
Test script to verify the Materials Ontology Expansion project setup
Tests components that don't require Neo4j or Ollama
"""

import sys
import os
sys.path.append('src')

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from knowledge_graph import MaterialsKG, Hypothesis
        print("✅ Knowledge Graph module imported successfully")
    except Exception as e:
        print(f"❌ Knowledge Graph import failed: {e}")
        return False
    
    try:
        from llm_integration import OllamaHypothesisGenerator
        print("✅ LLM Integration module imported successfully")
    except Exception as e:
        print(f"❌ LLM Integration import failed: {e}")
        return False
    
    try:
        from validation import MaterialsValidator
        print("✅ Validation module imported successfully")
    except Exception as e:
        print(f"❌ Validation import failed: {e}")
        return False
    
    return True

def test_validation_module():
    """Test the validation module without external dependencies"""
    print("\n🧪 Testing validation module...")
    
    try:
        from validation import MaterialsValidator
        
        validator = MaterialsValidator()
        
        # Test material properties
        props = validator.get_material_properties("BaTiO3")
        if props:
            print(f"✅ BaTiO3 properties: {props}")
        else:
            print("❌ No properties found for BaTiO3")
            return False
        
        # Test validation
        result = validator.validate_capacitor_material("SrTiO3")
        print(f"✅ SrTiO3 capacitor validation: {result.is_valid}, confidence: {result.confidence}")
        
        # Test recommendations
        recommendations = validator.get_recommended_materials("capacitor", limit=3)
        print(f"✅ Capacitor recommendations: {len(recommendations)} materials")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation module test failed: {e}")
        return False

def test_llm_integration():
    """Test LLM integration without actually calling Ollama"""
    print("\n🧪 Testing LLM integration module...")
    
    try:
        from llm_integration import OllamaHypothesisGenerator
        
        llm = OllamaHypothesisGenerator()
        
        # Test connection (will fail if Ollama not running, but that's ok)
        connection_ok = llm.test_connection()
        if connection_ok:
            print("✅ Ollama connection successful")
            models = llm.get_available_models()
            print(f"✅ Available models: {models}")
        else:
            print("⚠️  Ollama not connected (this is expected if not set up)")
        
        # Test response parsing
        test_response = '''
        {
          "hypotheses": [
            {
              "material": "CaTiO3",
              "application": "capacitor",
              "relationship": "USED_IN",
              "rationale": "Similar perovskite structure to BaTiO3",
              "confidence": 0.8
            }
          ],
          "reasoning": "Perovskite oxides with high dielectric constants are good for capacitors",
          "confidence": 0.7
        }
        '''
        
        parsed = llm._parse_response(test_response)
        if parsed and parsed.hypotheses:
            print(f"✅ Response parsing successful: {len(parsed.hypotheses)} hypotheses")
        else:
            print("❌ Response parsing failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ LLM integration test failed: {e}")
        return False

def test_data_files():
    """Test that data files exist and are valid"""
    print("\n🧪 Testing data files...")
    
    try:
        import json
        
        # Check seed data
        if os.path.exists('data/seed_data.json'):
            with open('data/seed_data.json', 'r') as f:
                seed_data = json.load(f)
            
            required_keys = ['materials', 'properties', 'applications', 'has_property_relations', 'used_in_relations']
            for key in required_keys:
                if key not in seed_data:
                    print(f"❌ Missing key in seed data: {key}")
                    return False
            
            print(f"✅ Seed data valid: {len(seed_data['materials'])} materials, {len(seed_data['properties'])} properties")
        else:
            print("❌ Seed data file not found")
            return False
        
        # Check environment file
        if os.path.exists('.env'):
            print("✅ Environment file exists")
        else:
            print("⚠️  Environment file not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Data files test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧬 Materials Ontology Expansion - Project Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_validation_module,
        test_llm_integration,
        test_data_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project setup is working correctly.")
        print("\nNext steps:")
        print("1. Set up Neo4j database")
        print("2. Set up Ollama with a model (e.g., llama3.1)")
        print("3. Update .env file with your credentials")
        print("4. Run: python src/init_kg.py")
        print("5. Run: streamlit run src/app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

