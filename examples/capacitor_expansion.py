"""
Example: Capacitor Materials Expansion
Demonstrates the LLM-guided expansion process for capacitor materials
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from knowledge_graph import MaterialsKG, Hypothesis
from llm_integration import OllamaHypothesisGenerator
from validation import MaterialsValidator

def run_capacitor_expansion():
    """Run the capacitor materials expansion example"""
    
    print("🧬 Capacitor Materials Expansion Example")
    print("=" * 50)
    
    # Initialize components
    kg = MaterialsKG()
    llm = OllamaHypothesisGenerator()
    validator = MaterialsValidator()
    
    try:
        # Check connections
        print("🔍 Checking connections...")
        
        if not llm.test_connection():
            print("❌ Ollama not connected. Please start Ollama and pull a model.")
            print("   Example: ollama pull llama3.1")
            return False
        
        print("✅ Ollama connected")
        
        # Get current capacitor materials
        print("\n📊 Current capacitor materials:")
        current_materials = kg.get_materials_for_application("capacitor")
        
        if not current_materials:
            print("❌ No current materials found. Please initialize the knowledge graph first.")
            print("   Run: python src/init_kg.py")
            return False
        
        for material in current_materials:
            print(f"   • {material['material']} (confidence: {material['confidence']})")
        
        # Get context for LLM
        print("\n🧠 Gathering context for LLM...")
        context = {}
        material_names = [m['material'] for m in current_materials]
        
        for material in material_names:
            neighbors = kg.get_neighbors(material)
            for entity_type, names in neighbors.items():
                if entity_type not in context:
                    context[entity_type] = []
                context[entity_type].extend(names)
        
        print(f"   Context includes {len(context.get('Property', []))} properties")
        
        # Generate hypotheses
        print("\n🤖 Generating hypotheses with LLM...")
        response = llm.generate_capacitor_hypotheses(material_names, context)
        
        if not response.hypotheses:
            print("❌ No hypotheses generated")
            return False
        
        print(f"✅ Generated {len(response.hypotheses)} hypotheses")
        print(f"   Overall confidence: {response.confidence:.2f}")
        
        # Display LLM reasoning
        print(f"\n🧠 LLM Reasoning:")
        print(f"   {response.reasoning}")
        
        # Validate and process hypotheses
        print(f"\n🔍 Validating hypotheses...")
        validated_count = 0
        
        for i, hypothesis_data in enumerate(response.hypotheses, 1):
            material = hypothesis_data.get('material', '')
            rationale = hypothesis_data.get('rationale', '')
            confidence = hypothesis_data.get('confidence', 0.5)
            
            if not material:
                continue
            
            print(f"\n   {i}. {material} → capacitor")
            print(f"      Rationale: {rationale}")
            print(f"      LLM Confidence: {confidence:.2f}")
            
            # Validate hypothesis
            validation_result = validator.validate_capacitor_material(material)
            
            print(f"      Validation: {'✅ Valid' if validation_result.is_valid else '❌ Invalid'}")
            print(f"      Evidence: {validation_result.evidence}")
            print(f"      Final Confidence: {validation_result.confidence:.2f}")
            
            # Add to knowledge graph if valid
            if validation_result.is_valid and validation_result.confidence >= 0.6:
                hypothesis = Hypothesis(
                    material=material,
                    application="capacitor",
                    relationship="USED_IN",
                    confidence=validation_result.confidence * confidence,
                    source="LLM",
                    validated_by=validation_result.source,
                    rationale=rationale
                )
                
                kg.add_hypothesis(hypothesis)
                print(f"      ✅ Added to knowledge graph!")
                validated_count += 1
            else:
                print(f"      ⏭️  Not added (low confidence)")
        
        # Summary
        print(f"\n📈 Summary:")
        print(f"   • Hypotheses generated: {len(response.hypotheses)}")
        print(f"   • Hypotheses validated: {validated_count}")
        print(f"   • Success rate: {validated_count/len(response.hypotheses)*100:.1f}%")
        
        # Show updated materials list
        print(f"\n📋 Updated capacitor materials:")
        updated_materials = kg.get_materials_for_application("capacitor")
        for material in updated_materials:
            source = "🆕 LLM" if material['source'] == 'LLM' else "📚 Curated"
            print(f"   • {material['material']} {source} (confidence: {material['confidence']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during expansion: {e}")
        return False
    
    finally:
        kg.close()

def main():
    """Main function"""
    success = run_capacitor_expansion()
    
    if success:
        print(f"\n🎉 Capacitor expansion completed successfully!")
        print(f"   You can now query the expanded knowledge graph.")
    else:
        print(f"\n❌ Capacitor expansion failed.")
        print(f"   Please check your setup and try again.")

if __name__ == "__main__":
    main()

