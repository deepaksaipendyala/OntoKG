"""
LLM Integration for Materials Ontology Expansion
Uses Ollama for hypothesis generation with structured prompts
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMResponse:
    """Structured response from LLM"""
    hypotheses: List[Dict[str, Any]]
    reasoning: str
    confidence: float

class OllamaHypothesisGenerator:
    """Generate hypotheses using Ollama LLM"""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.model = model or os.getenv('OLLAMA_MODEL', 'llama3.2:latest')
        
    def _make_request(self, prompt: str, system_prompt: str = None) -> str:
        """Make a request to Ollama API"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return ""
    
    def generate_capacitor_hypotheses(self, known_materials: List[str], 
                                    context: Dict[str, List[str]]) -> LLMResponse:
        """Generate hypotheses for capacitor materials"""
        
        system_prompt = """You are a materials science expert. Analyze patterns in materials used for specific applications and suggest new materials based on scientific reasoning. Return your response in JSON format with the following structure:
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
}"""
        
        prompt = f"""
Given the following known capacitor materials and their context:

Known Capacitor Materials: {', '.join(known_materials)}

Context:
- Properties: {context.get('Property', [])}
- Other Applications: {context.get('Application', [])}

Based on materials science principles, suggest 3-5 additional materials that could be used in capacitors. Consider:
1. Similar crystal structures (perovskites, spinels, etc.)
2. High dielectric constant materials
3. Ferroelectric properties
4. Chemical similarity to known capacitor materials

Focus on materials that are likely to have high dielectric constants or ferroelectric properties suitable for capacitor applications.

Return your analysis in the JSON format specified above.
"""
        
        response_text = self._make_request(prompt, system_prompt)
        return self._parse_response(response_text)
    
    def generate_thermoelectric_hypotheses(self, known_materials: List[str],
                                         context: Dict[str, List[str]]) -> LLMResponse:
        """Generate hypotheses for thermoelectric materials"""
        
        system_prompt = """You are a materials science expert specializing in thermoelectric materials. Analyze patterns and suggest new materials based on scientific reasoning. Return your response in JSON format with the following structure:
{
  "hypotheses": [
    {
      "material": "Material Name",
      "application": "thermoelectric_device", 
      "relationship": "USED_IN",
      "rationale": "Scientific reasoning",
      "confidence": 0.8
    }
  ],
  "reasoning": "Overall analysis",
  "confidence": 0.7
}"""
        
        prompt = f"""
Given the following known thermoelectric materials and their context:

Known Thermoelectric Materials: {', '.join(known_materials)}

Context:
- Properties: {context.get('Property', [])}
- Other Applications: {context.get('Application', [])}

Based on thermoelectric materials science, suggest 3-5 additional materials that could be used in thermoelectric devices. Consider:
1. Materials with low thermal conductivity and high electrical conductivity
2. Chalcogenides (tellurides, selenides, sulfides)
3. Skutterudites and clathrates
4. Zintl phases
5. Materials with complex crystal structures that scatter phonons

Focus on materials that are likely to have good thermoelectric figure of merit (ZT) values.

Return your analysis in the JSON format specified above.
"""
        
        response_text = self._make_request(prompt, system_prompt)
        return self._parse_response(response_text)
    
    def generate_solar_cell_hypotheses(self, known_materials: List[str],
                                     context: Dict[str, List[str]]) -> LLMResponse:
        """Generate hypotheses for solar cell materials"""
        
        system_prompt = """You are a materials science expert specializing in photovoltaic materials. Analyze patterns and suggest new materials based on scientific reasoning. Return your response in JSON format with the following structure:
{
  "hypotheses": [
    {
      "material": "Material Name",
      "application": "solar_cell", 
      "relationship": "USED_IN",
      "rationale": "Scientific reasoning",
      "confidence": 0.8
    }
  ],
  "reasoning": "Overall analysis",
  "confidence": 0.7
}"""
        
        prompt = f"""
Given the following known solar cell materials and their context:

Known Solar Cell Materials: {', '.join(known_materials)}

Context:
- Properties: {context.get('Property', [])}
- Other Applications: {context.get('Application', [])}

Based on photovoltaic materials science, suggest 3-5 additional materials that could be used in solar cells. Consider:
1. Materials with appropriate band gaps (1.0-1.8 eV for optimal efficiency)
2. Direct band gap semiconductors
3. Good absorption coefficients
4. Suitable carrier mobilities
5. Stability under solar irradiation

Focus on materials that are likely to have band gaps in the optimal range for solar absorption and good photovoltaic properties.

Return your analysis in the JSON format specified above.
"""
        
        response_text = self._make_request(prompt, system_prompt)
        return self._parse_response(response_text)
    
    def generate_hypotheses(self, application: str, context: Dict[str, List[str]], 
                          known_materials: List[str] = None) -> LLMResponse:
        """Generate hypotheses for a specific application"""
        
        if known_materials is None:
            known_materials = context.get('Material', [])
        
        if application.lower() in ['capacitor', 'capacitors']:
            return self.generate_capacitor_hypotheses(known_materials, context)
        elif application.lower() in ['thermoelectric', 'thermoelectric_device', 'thermoelectrics']:
            return self.generate_thermoelectric_hypotheses(known_materials, context)
        elif application.lower() in ['solar_cell', 'solar_cells', 'photovoltaic', 'photovoltaics']:
            return self.generate_solar_cell_hypotheses(known_materials, context)
        else:
            return self.generate_generic_hypotheses(application, known_materials, context)
    
    def generate_generic_hypotheses(self, application: str, known_materials: List[str],
                                  context: Dict[str, List[str]]) -> LLMResponse:
        """Generate hypotheses for any application"""
        
        system_prompt = """You are a materials science expert. Analyze patterns in materials used for specific applications and suggest new materials based on scientific reasoning. Return your response in JSON format with the following structure:
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
}"""
        
        prompt = f"""
Given the following known materials for {application} and their context:

Known Materials: {', '.join(known_materials)}

Context:
- Properties: {context.get('Property', [])}
- Other Applications: {context.get('Application', [])}

Based on materials science principles, suggest 3-5 additional materials that could be used in {application}. Consider:
1. Chemical and structural similarity to known materials
2. Relevant physical properties
3. Processing considerations
4. Cost and availability factors

Focus on materials that are likely to have properties suitable for {application}.

Return your analysis in the JSON format specified above.
"""
        
        response_text = self._make_request(prompt, system_prompt)
        return self._parse_response(response_text)
    
    def _parse_response(self, response_text: str) -> LLMResponse:
        """Parse LLM response into structured format"""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                data = json.loads(json_str)
                
                return LLMResponse(
                    hypotheses=data.get('hypotheses', []),
                    reasoning=data.get('reasoning', ''),
                    confidence=data.get('confidence', 0.5)
                )
            else:
                # Fallback: try to extract structured information
                return self._fallback_parse(response_text)
                
        except json.JSONDecodeError:
            return self._fallback_parse(response_text)
    
    def _fallback_parse(self, response_text: str) -> LLMResponse:
        """Fallback parsing when JSON parsing fails"""
        # Simple fallback - extract material names and create basic hypotheses
        lines = response_text.split('\n')
        hypotheses = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['material', 'compound', 'alloy']):
                # Try to extract material name
                words = line.split()
                for word in words:
                    if any(char.isupper() for char in word) and len(word) > 2:
                        hypotheses.append({
                            "material": word,
                            "application": "unknown",
                            "relationship": "USED_IN",
                            "rationale": "Extracted from LLM response",
                            "confidence": 0.6
                        })
                        break
        
        return LLMResponse(
            hypotheses=hypotheses[:5],  # Limit to 5 hypotheses
            reasoning="Parsed from unstructured response",
            confidence=0.5
        )
    
    def test_connection(self) -> bool:
        """Test connection to Ollama"""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except:
            return []

