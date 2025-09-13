"""
Advanced LLM Integration for Materials Ontology Expansion
Features multi-model ensemble, chain-of-thought reasoning, and scientific validation
"""

import os
import json
import requests
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from datetime import datetime

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class AdvancedLLMResponse:
    """Enhanced structured response from LLM ensemble"""
    hypotheses: List[Dict[str, Any]]
    reasoning: str
    confidence: float
    ensemble_scores: Dict[str, float] = field(default_factory=dict)
    scientific_validation: Dict[str, Any] = field(default_factory=dict)
    chain_of_thought: List[str] = field(default_factory=list)
    uncertainty_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class ModelConfig:
    """Configuration for individual models"""
    name: str
    weight: float = 1.0
    temperature: float = 0.7
    max_tokens: int = 2048
    specialization: str = "general"  # general, chemistry, physics, materials
    
class AdvancedHypothesisGenerator:
    """Advanced hypothesis generation using multiple LLM models with ensemble voting"""
    
    def __init__(self, base_url: str = None, models: List[str] = None):
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Get available models from Ollama if not specified
        if models is None:
            try:
                import ollama
                available_models = [model.model for model in ollama.list().models]
                # Use available models, prioritizing scientific and general models
                preferred_models = ['sciphi/triplex:latest', 'llama3.2:latest', 'gpt-oss:20b', 'duckdb-nsql:7b-q4_K_M']
                self.models = [model for model in preferred_models if model in available_models]
                
                # If no preferred models, use first 3 available
                if not self.models and available_models:
                    self.models = available_models[:3]
                elif not self.models:
                    self.models = ['llama3.2:latest']  # Fallback
                    
            except Exception as e:
                logger.error(f"Error getting available models: {e}")
                self.models = ['llama3.2:latest', 'sciphi/triplex:latest']
        else:
            self.models = models
        
        # Model configurations with specializations
        self.model_configs = {
            'llama3.2:latest': ModelConfig('llama3.2:latest', 1.2, 0.6, 2048, 'general'),
            'llama3:latest': ModelConfig('llama3:latest', 1.1, 0.7, 2048, 'general'),
            'mistral:latest': ModelConfig('mistral:latest', 1.0, 0.8, 2048, 'reasoning'),
            'sciphi/triplex:latest': ModelConfig('sciphi/triplex:latest', 1.3, 0.5, 2048, 'scientific'),
            'gpt-oss:20b': ModelConfig('gpt-oss:20b', 1.4, 0.6, 2048, 'general'),
            'duckdb-nsql:7b-q4_K_M': ModelConfig('duckdb-nsql:7b-q4_K_M', 1.1, 0.7, 2048, 'reasoning'),
            'llama3.2-vision:latest': ModelConfig('llama3.2-vision:latest', 1.0, 0.7, 2048, 'multimodal')
        }
        
        # Ensure we have configs for all our models
        for model in self.models:
            if model not in self.model_configs:
                self.model_configs[model] = ModelConfig(model, 1.0, 0.7, 2048, 'general')
        
        # Scientific validation prompts
        self.validation_prompts = {
            'chemistry': self._get_chemistry_validation_prompt(),
            'physics': self._get_physics_validation_prompt(),
            'materials': self._get_materials_validation_prompt()
        }
        
    def _get_chemistry_validation_prompt(self) -> str:
        return """
        As a chemistry expert, evaluate the chemical feasibility of the proposed material-application relationship.
        Consider: chemical stability, reactivity, synthesis feasibility, thermodynamics.
        Rate confidence 0-1 and provide reasoning.
        """
        
    def _get_physics_validation_prompt(self) -> str:
        return """
        As a physics expert, evaluate the physical properties alignment for the proposed application.
        Consider: electronic structure, optical properties, mechanical properties, thermal properties.
        Rate confidence 0-1 and provide reasoning.
        """
        
    def _get_materials_validation_prompt(self) -> str:
        return """
        As a materials science expert, evaluate the practical viability of this material for the application.
        Consider: processing requirements, scalability, cost, performance trade-offs, existing alternatives.
        Rate confidence 0-1 and provide reasoning.
        """

    async def _make_async_request(self, session: aiohttp.ClientSession, 
                                model: str, prompt: str, system_prompt: str = None) -> Tuple[str, str]:
        """Make asynchronous request to Ollama API"""
        url = f"{self.base_url}/api/generate"
        
        config = self.model_configs.get(model, ModelConfig(model))
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_p": 0.9,
                "max_tokens": config.max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            async with session.post(url, json=payload, timeout=120) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", ""), model
                else:
                    logger.error(f"Error from {model}: {response.status}")
                    return "", model
        except Exception as e:
            logger.error(f"Error calling {model}: {e}")
            return "", model

    async def generate_ensemble_hypotheses(self, application: str, 
                                         known_materials: List[str],
                                         context: Dict[str, Any]) -> AdvancedLLMResponse:
        """Generate hypotheses using ensemble of models with chain-of-thought reasoning"""
        
        # Create enhanced prompt with chain-of-thought
        base_prompt = self._create_chain_of_thought_prompt(application, known_materials, context)
        system_prompt = self._create_scientific_system_prompt()
        
        # Run ensemble inference
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._make_async_request(session, model, base_prompt, system_prompt)
                for model in self.models
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process ensemble responses
        valid_responses = [(resp, model) for resp, model in responses 
                          if isinstance(resp, tuple) and resp[0]]
        
        if not valid_responses:
            return self._create_fallback_response(application)
        
        # Parse and combine responses
        parsed_responses = []
        for response_text, model in valid_responses:
            try:
                parsed = self._parse_enhanced_response(response_text, model)
                parsed_responses.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse response from {model}: {e}")
        
        # Ensemble voting and combination
        final_response = self._combine_ensemble_responses(parsed_responses, application)
        
        # Add scientific validation
        await self._add_scientific_validation(final_response, application)
        
        return final_response

    def _create_chain_of_thought_prompt(self, application: str, 
                                      known_materials: List[str],
                                      context: Dict[str, Any]) -> str:
        """Create enhanced prompt with chain-of-thought reasoning"""
        
        context_str = self._format_context(context)
        
        return f"""
        You are a world-class materials scientist with expertise in {application} applications.
        
        TASK: Suggest new materials for {application} applications using systematic reasoning.
        
        KNOWN MATERIALS: {', '.join(known_materials)}
        
        CONTEXT: {context_str}
        
        REASONING PROCESS:
        1. PATTERN ANALYSIS: What patterns do you see in the known materials?
        2. PROPERTY REQUIREMENTS: What key properties are needed for {application}?
        3. MATERIAL FAMILIES: What material families or structures might work?
        4. SPECIFIC CANDIDATES: Based on 1-3, suggest specific materials
        5. CONFIDENCE ASSESSMENT: Rate each suggestion's likelihood of success
        
        RESPONSE FORMAT (JSON):
        {{
            "chain_of_thought": [
                "Pattern analysis: [your analysis]",
                "Property requirements: [requirements]", 
                "Material families: [families]",
                "Specific reasoning for each candidate"
            ],
            "hypotheses": [
                {{
                    "material": "MaterialName",
                    "formula": "Chemical formula",
                    "application": "{application}",
                    "relationship": "USED_IN",
                    "rationale": "Detailed scientific reasoning",
                    "confidence": 0.85,
                    "key_properties": ["property1", "property2"],
                    "material_class": "perovskite/chalcogenide/etc",
                    "synthesis_feasibility": "high/medium/low",
                    "novelty": "known/emerging/novel"
                }}
            ],
            "overall_confidence": 0.8,
            "reasoning_quality": "high/medium/low"
        }}
        
        Be specific, scientific, and justify all suggestions with materials science principles.
        """

    def _create_scientific_system_prompt(self) -> str:
        """Create system prompt emphasizing scientific rigor"""
        return """
        You are a leading materials scientist with deep knowledge of:
        - Crystal structures and their property relationships
        - Electronic band theory and optical properties  
        - Thermodynamics and phase stability
        - Materials processing and synthesis
        - Application-specific requirements
        
        Always provide scientifically grounded reasoning based on:
        - Structure-property relationships
        - Thermodynamic stability
        - Electronic/optical/mechanical properties
        - Known synthesis routes
        - Literature precedents
        
        Be precise with chemical formulas and property values.
        Acknowledge uncertainty when appropriate.
        """

    def _parse_enhanced_response(self, response_text: str, model: str) -> Dict[str, Any]:
        """Parse enhanced response with error handling"""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return self._fallback_parse_enhanced(response_text, model)
            
            json_str = response_text[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            # Validate structure
            required_keys = ['hypotheses', 'chain_of_thought']
            if not all(key in parsed for key in required_keys):
                return self._fallback_parse_enhanced(response_text, model)
            
            # Add model metadata
            parsed['source_model'] = model
            parsed['model_config'] = self.model_configs.get(model, ModelConfig(model))
            
            return parsed
            
        except Exception as e:
            logger.warning(f"JSON parsing failed for {model}: {e}")
            return self._fallback_parse_enhanced(response_text, model)

    def _fallback_parse_enhanced(self, response_text: str, model: str) -> Dict[str, Any]:
        """Enhanced fallback parsing"""
        # Extract material names using improved heuristics
        hypotheses = []
        lines = response_text.split('\n')
        
        material_indicators = ['material', 'compound', 'alloy', 'crystal', 'phase']
        formula_pattern = r'[A-Z][a-z]?[0-9]*(?:[A-Z][a-z]?[0-9]*)*'
        
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in material_indicators):
                # Try to extract material name and formula
                words = line.split()
                for i, word in enumerate(words):
                    if any(char.isupper() for char in word) and len(word) > 2:
                        hypothesis = {
                            "material": word.strip('.,()'),
                            "formula": word.strip('.,()'),  # Simplified
                            "application": "unknown",
                            "relationship": "USED_IN",
                            "rationale": f"Extracted from {model} response",
                            "confidence": 0.5,
                            "key_properties": [],
                            "material_class": "unknown",
                            "synthesis_feasibility": "unknown",
                            "novelty": "unknown"
                        }
                        hypotheses.append(hypothesis)
                        break
        
        return {
            'hypotheses': hypotheses[:3],  # Limit to 3
            'chain_of_thought': [f"Fallback parsing from {model}"],
            'overall_confidence': 0.4,
            'reasoning_quality': 'low',
            'source_model': model,
            'model_config': self.model_configs.get(model, ModelConfig(model))
        }

    def _combine_ensemble_responses(self, responses: List[Dict[str, Any]], 
                                  application: str) -> AdvancedLLMResponse:
        """Combine responses from multiple models using weighted voting"""
        
        if not responses:
            return self._create_fallback_response(application)
        
        # Collect all hypotheses with model weights
        all_hypotheses = []
        ensemble_scores = {}
        combined_chain_of_thought = []
        
        for response in responses:
            model = response.get('source_model', 'unknown')
            config = response.get('model_config', ModelConfig(model))
            weight = config.weight
            
            ensemble_scores[model] = {
                'weight': weight,
                'confidence': response.get('overall_confidence', 0.5),
                'reasoning_quality': response.get('reasoning_quality', 'medium')
            }
            
            # Weight hypotheses by model confidence
            for hyp in response.get('hypotheses', []):
                hyp['model_source'] = model
                hyp['model_weight'] = weight
                hyp['weighted_confidence'] = hyp.get('confidence', 0.5) * weight
                all_hypotheses.append(hyp)
            
            # Combine chain of thought
            cot = response.get('chain_of_thought', [])
            combined_chain_of_thought.extend([f"[{model}] {step}" for step in cot])
        
        # Group similar materials and combine scores
        material_groups = self._group_similar_materials(all_hypotheses)
        final_hypotheses = self._select_best_hypotheses(material_groups)
        
        # Calculate ensemble confidence
        weighted_confidences = [h['weighted_confidence'] for h in all_hypotheses if 'weighted_confidence' in h]
        ensemble_confidence = np.mean(weighted_confidences) if weighted_confidences else 0.5
        
        # Calculate uncertainty metrics
        uncertainty_metrics = self._calculate_uncertainty_metrics(all_hypotheses, responses)
        
        return AdvancedLLMResponse(
            hypotheses=final_hypotheses,
            reasoning=f"Ensemble of {len(responses)} models with weighted voting",
            confidence=float(ensemble_confidence),
            ensemble_scores=ensemble_scores,
            chain_of_thought=combined_chain_of_thought,
            uncertainty_metrics=uncertainty_metrics
        )

    def _group_similar_materials(self, hypotheses: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group similar materials for ensemble voting"""
        groups = {}
        
        for hyp in hypotheses:
            material = hyp.get('material', '').strip()
            formula = hyp.get('formula', '').strip()
            
            # Use material name as primary key, formula as secondary
            key = material.lower()
            if key not in groups:
                groups[key] = []
            groups[key].append(hyp)
        
        return groups

    def _select_best_hypotheses(self, material_groups: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Select best hypotheses from grouped materials"""
        final_hypotheses = []
        
        for material, group in material_groups.items():
            if len(group) == 1:
                # Single suggestion
                final_hypotheses.append(group[0])
            else:
                # Multiple suggestions for same material - combine
                combined = self._combine_material_hypotheses(group)
                final_hypotheses.append(combined)
        
        # Sort by weighted confidence and return top 5
        final_hypotheses.sort(key=lambda x: x.get('weighted_confidence', 0), reverse=True)
        return final_hypotheses[:5]

    def _combine_material_hypotheses(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple hypotheses for the same material"""
        if not group:
            return {}
        
        # Use the hypothesis with highest confidence as base
        base = max(group, key=lambda x: x.get('confidence', 0))
        
        # Average weighted confidences
        weighted_confidences = [h.get('weighted_confidence', 0) for h in group]
        avg_weighted_confidence = np.mean(weighted_confidences)
        
        # Combine rationales
        rationales = [h.get('rationale', '') for h in group if h.get('rationale')]
        combined_rationale = ' | '.join(rationales)
        
        # Count model support
        models = [h.get('model_source', 'unknown') for h in group]
        model_support = len(set(models))
        
        result = base.copy()
        result.update({
            'weighted_confidence': float(avg_weighted_confidence),
            'rationale': combined_rationale,
            'model_support': model_support,
            'supporting_models': list(set(models)),
            'ensemble_agreement': model_support / len(group) if group else 0
        })
        
        return result

    def _calculate_uncertainty_metrics(self, hypotheses: List[Dict[str, Any]], 
                                     responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate uncertainty metrics for the ensemble"""
        
        confidences = [h.get('confidence', 0.5) for h in hypotheses]
        
        return {
            'confidence_std': float(np.std(confidences)) if confidences else 0.0,
            'confidence_range': float(max(confidences) - min(confidences)) if confidences else 0.0,
            'model_agreement': len(responses) / len(self.models) if self.models else 0.0,
            'hypothesis_diversity': len(set(h.get('material', '') for h in hypotheses)) / max(len(hypotheses), 1)
        }

    async def _add_scientific_validation(self, response: AdvancedLLMResponse, application: str):
        """Add scientific validation using specialized prompts"""
        
        validation_results = {}
        
        for domain, prompt in self.validation_prompts.items():
            try:
                # Create validation prompt for each hypothesis
                validation_prompt = f"""
                {prompt}
                
                Application: {application}
                
                Hypotheses to validate:
                {json.dumps([h for h in response.hypotheses[:3]], indent=2)}
                
                Provide validation for each hypothesis in JSON format:
                {{
                    "validations": [
                        {{
                            "material": "MaterialName",
                            "domain_confidence": 0.8,
                            "reasoning": "Scientific reasoning",
                            "concerns": ["concern1", "concern2"],
                            "supporting_evidence": ["evidence1", "evidence2"]
                        }}
                    ]
                }}
                """
                
                # Use best available model for validation
                best_model = max(self.models, key=lambda m: self.model_configs.get(m, ModelConfig(m)).weight)
                
                async with aiohttp.ClientSession() as session:
                    validation_response, _ = await self._make_async_request(
                        session, best_model, validation_prompt
                    )
                
                if validation_response:
                    try:
                        validation_data = json.loads(validation_response)
                        validation_results[domain] = validation_data
                    except:
                        validation_results[domain] = {"error": "Failed to parse validation"}
                        
            except Exception as e:
                logger.error(f"Error in {domain} validation: {e}")
                validation_results[domain] = {"error": str(e)}
        
        response.scientific_validation = validation_results

    def _create_fallback_response(self, application: str) -> AdvancedLLMResponse:
        """Create fallback response when ensemble fails"""
        return AdvancedLLMResponse(
            hypotheses=[],
            reasoning="Ensemble generation failed - no valid responses",
            confidence=0.0,
            chain_of_thought=["Fallback response due to model failures"],
            uncertainty_metrics={'model_agreement': 0.0}
        )

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information for prompts"""
        if not context:
            return "No additional context provided."
        
        formatted = []
        for key, values in context.items():
            if isinstance(values, list) and values:
                formatted.append(f"{key}: {', '.join(map(str, values))}")
        
        return '; '.join(formatted) if formatted else "No additional context provided."

    async def test_ensemble_connection(self) -> Dict[str, bool]:
        """Test connection to all models in ensemble"""
        results = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for model in self.models:
                task = self._test_single_model(session, model)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for model, response in zip(self.models, responses):
                results[model] = isinstance(response, bool) and response
        
        return results

    async def _test_single_model(self, session: aiohttp.ClientSession, model: str) -> bool:
        """Test connection to a single model"""
        try:
            test_response, _ = await self._make_async_request(
                session, model, "Test connection"
            )
            return bool(test_response)
        except Exception as e:
            logger.debug(f"Model {model} test failed: {e}")
            return False

    def get_model_specializations(self) -> Dict[str, str]:
        """Get specialization info for each model"""
        return {model: config.specialization 
                for model, config in self.model_configs.items()}

# Synchronous wrapper for backward compatibility
class EnhancedHypothesisGenerator:
    """Synchronous wrapper for the advanced async generator"""
    
    def __init__(self, base_url: str = None, models: List[str] = None):
        self.async_generator = AdvancedHypothesisGenerator(base_url, models)
    
    def generate_hypotheses(self, application: str, known_materials: List[str],
                          context: Dict[str, Any]) -> AdvancedLLMResponse:
        """Generate hypotheses synchronously"""
        return asyncio.run(
            self.async_generator.generate_ensemble_hypotheses(application, known_materials, context)
        )
    
    def test_connection(self) -> bool:
        """Test if any models are available"""
        try:
            results = asyncio.run(self.async_generator.test_ensemble_connection())
            return any(results.values())
        except Exception as e:
            logger.error(f"Error testing connection: {e}")
            # Fallback: test basic Ollama connection
            try:
                import ollama
                models = ollama.list()
                return len(models.models) > 0
            except:
                return False
    
    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get status of all models in ensemble"""
        try:
            results = asyncio.run(self.async_generator.test_ensemble_connection())
            specializations = self.async_generator.get_model_specializations()
            
            return {
                'model_status': results,
                'specializations': specializations,
                'available_models': [model for model, status in results.items() if status],
                'total_models': len(self.async_generator.models),
                'success_rate': sum(results.values()) / len(results) if results else 0
            }
        except Exception as e:
            logger.error(f"Error getting ensemble status: {e}")
            # Fallback: use basic Ollama info
            try:
                import ollama
                models = ollama.list()
                available_model_names = [model.model for model in models.models]
                
                return {
                    'model_status': {model: True for model in available_model_names},
                    'specializations': {model: 'general' for model in available_model_names},
                    'available_models': available_model_names,
                    'total_models': len(available_model_names),
                    'success_rate': 1.0 if available_model_names else 0.0
                }
            except:
                return {
                    'model_status': {},
                    'specializations': {},
                    'available_models': [],
                    'total_models': 0,
                    'success_rate': 0.0
                }
