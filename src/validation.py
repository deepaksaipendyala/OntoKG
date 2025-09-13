"""
Validation module for Materials Ontology Expansion
Validates LLM hypotheses against Matbench and materials databases
"""

import os
import requests
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import logging

from config import SystemConfig

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of hypothesis validation"""
    is_valid: bool
    confidence: float
    evidence: Dict[str, Any]
    source: str
    notes: str

class MaterialsValidator:
    """Validates materials hypotheses against known databases"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config
        self.mp_api_key = os.getenv('MP_API_KEY')
        self.validation_cache = {}
        
        # Initialize data manager if config provided
        if config:
            # Import DataManager here to avoid circular import
            from data_manager import DataManager
            self.data_manager = DataManager(config)
        else:
            self.data_manager = None
        
        # Known materials properties for validation
        self.materials_properties = {
            # Capacitor materials
            "BaTiO3": {"dielectric_constant": 1500, "band_gap": 3.2, "type": "perovskite"},
            "PbTiO3": {"dielectric_constant": 200, "band_gap": 3.4, "type": "perovskite"},
            "SrTiO3": {"dielectric_constant": 300, "band_gap": 3.2, "type": "perovskite"},
            "CaTiO3": {"dielectric_constant": 150, "band_gap": 3.5, "type": "perovskite"},
            "KNbO3": {"dielectric_constant": 700, "band_gap": 3.3, "type": "perovskite"},
            
            # Thermoelectric materials
            "Bi2Te3": {"zt": 0.8, "thermal_conductivity": 1.5, "type": "chalcogenide"},
            "PbTe": {"zt": 0.8, "thermal_conductivity": 2.0, "type": "chalcogenide"},
            "SnSe": {"zt": 2.6, "thermal_conductivity": 0.7, "type": "chalcogenide"},
            "Bi2Se3": {"zt": 0.4, "thermal_conductivity": 2.0, "type": "chalcogenide"},
            "AgSbTe2": {"zt": 1.4, "thermal_conductivity": 1.0, "type": "chalcogenide"},
            
            # Solar cell materials
            "CH3NH3PbI3": {"band_gap": 1.6, "absorption_coefficient": 1e5, "type": "perovskite"},
            "CsSnI3": {"band_gap": 1.3, "absorption_coefficient": 8e4, "type": "perovskite"},
            "Si": {"band_gap": 1.1, "absorption_coefficient": 1e4, "type": "elemental"},
            "CdTe": {"band_gap": 1.5, "absorption_coefficient": 5e4, "type": "compound"},
            "CIGS": {"band_gap": 1.2, "absorption_coefficient": 1e5, "type": "compound"}
        }
        
        # Application-specific validation criteria
        self.validation_criteria = {
            "capacitor": {
                "required_properties": ["dielectric_constant"],
                "thresholds": {"dielectric_constant": 50},  # Minimum for capacitors
                "preferred_types": ["perovskite", "ferroelectric"]
            },
            "thermoelectric_device": {
                "required_properties": ["zt"],
                "thresholds": {"zt": 0.5},  # Minimum ZT for thermoelectrics
                "preferred_types": ["chalcogenide", "skutterudite"]
            },
            "solar_cell": {
                "required_properties": ["band_gap"],
                "thresholds": {"band_gap_min": 1.0, "band_gap_max": 1.8},
                "preferred_types": ["perovskite", "compound", "elemental"]
            }
        }
    
    def validate_hypothesis(self, material: str, application: str, 
                          relationship: str = "USED_IN") -> ValidationResult:
        """Validate a hypothesis about material-application relationship"""
        
        cache_key = f"{material}_{application}_{relationship}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        # Try data manager validation first if available
        if self.data_manager:
            # Import Hypothesis here to avoid circular import
            from knowledge_graph import Hypothesis
            hypothesis = Hypothesis(
                material=material,
                application=application,
                relationship=relationship
            )
            
            data_manager_result = self.data_manager.validate_hypothesis(hypothesis)
            
            if data_manager_result.get('is_valid', False):
                result = ValidationResult(
                    is_valid=True,
                    confidence=data_manager_result['confidence'],
                    evidence=data_manager_result['evidence'],
                    source=data_manager_result['source'],
                    notes=f"Validated by {data_manager_result['source']}"
                )
                self.validation_cache[cache_key] = result
                return result
        
        # Check if material exists in our database
        if material not in self.materials_properties:
            result = ValidationResult(
                is_valid=False,
                confidence=0.0,
                evidence={},
                source="Unknown material",
                notes=f"Material {material} not found in validation database"
            )
            self.validation_cache[cache_key] = result
            return result
        
        material_props = self.materials_properties[material]
        
        # Get validation criteria for the application
        if application not in self.validation_criteria:
            result = ValidationResult(
                is_valid=False,
                confidence=0.3,
                evidence=material_props,
                source="Unknown application",
                notes=f"Application {application} not in validation criteria"
            )
            self.validation_cache[cache_key] = result
            return result
        
        criteria = self.validation_criteria[application]
        
        # Check required properties
        evidence = {}
        confidence_factors = []
        
        for prop in criteria["required_properties"]:
            if prop in material_props:
                evidence[prop] = material_props[prop]
                
                # Check against thresholds
                if prop in criteria["thresholds"]:
                    threshold = criteria["thresholds"][prop]
                    if prop == "band_gap":
                        # Special handling for band gap range
                        bg_min = criteria["thresholds"].get("band_gap_min", 0)
                        bg_max = criteria["thresholds"].get("band_gap_max", float('inf'))
                        if bg_min <= material_props[prop] <= bg_max:
                            confidence_factors.append(0.9)
                        else:
                            confidence_factors.append(0.3)
                    elif material_props[prop] >= threshold:
                        confidence_factors.append(0.9)
                    else:
                        confidence_factors.append(0.4)
                else:
                    confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.2)
        
        # Check material type preference
        material_type = material_props.get("type", "unknown")
        if material_type in criteria.get("preferred_types", []):
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Calculate overall confidence
        if confidence_factors:
            overall_confidence = sum(confidence_factors) / len(confidence_factors)
        else:
            overall_confidence = 0.3
        
        # Determine validity
        is_valid = overall_confidence >= 0.6
        
        result = ValidationResult(
            is_valid=is_valid,
            confidence=overall_confidence,
            evidence=evidence,
            source="Materials Database",
            notes=f"Validated against {application} criteria"
        )
        
        self.validation_cache[cache_key] = result
        return result
    
    def validate_capacitor_material(self, material: str) -> ValidationResult:
        """Specific validation for capacitor materials"""
        return self.validate_hypothesis(material, "capacitor", "USED_IN")
    
    def validate_thermoelectric_material(self, material: str) -> ValidationResult:
        """Specific validation for thermoelectric materials"""
        return self.validate_hypothesis(material, "thermoelectric_device", "USED_IN")
    
    def validate_solar_cell_material(self, material: str) -> ValidationResult:
        """Specific validation for solar cell materials"""
        return self.validate_hypothesis(material, "solar_cell", "USED_IN")
    
    def get_material_properties(self, material: str) -> Dict[str, Any]:
        """Get properties for a material"""
        return self.materials_properties.get(material, {})
    
    def search_materials_by_property(self, property_name: str, 
                                   min_value: float = None,
                                   max_value: float = None) -> List[Tuple[str, float]]:
        """Search materials by property value range"""
        results = []
        
        for material, props in self.materials_properties.items():
            if property_name in props:
                value = props[property_name]
                
                if min_value is not None and value < min_value:
                    continue
                if max_value is not None and value > max_value:
                    continue
                    
                results.append((material, value))
        
        # Sort by property value
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_recommended_materials(self, application: str, 
                                limit: int = 10) -> List[Dict[str, Any]]:
        """Get recommended materials for an application based on properties"""
        if application not in self.validation_criteria:
            return []
        
        criteria = self.validation_criteria[application]
        recommendations = []
        
        for material, props in self.materials_properties.items():
            validation_result = self.validate_hypothesis(material, application)
            
            if validation_result.is_valid:
                recommendations.append({
                    "material": material,
                    "confidence": validation_result.confidence,
                    "properties": props,
                    "evidence": validation_result.evidence
                })
        
        # Sort by confidence and limit results
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        return recommendations[:limit]
    
    def add_material_to_database(self, material: str, properties: Dict[str, Any]):
        """Add a new material to the validation database"""
        self.materials_properties[material] = properties
        # Clear cache for this material
        keys_to_remove = [key for key in self.validation_cache.keys() if material in key]
        for key in keys_to_remove:
            del self.validation_cache[key]
    
    def export_validation_database(self) -> Dict[str, Any]:
        """Export the validation database for backup or sharing"""
        return {
            "materials_properties": self.materials_properties,
            "validation_criteria": self.validation_criteria,
            "cache_size": len(self.validation_cache)
        }
    
    def import_validation_database(self, data: Dict[str, Any]):
        """Import validation database from backup"""
        if "materials_properties" in data:
            self.materials_properties.update(data["materials_properties"])
        if "validation_criteria" in data:
            self.validation_criteria.update(data["validation_criteria"])
        # Clear cache after import
        self.validation_cache.clear()

class MatbenchValidator:
    """Validator using Matbench datasets (simplified interface)"""
    
    def __init__(self):
        self.available_datasets = {
            "dielectric": "Materials with dielectric constant data",
            "band_gap": "Materials with band gap data",
            "formation_energy": "Materials with formation energy data"
        }
    
    def validate_dielectric_constant(self, material: str, min_value: float = 50) -> bool:
        """Validate dielectric constant for capacitor applications"""
        # Simplified validation - in real implementation would query Matbench
        known_dielectrics = {
            "BaTiO3": 1500, "SrTiO3": 300, "CaTiO3": 150, "PbTiO3": 200
        }
        
        if material in known_dielectrics:
            return known_dielectrics[material] >= min_value
        
        return False
    
    def validate_band_gap(self, material: str, min_value: float = 1.0, 
                         max_value: float = 1.8) -> bool:
        """Validate band gap for solar cell applications"""
        known_band_gaps = {
            "Si": 1.1, "CdTe": 1.5, "CH3NH3PbI3": 1.6, "CsSnI3": 1.3
        }
        
        if material in known_band_gaps:
            bg = known_band_gaps[material]
            return min_value <= bg <= max_value
        
        return False

