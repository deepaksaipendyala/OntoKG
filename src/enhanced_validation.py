"""
Enhanced Validation Module for Materials Ontology Expansion
Features ML-based property prediction, uncertainty quantification, and multi-source validation
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import requests
from datetime import datetime
import hashlib

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class EnhancedValidationResult:
    """Enhanced validation result with uncertainty quantification"""
    is_valid: bool
    confidence: float
    evidence: Dict[str, Any]
    source: str
    notes: str
    uncertainty_bounds: Tuple[float, float] = (0.0, 1.0)
    ml_predictions: Dict[str, float] = field(default_factory=dict)
    consensus_score: float = 0.0
    validation_methods: List[str] = field(default_factory=list)
    property_predictions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    risk_assessment: Dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class PropertyPrediction:
    """ML-based property prediction with uncertainty"""
    property_name: str
    predicted_value: float
    uncertainty: float
    confidence_interval: Tuple[float, float]
    method: str
    feature_importance: Dict[str, float] = field(default_factory=dict)

class MLPropertyPredictor:
    """Machine learning-based property predictor"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_encoders = {}
        self.is_trained = False
        
        # Property-specific models
        self.property_models = {
            'dielectric_constant': {
                'model': RandomForestRegressor(n_estimators=100, random_state=42),
                'features': ['n_atoms', 'volume', 'electronegativity_diff', 'ionic_character'],
                'target_range': (1, 10000)
            },
            'band_gap': {
                'model': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'features': ['n_atoms', 'volume', 'electronegativity_avg', 'crystal_system'],
                'target_range': (0, 10)
            },
            'zt': {
                'model': RandomForestRegressor(n_estimators=150, random_state=42),
                'features': ['n_atoms', 'volume', 'thermal_conductivity', 'electrical_conductivity'],
                'target_range': (0, 5)
            }
        }
        
        self._initialize_training_data()
    
    def _initialize_training_data(self):
        """Initialize training data from known materials"""
        self.training_data = {
            'materials': [
                # Capacitor materials
                {'name': 'BaTiO3', 'formula': 'BaTiO3', 'n_atoms': 5, 'volume': 64.1, 
                 'electronegativity_diff': 2.4, 'ionic_character': 0.8, 'crystal_system': 1,
                 'dielectric_constant': 1500, 'band_gap': 3.2},
                {'name': 'SrTiO3', 'formula': 'SrTiO3', 'n_atoms': 5, 'volume': 59.5,
                 'electronegativity_diff': 2.2, 'ionic_character': 0.75, 'crystal_system': 1,
                 'dielectric_constant': 300, 'band_gap': 3.2},
                {'name': 'PbTiO3', 'formula': 'PbTiO3', 'n_atoms': 5, 'volume': 63.8,
                 'electronegativity_diff': 1.8, 'ionic_character': 0.6, 'crystal_system': 1,
                 'dielectric_constant': 200, 'band_gap': 3.4},
                
                # Thermoelectric materials  
                {'name': 'Bi2Te3', 'formula': 'Bi2Te3', 'n_atoms': 5, 'volume': 174.2,
                 'thermal_conductivity': 1.5, 'electrical_conductivity': 1000, 'crystal_system': 2,
                 'zt': 0.8, 'band_gap': 0.15},
                {'name': 'SnSe', 'formula': 'SnSe', 'n_atoms': 2, 'volume': 49.8,
                 'thermal_conductivity': 0.7, 'electrical_conductivity': 100, 'crystal_system': 3,
                 'zt': 2.6, 'band_gap': 0.9},
                {'name': 'PbTe', 'formula': 'PbTe', 'n_atoms': 2, 'volume': 68.4,
                 'thermal_conductivity': 2.0, 'electrical_conductivity': 500, 'crystal_system': 1,
                 'zt': 0.8, 'band_gap': 0.31},
                
                # Solar cell materials
                {'name': 'Si', 'formula': 'Si', 'n_atoms': 1, 'volume': 20.0,
                 'electronegativity_avg': 1.9, 'crystal_system': 4,
                 'band_gap': 1.1},
                {'name': 'CdTe', 'formula': 'CdTe', 'n_atoms': 2, 'volume': 58.8,
                 'electronegativity_avg': 1.7, 'crystal_system': 5,
                 'band_gap': 1.5},
                {'name': 'CH3NH3PbI3', 'formula': 'CH3NH3PbI3', 'n_atoms': 12, 'volume': 249.5,
                 'electronegativity_avg': 2.1, 'crystal_system': 1,
                 'band_gap': 1.6}
            ]
        }
        
        # Train models with synthetic data augmentation
        self._train_models()
    
    def _train_models(self):
        """Train ML models for property prediction"""
        df = pd.DataFrame(self.training_data['materials'])
        
        for prop_name, prop_config in self.property_models.items():
            if prop_name in df.columns:
                try:
                    # Prepare features and target
                    features = prop_config['features']
                    available_features = [f for f in features if f in df.columns]
                    
                    if len(available_features) < 2:
                        logger.warning(f"Insufficient features for {prop_name}")
                        continue
                    
                    X = df[available_features].fillna(0)
                    y = df[prop_name].dropna()
                    
                    # Align X and y
                    common_idx = X.index.intersection(y.index)
                    X = X.loc[common_idx]
                    y = y.loc[common_idx]
                    
                    if len(X) < 3:
                        logger.warning(f"Insufficient data for {prop_name}")
                        continue
                    
                    # Data augmentation with noise
                    X_aug, y_aug = self._augment_data(X, y, n_samples=50)
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_aug)
                    
                    # Train model
                    model = prop_config['model']
                    model.fit(X_scaled, y_aug)
                    
                    # Store model and scaler
                    self.models[prop_name] = model
                    self.scalers[prop_name] = scaler
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_scaled, y_aug, cv=3)
                    logger.info(f"{prop_name} model CV score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {prop_name} model: {e}")
        
        self.is_trained = True
    
    def _augment_data(self, X: pd.DataFrame, y: pd.Series, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Augment training data with synthetic samples"""
        X_orig = X.values
        y_orig = y.values
        
        # Generate synthetic samples by adding noise
        noise_std = 0.1
        X_synthetic = []
        y_synthetic = []
        
        for _ in range(n_samples):
            # Random sample from original data
            idx = np.random.randint(0, len(X_orig))
            x_base = X_orig[idx]
            y_base = y_orig[idx]
            
            # Add noise
            x_noisy = x_base + np.random.normal(0, noise_std * np.abs(x_base), size=x_base.shape)
            y_noisy = y_base + np.random.normal(0, noise_std * abs(y_base))
            
            X_synthetic.append(x_noisy)
            y_synthetic.append(y_noisy)
        
        # Combine original and synthetic
        X_combined = np.vstack([X_orig, np.array(X_synthetic)])
        y_combined = np.hstack([y_orig, np.array(y_synthetic)])
        
        return X_combined, y_combined
    
    def predict_property(self, material_features: Dict[str, float], 
                        property_name: str) -> Optional[PropertyPrediction]:
        """Predict material property with uncertainty estimation"""
        
        if not self.is_trained or property_name not in self.models:
            return None
        
        try:
            model = self.models[property_name]
            scaler = self.scalers[property_name]
            prop_config = self.property_models[property_name]
            
            # Prepare features
            features = prop_config['features']
            feature_values = []
            
            for feature in features:
                if feature in material_features:
                    feature_values.append(material_features[feature])
                else:
                    # Use default value or skip
                    feature_values.append(0.0)
            
            if not feature_values:
                return None
            
            # Scale and predict
            X = np.array(feature_values).reshape(1, -1)
            X_scaled = scaler.transform(X)
            
            prediction = model.predict(X_scaled)[0]
            
            # Estimate uncertainty using ensemble predictions
            uncertainty = self._estimate_uncertainty(model, X_scaled, property_name)
            
            # Confidence interval
            ci_lower = max(prediction - 2 * uncertainty, prop_config['target_range'][0])
            ci_upper = min(prediction + 2 * uncertainty, prop_config['target_range'][1])
            
            # Feature importance (for tree-based models)
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                for feature, importance in zip(features, model.feature_importances_):
                    feature_importance[feature] = float(importance)
            
            return PropertyPrediction(
                property_name=property_name,
                predicted_value=float(prediction),
                uncertainty=float(uncertainty),
                confidence_interval=(float(ci_lower), float(ci_upper)),
                method=f"ML_{type(model).__name__}",
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Error predicting {property_name}: {e}")
            return None
    
    def _estimate_uncertainty(self, model, X_scaled: np.ndarray, property_name: str) -> float:
        """Estimate prediction uncertainty"""
        try:
            if hasattr(model, 'estimators_'):
                # For ensemble methods, use prediction variance
                predictions = np.array([estimator.predict(X_scaled)[0] for estimator in model.estimators_])
                uncertainty = np.std(predictions)
            else:
                # For other models, use a heuristic based on training performance
                # This is simplified - in practice, you'd use more sophisticated methods
                prop_config = self.property_models[property_name]
                target_range = prop_config['target_range']
                uncertainty = (target_range[1] - target_range[0]) * 0.1  # 10% of range
            
            return max(uncertainty, 0.01)  # Minimum uncertainty
            
        except Exception as e:
            logger.error(f"Error estimating uncertainty: {e}")
            return 0.1  # Default uncertainty

class EnhancedMaterialsValidator:
    """Enhanced materials validator with ML predictions and multi-source validation"""
    
    def __init__(self, config=None):
        self.config = config
        self.ml_predictor = MLPropertyPredictor()
        self.validation_cache = {}
        self.external_apis = self._initialize_external_apis()
        
        # Enhanced materials database
        self.materials_database = self._load_enhanced_database()
        
        # Validation thresholds for applications
        self.application_criteria = {
            'capacitor': {
                'dielectric_constant': {'min': 50, 'weight': 0.8},
                'band_gap': {'min': 2.0, 'weight': 0.3},
                'crystal_stability': {'min': 0.7, 'weight': 0.5}
            },
            'thermoelectric_device': {
                'zt': {'min': 0.5, 'weight': 0.9},
                'thermal_conductivity': {'max': 3.0, 'weight': 0.6},
                'electrical_conductivity': {'min': 100, 'weight': 0.4}
            },
            'solar_cell': {
                'band_gap': {'min': 1.0, 'max': 1.8, 'weight': 0.9},
                'absorption_coefficient': {'min': 1000, 'weight': 0.7},
                'stability': {'min': 0.6, 'weight': 0.5}
            }
        }
    
    def _initialize_external_apis(self) -> Dict[str, Dict[str, str]]:
        """Initialize external API configurations"""
        return {
            'materials_project': {
                'base_url': 'https://api.materialsproject.org/v1',
                'api_key': os.getenv('MP_API_KEY', ''),
                'enabled': bool(os.getenv('MP_API_KEY'))
            },
            'aflow': {
                'base_url': 'http://aflowlib.org/API',
                'enabled': True
            },
            'oqmd': {
                'base_url': 'http://oqmd.org/oqmdapi',
                'enabled': True
            }
        }
    
    def _load_enhanced_database(self) -> Dict[str, Dict[str, Any]]:
        """Load enhanced materials database with more properties"""
        return {
            # Capacitor materials
            "BaTiO3": {
                "dielectric_constant": 1500, "band_gap": 3.2, "type": "perovskite",
                "crystal_stability": 0.9, "synthesis_difficulty": "easy",
                "cost_factor": "low", "environmental_impact": "low"
            },
            "SrTiO3": {
                "dielectric_constant": 300, "band_gap": 3.2, "type": "perovskite",
                "crystal_stability": 0.85, "synthesis_difficulty": "easy",
                "cost_factor": "low", "environmental_impact": "low"
            },
            "PbTiO3": {
                "dielectric_constant": 200, "band_gap": 3.4, "type": "perovskite",
                "crystal_stability": 0.8, "synthesis_difficulty": "medium",
                "cost_factor": "medium", "environmental_impact": "high"
            },
            
            # Thermoelectric materials
            "Bi2Te3": {
                "zt": 0.8, "thermal_conductivity": 1.5, "type": "chalcogenide",
                "electrical_conductivity": 1000, "crystal_stability": 0.7,
                "synthesis_difficulty": "medium", "cost_factor": "high"
            },
            "SnSe": {
                "zt": 2.6, "thermal_conductivity": 0.7, "type": "chalcogenide",
                "electrical_conductivity": 100, "crystal_stability": 0.6,
                "synthesis_difficulty": "hard", "cost_factor": "medium"
            },
            "PbTe": {
                "zt": 0.8, "thermal_conductivity": 2.0, "type": "chalcogenide",
                "electrical_conductivity": 500, "crystal_stability": 0.75,
                "synthesis_difficulty": "medium", "cost_factor": "high"
            },
            
            # Solar cell materials
            "Si": {
                "band_gap": 1.1, "absorption_coefficient": 10000, "type": "elemental",
                "stability": 0.95, "synthesis_difficulty": "easy",
                "cost_factor": "low", "environmental_impact": "low"
            },
            "CdTe": {
                "band_gap": 1.5, "absorption_coefficient": 50000, "type": "compound",
                "stability": 0.8, "synthesis_difficulty": "medium",
                "cost_factor": "medium", "environmental_impact": "high"
            },
            "CH3NH3PbI3": {
                "band_gap": 1.6, "absorption_coefficient": 100000, "type": "perovskite",
                "stability": 0.4, "synthesis_difficulty": "medium",
                "cost_factor": "medium", "environmental_impact": "high"
            },
            "CsSnI3": {
                "band_gap": 1.3, "absorption_coefficient": 80000, "type": "perovskite",
                "stability": 0.6, "synthesis_difficulty": "hard",
                "cost_factor": "high", "environmental_impact": "medium"
            }
        }
    
    async def validate_hypothesis_enhanced(self, material: str, application: str, 
                                         material_properties: Optional[Dict[str, float]] = None) -> EnhancedValidationResult:
        """Enhanced validation with ML predictions and multi-source verification"""
        
        cache_key = self._get_cache_key(material, application, material_properties)
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        validation_methods = []
        evidence = {}
        ml_predictions = {}
        property_predictions = {}
        risk_assessment = {}
        
        # 1. Database lookup validation
        db_result = self._validate_against_database(material, application)
        if db_result:
            validation_methods.append("database_lookup")
            evidence.update(db_result)
        
        # 2. ML-based property prediction
        if material_properties:
            ml_result = self._validate_with_ml_predictions(material, application, material_properties)
            if ml_result:
                validation_methods.append("ml_prediction")
                ml_predictions.update(ml_result['predictions'])
                property_predictions.update(ml_result['property_details'])
        
        # 3. External API validation
        api_result = await self._validate_with_external_apis(material, application)
        if api_result:
            validation_methods.append("external_apis")
            evidence.update(api_result)
        
        # 4. Structure-property relationship validation
        structure_result = self._validate_structure_property_relationships(material, application)
        if structure_result:
            validation_methods.append("structure_property")
            evidence.update(structure_result)
        
        # 5. Risk assessment
        risk_assessment = self._assess_risks(material, application, evidence)
        
        # Combine all validation results
        final_result = self._combine_validation_results(
            validation_methods, evidence, ml_predictions, property_predictions, risk_assessment
        )
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus_score(validation_methods, evidence)
        
        # Estimate uncertainty bounds
        uncertainty_bounds = self._estimate_uncertainty_bounds(evidence, ml_predictions)
        
        result = EnhancedValidationResult(
            is_valid=final_result['is_valid'],
            confidence=final_result['confidence'],
            evidence=evidence,
            source=f"Enhanced validation ({len(validation_methods)} methods)",
            notes=final_result['notes'],
            uncertainty_bounds=uncertainty_bounds,
            ml_predictions=ml_predictions,
            consensus_score=consensus_score,
            validation_methods=validation_methods,
            property_predictions=property_predictions,
            risk_assessment=risk_assessment
        )
        
        self.validation_cache[cache_key] = result
        return result
    
    def _validate_against_database(self, material: str, application: str) -> Optional[Dict[str, Any]]:
        """Validate against enhanced materials database"""
        if material not in self.materials_database:
            return None
        
        material_props = self.materials_database[material]
        criteria = self.application_criteria.get(application, {})
        
        validation_score = 0.0
        total_weight = 0.0
        property_checks = {}
        
        for prop, requirements in criteria.items():
            if prop in material_props:
                value = material_props[prop]
                weight = requirements.get('weight', 1.0)
                
                # Check constraints
                passes = True
                if 'min' in requirements and value < requirements['min']:
                    passes = False
                if 'max' in requirements and value > requirements['max']:
                    passes = False
                
                property_checks[prop] = {
                    'value': value,
                    'passes': passes,
                    'weight': weight
                }
                
                if passes:
                    validation_score += weight
                total_weight += weight
        
        if total_weight > 0:
            normalized_score = validation_score / total_weight
        else:
            normalized_score = 0.0
        
        return {
            'database_score': normalized_score,
            'property_checks': property_checks,
            'material_properties': material_props
        }
    
    def _validate_with_ml_predictions(self, material: str, application: str, 
                                    material_features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Validate using ML property predictions"""
        criteria = self.application_criteria.get(application, {})
        predictions = {}
        property_details = {}
        
        for prop in criteria.keys():
            prediction = self.ml_predictor.predict_property(material_features, prop)
            if prediction:
                predictions[prop] = prediction.predicted_value
                property_details[prop] = {
                    'predicted_value': prediction.predicted_value,
                    'uncertainty': prediction.uncertainty,
                    'confidence_interval': prediction.confidence_interval,
                    'method': prediction.method,
                    'feature_importance': prediction.feature_importance
                }
        
        if not predictions:
            return None
        
        return {
            'predictions': predictions,
            'property_details': property_details
        }
    
    async def _validate_with_external_apis(self, material: str, application: str) -> Optional[Dict[str, Any]]:
        """Validate using external materials databases"""
        # This is a simplified implementation
        # In practice, you would make actual API calls
        
        api_results = {}
        
        # Simulate API calls with known data
        if material in self.materials_database:
            api_results['materials_project'] = {
                'found': True,
                'properties': self.materials_database[material]
            }
        
        return api_results if api_results else None
    
    def _validate_structure_property_relationships(self, material: str, application: str) -> Optional[Dict[str, Any]]:
        """Validate based on structure-property relationships"""
        if material not in self.materials_database:
            return None
        
        material_props = self.materials_database[material]
        material_type = material_props.get('type', 'unknown')
        
        # Structure-property rules
        structure_rules = {
            'capacitor': {
                'perovskite': 0.9,  # Perovskites are excellent for capacitors
                'compound': 0.6,
                'elemental': 0.2
            },
            'thermoelectric_device': {
                'chalcogenide': 0.9,  # Chalcogenides are good thermoelectrics
                'compound': 0.5,
                'elemental': 0.3
            },
            'solar_cell': {
                'perovskite': 0.8,  # Perovskites are promising for solar cells
                'compound': 0.7,
                'elemental': 0.9  # Silicon is excellent
            }
        }
        
        structure_score = structure_rules.get(application, {}).get(material_type, 0.5)
        
        return {
            'structure_property_score': structure_score,
            'material_type': material_type,
            'structure_rationale': f"{material_type} materials are suitable for {application}"
        }
    
    def _assess_risks(self, material: str, application: str, evidence: Dict[str, Any]) -> Dict[str, str]:
        """Assess risks associated with the material-application combination"""
        risks = {}
        
        if material not in self.materials_database:
            risks['data_availability'] = 'high'
            return risks
        
        material_props = self.materials_database[material]
        
        # Environmental risk
        env_impact = material_props.get('environmental_impact', 'unknown')
        if env_impact == 'high':
            risks['environmental'] = 'high'
        elif env_impact == 'medium':
            risks['environmental'] = 'medium'
        else:
            risks['environmental'] = 'low'
        
        # Synthesis risk
        synthesis_difficulty = material_props.get('synthesis_difficulty', 'unknown')
        if synthesis_difficulty == 'hard':
            risks['synthesis'] = 'high'
        elif synthesis_difficulty == 'medium':
            risks['synthesis'] = 'medium'
        else:
            risks['synthesis'] = 'low'
        
        # Cost risk
        cost_factor = material_props.get('cost_factor', 'unknown')
        if cost_factor == 'high':
            risks['cost'] = 'high'
        elif cost_factor == 'medium':
            risks['cost'] = 'medium'
        else:
            risks['cost'] = 'low'
        
        # Stability risk
        stability = material_props.get('crystal_stability', material_props.get('stability', 0.5))
        if stability < 0.5:
            risks['stability'] = 'high'
        elif stability < 0.8:
            risks['stability'] = 'medium'
        else:
            risks['stability'] = 'low'
        
        return risks
    
    def _combine_validation_results(self, methods: List[str], evidence: Dict[str, Any],
                                  ml_predictions: Dict[str, float], 
                                  property_predictions: Dict[str, Dict[str, float]],
                                  risk_assessment: Dict[str, str]) -> Dict[str, Any]:
        """Combine results from multiple validation methods"""
        
        scores = []
        notes = []
        
        # Database score
        if 'database_lookup' in methods and 'database_score' in evidence:
            scores.append(evidence['database_score'])
            notes.append(f"Database validation: {evidence['database_score']:.2f}")
        
        # ML prediction score
        if 'ml_prediction' in methods and ml_predictions:
            ml_score = np.mean(list(ml_predictions.values())) / 100  # Normalize
            scores.append(min(ml_score, 1.0))
            notes.append(f"ML prediction score: {ml_score:.2f}")
        
        # Structure-property score
        if 'structure_property' in methods and 'structure_property_score' in evidence:
            scores.append(evidence['structure_property_score'])
            notes.append(f"Structure-property score: {evidence['structure_property_score']:.2f}")
        
        # External API score (simplified)
        if 'external_apis' in methods:
            api_score = 0.7  # Placeholder
            scores.append(api_score)
            notes.append(f"External API validation: {api_score:.2f}")
        
        # Calculate final confidence
        if scores:
            final_confidence = np.mean(scores)
            is_valid = final_confidence > 0.6
        else:
            final_confidence = 0.0
            is_valid = False
        
        # Adjust for risks
        high_risks = sum(1 for risk in risk_assessment.values() if risk == 'high')
        if high_risks > 2:
            final_confidence *= 0.7  # Reduce confidence for high-risk materials
        
        return {
            'is_valid': is_valid,
            'confidence': final_confidence,
            'notes': '; '.join(notes)
        }
    
    def _calculate_consensus_score(self, methods: List[str], evidence: Dict[str, Any]) -> float:
        """Calculate consensus score across validation methods"""
        if not methods:
            return 0.0
        
        # Weight different methods
        method_weights = {
            'database_lookup': 1.0,
            'ml_prediction': 0.8,
            'external_apis': 1.2,
            'structure_property': 0.6
        }
        
        total_weight = sum(method_weights.get(method, 1.0) for method in methods)
        weighted_score = sum(method_weights.get(method, 1.0) for method in methods)
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _estimate_uncertainty_bounds(self, evidence: Dict[str, Any], 
                                   ml_predictions: Dict[str, float]) -> Tuple[float, float]:
        """Estimate uncertainty bounds for the validation"""
        
        uncertainties = []
        
        # Database uncertainty (low for known materials)
        if 'database_score' in evidence:
            uncertainties.append(0.1)  # Low uncertainty for database lookups
        
        # ML prediction uncertainty
        if ml_predictions:
            uncertainties.append(0.3)  # Higher uncertainty for ML predictions
        
        # Default uncertainty
        if not uncertainties:
            uncertainties.append(0.5)
        
        avg_uncertainty = np.mean(uncertainties)
        
        return (max(0.0, 0.5 - avg_uncertainty), min(1.0, 0.5 + avg_uncertainty))
    
    def _get_cache_key(self, material: str, application: str, 
                      material_properties: Optional[Dict[str, float]]) -> str:
        """Generate cache key for validation result"""
        key_data = f"{material}_{application}"
        if material_properties:
            prop_str = json.dumps(material_properties, sort_keys=True)
            key_data += f"_{hashlib.md5(prop_str.encode()).hexdigest()[:8]}"
        return key_data
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics about validation performance"""
        return {
            'cache_size': len(self.validation_cache),
            'ml_models_trained': len(self.ml_predictor.models),
            'database_size': len(self.materials_database),
            'external_apis_available': sum(1 for api in self.external_apis.values() if api['enabled']),
            'validation_methods': ['database_lookup', 'ml_prediction', 'external_apis', 'structure_property']
        }

# Synchronous wrapper for backward compatibility
def validate_hypothesis_sync(material: str, application: str, 
                           material_properties: Optional[Dict[str, float]] = None) -> EnhancedValidationResult:
    """Synchronous wrapper for enhanced validation"""
    import asyncio
    validator = EnhancedMaterialsValidator()
    return asyncio.run(validator.validate_hypothesis_enhanced(material, application, material_properties))
