"""
数据生成和预处理模块
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from typing import Tuple, Dict
import warnings

class SyntheticDataGenerator:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_medical_data(self, n_total: int = 2000, n_features: int = 8) -> Tuple[np.ndarray, np.ndarray, list]:
        X_raw, y_raw = make_regression(
            n_samples=n_total,
            n_features=n_features,
            noise=0.1,
            random_state=self.random_state
        )
        
        feature_names = [
            'age', 'education_years', 'apoe4_carriers', 'baseline_mmse',
            'hippocampus_volume', 'tau_protein', 'amyloid_beta', 'cognitive_reserve'
        ][:n_features]
        
        X = self._transform_to_medical_features(X_raw, feature_names)
        y = self._transform_to_cognitive_change(y_raw, X)
        
        return X, y, feature_names
    
    def _transform_to_medical_features(self, X_raw: np.ndarray, feature_names: list) -> np.ndarray:
        X = X_raw.copy()
        n_samples = X.shape[0]
        
        if 'age' in feature_names:
            age_idx = feature_names.index('age')
            X[:, age_idx] = 65 + 10 * (X[:, age_idx] - X[:, age_idx].min()) / (X[:, age_idx].max() - X[:, age_idx].min())
            X[:, age_idx] = np.clip(X[:, age_idx], 60, 90)
        
        if 'education_years' in feature_names:
            edu_idx = feature_names.index('education_years')
            X[:, edu_idx] = 8 + 12 * (X[:, edu_idx] - X[:, edu_idx].min()) / (X[:, edu_idx].max() - X[:, edu_idx].min())
            X[:, edu_idx] = np.round(X[:, edu_idx]).astype(int)
        
        if 'apoe4_carriers' in feature_names:
            apoe_idx = feature_names.index('apoe4_carriers')
            X[:, apoe_idx] = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
        
        if 'baseline_mmse' in feature_names:
            mmse_idx = feature_names.index('baseline_mmse')
            X[:, mmse_idx] = 20 + 10 * (X[:, mmse_idx] - X[:, mmse_idx].min()) / (X[:, mmse_idx].max() - X[:, mmse_idx].min())
            X[:, mmse_idx] = np.clip(X[:, mmse_idx], 18, 30)
        
        return X
    
    def _transform_to_cognitive_change(self, y_raw: np.ndarray, X: np.ndarray) -> np.ndarray:
        y = (y_raw - y_raw.mean()) / y_raw.std()
        
        if X.shape[1] > 0:
            age_effect = (X[:, 0] - 70) * 0.1
            y += age_effect
        
        if X.shape[1] > 2:
            apoe_effect = X[:, 2] * 0.5
            y += apoe_effect
        
        y_min, y_max = y.min(), y.max()
        y_scaled = 5 + 15 * (y - y_min) / (y_max - y_min)
        
        return y_scaled

class DataSplitter:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def split_for_ppi(self, X: np.ndarray, y: np.ndarray,
                     n_labeled: int = 200, n_unlabeled: int = 1000,
                     n_pretrain: int = 500) -> Dict[str, np.ndarray]:
        total_needed = n_labeled + n_unlabeled + n_pretrain
        
        if len(X) < total_needed:
            scale = len(X) / total_needed
            n_labeled = int(n_labeled * scale)
            n_unlabeled = int(n_unlabeled * scale)
            n_pretrain = len(X) - n_labeled - n_unlabeled
        
        X_pretrain, X_remaining, y_pretrain, y_remaining = train_test_split(
            X, y, test_size=n_labeled + n_unlabeled, random_state=self.random_state
        )
        
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X_remaining, y_remaining, test_size=n_unlabeled, random_state=self.random_state
        )
        
        return {
            'X_pretrain': X_pretrain,
            'y_pretrain': y_pretrain,
            'X_labeled': X_labeled,
            'y_labeled': y_labeled,
            'X_unlabeled': X_unlabeled,
            'y_unlabeled': y_unlabeled,
        }

class PretrainedModel:
    def __init__(self, model_type: str = 'gradient_boosting',
                 random_state: int = 42, imperfection_level: float = 0.1):
        self.model_type = model_type
        self.random_state = random_state
        self.imperfection_level = imperfection_level
        self.model = None
        self.is_fitted = False
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        if self.model_type == 'gradient_boosting':
            n_estimators = max(10, int(100 * (1 - self.imperfection_level)))
            max_depth = max(2, int(6 * (1 - self.imperfection_level)))
            
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                random_state=self.random_state
            )
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        base_pred = self.model.predict(X)
        
        if self.imperfection_level > 0:
            bias = self.imperfection_level * np.mean(base_pred) * 0.1
            noise_std = self.imperfection_level * np.std(base_pred) * 0.2
            noise = np.random.normal(0, noise_std, len(base_pred))
            pred_with_error = base_pred + bias + noise
        else:
            pred_with_error = base_pred
            
        return pred_with_error 