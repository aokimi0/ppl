"""
预测驱动推断 (Prediction-Powered Inference, PPI) 核心实现
参考文献: Angelopoulos, A. N., et al. (2023). Prediction-powered inference. Science, 382(6671), 669-674.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from typing import Tuple, Union, Optional

class PPIInference:
    """
    预测驱动推断框架的实现
    
    核心思想：利用大量未标注数据的预测值和少量标注数据进行高效统计推断
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        初始化PPI推断器
        
        Args:
            alpha: 显著性水平，用于构建(1-alpha)置信区间
        """
        self.alpha = alpha
        self.beta_ppi = None
        self.se_ppi = None
        self.confidence_interval = None
        
    def estimate_mean_ppi(self, 
                         Y_labeled: np.ndarray, 
                         Yhat_labeled: np.ndarray, 
                         Yhat_unlabeled: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
        """
        使用PPI方法估计总体均值
        
        Args:
            Y_labeled: 标注数据的真实值 (n,)
            Yhat_labeled: 标注数据的预测值 (n,)
            Yhat_unlabeled: 未标注数据的预测值 (N,)
            
        Returns:
            tuple: (PPI估计值, 标准误, 置信区间)
        """
        n = len(Y_labeled)
        N = len(Yhat_unlabeled)
        
        # 经典估计量（仅使用标注数据）
        mu_classical = np.mean(Y_labeled)
        
        # 预测均值（未标注数据）
        mu_pred_unlabeled = np.mean(Yhat_unlabeled)
        
        # 预测均值（标注数据）
        mu_pred_labeled = np.mean(Yhat_labeled)
        
        # PPI校正项：消除预测偏差
        rectifier = mu_pred_labeled - np.mean(Y_labeled)
        
        # PPI估计量
        mu_ppi = mu_pred_unlabeled - rectifier
        
        # 方差估计
        # 标注数据的方差
        var_labeled = np.var(Y_labeled, ddof=1)
        
        # 预测误差的方差
        pred_error = Yhat_labeled - Y_labeled
        var_pred_error = np.var(pred_error, ddof=1)
        
        # PPI方差公式（渐近方差）
        var_ppi = var_labeled / n + var_pred_error / n - var_pred_error / N
        se_ppi = np.sqrt(var_ppi)
        
        # 置信区间
        t_critical = stats.t.ppf(1 - self.alpha/2, df=n-1)
        ci_lower = mu_ppi - t_critical * se_ppi
        ci_upper = mu_ppi + t_critical * se_ppi
        
        return mu_ppi, se_ppi, (ci_lower, ci_upper)
    
    def estimate_regression_ppi(self, 
                               X_labeled: np.ndarray, 
                               Y_labeled: np.ndarray,
                               Yhat_labeled: np.ndarray,
                               X_unlabeled: np.ndarray,
                               Yhat_unlabeled: np.ndarray) -> dict:
        """
        使用PPI方法进行回归系数推断
        
        Args:
            X_labeled: 标注数据的特征矩阵 (n, p)
            Y_labeled: 标注数据的真实值 (n,)
            Yhat_labeled: 标注数据的预测值 (n,)
            X_unlabeled: 未标注数据的特征矩阵 (N, p)
            Yhat_unlabeled: 未标注数据的预测值 (N,)
            
        Returns:
            dict: 包含PPI回归结果的字典
        """
        n = len(Y_labeled)
        N = len(Yhat_unlabeled)
        p = X_labeled.shape[1]
        
        # 经典线性回归（仅使用标注数据）
        reg_classical = LinearRegression().fit(X_labeled, Y_labeled)
        beta_classical = reg_classical.coef_
        
        # 基于预测值的回归
        X_combined = np.vstack([X_labeled, X_unlabeled])
        Y_combined = np.concatenate([Yhat_labeled, Yhat_unlabeled])
        reg_pred = LinearRegression().fit(X_combined, Y_combined)
        beta_pred = reg_pred.coef_
        
        # 预测值在标注数据上的回归
        reg_pred_labeled = LinearRegression().fit(X_labeled, Yhat_labeled)
        beta_pred_labeled = reg_pred_labeled.coef_
        
        # PPI校正
        rectifier = beta_pred_labeled - beta_classical
        beta_ppi = beta_pred - rectifier
        
        # 方差估计（简化版本）
        residuals_classical = Y_labeled - reg_classical.predict(X_labeled)
        mse_classical = np.mean(residuals_classical**2)
        
        # 协方差矩阵估计
        XTX_inv = np.linalg.inv(X_labeled.T @ X_labeled)
        var_beta_classical = mse_classical * XTX_inv
        
        # PPI方差（简化处理）
        pred_residuals = Yhat_labeled - Y_labeled
        mse_pred_error = np.mean(pred_residuals**2)
        
        # 近似PPI方差
        var_beta_ppi = var_beta_classical * (1 + mse_pred_error / mse_classical / N)
        se_beta_ppi = np.sqrt(np.diag(var_beta_ppi))
        
        # 置信区间
        t_critical = stats.t.ppf(1 - self.alpha/2, df=n-p-1)
        ci_lower = beta_ppi - t_critical * se_beta_ppi
        ci_upper = beta_ppi + t_critical * se_beta_ppi
        
        self.beta_ppi = beta_ppi
        self.se_ppi = se_beta_ppi
        self.confidence_interval = (ci_lower, ci_upper)
        
        return {
            'beta_classical': beta_classical,
            'beta_ppi': beta_ppi,
            'se_classical': np.sqrt(np.diag(var_beta_classical)),
            'se_ppi': se_beta_ppi,
            'ci_ppi': (ci_lower, ci_upper),
            'mse_classical': mse_classical,
            'mse_pred_error': mse_pred_error
        }

class BaselineComparison:
    """
    基准方法比较类：经典方法 vs 朴素ML方法 vs PPI方法
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def classical_inference(self, Y_labeled: np.ndarray) -> dict:
        """
        经典统计推断（仅使用标注数据）
        """
        n = len(Y_labeled)
        mean_est = np.mean(Y_labeled)
        se_est = np.std(Y_labeled, ddof=1) / np.sqrt(n)
        
        t_critical = stats.t.ppf(1 - self.alpha/2, df=n-1)
        ci_lower = mean_est - t_critical * se_est
        ci_upper = mean_est + t_critical * se_est
        
        return {
            'estimate': mean_est,
            'se': se_est,
            'ci': (ci_lower, ci_upper),
            'ci_width': ci_upper - ci_lower
        }
    
    def naive_ml_inference(self, Yhat_all: np.ndarray) -> dict:
        """
        朴素ML推断（直接使用所有预测值）
        """
        n = len(Yhat_all)
        mean_est = np.mean(Yhat_all)
        se_est = np.std(Yhat_all, ddof=1) / np.sqrt(n)
        
        t_critical = stats.norm.ppf(1 - self.alpha/2)  # 使用正态分布
        ci_lower = mean_est - t_critical * se_est
        ci_upper = mean_est + t_critical * se_est
        
        return {
            'estimate': mean_est,
            'se': se_est,
            'ci': (ci_lower, ci_upper),
            'ci_width': ci_upper - ci_lower
        }

def simulate_imperfect_predictions(Y_true: np.ndarray, 
                                 noise_level: float = 0.1,
                                 bias: float = 0.0) -> np.ndarray:
    """
    模拟不完美的预测值
    
    Args:
        Y_true: 真实值
        noise_level: 噪声水平（相对于Y的标准差）
        bias: 系统性偏差
        
    Returns:
        带有噪声和偏差的预测值
    """
    noise = np.random.normal(0, noise_level * np.std(Y_true), len(Y_true))
    Y_pred = Y_true + bias + noise
    return Y_pred

def evaluate_coverage_rate(true_param: float, 
                          confidence_intervals: list,
                          method_name: str = "Unknown") -> float:
    """
    评估置信区间的覆盖率
    
    Args:
        true_param: 真实参数值
        confidence_intervals: 置信区间列表 [(lower, upper), ...]
        method_name: 方法名称
        
    Returns:
        覆盖率
    """
    covered = sum(1 for ci in confidence_intervals if ci[0] <= true_param <= ci[1])
    coverage_rate = covered / len(confidence_intervals)
    
    print(f"{method_name} 覆盖率: {coverage_rate:.3f}")
    return coverage_rate 