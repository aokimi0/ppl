"""
可视化模块 - 生成高质量的学术图表
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
import matplotlib.patches as mpatches

# Optional: Use LaTeX for text rendering in plots for consistency with LaTeX reports
# Requires a working LaTeX distribution on your system.
USE_LATEX_IN_PLOTS = False # Set to False if you don't have LaTeX or encounter issues

if USE_LATEX_IN_PLOTS:
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif' # Or 'Computer Modern Roman', etc.
        # You can add more to the preamble, e.g., for specific math packages
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}'
        print("Successfully configured Matplotlib to use LaTeX for text rendering.")
    except Exception as e:
        print(f"Could not configure Matplotlib for LaTeX rendering (is LaTeX installed?): {e}")
        print("Falling back to default Matplotlib text rendering.")
        USE_LATEX_IN_PLOTS = False # Fallback if error
        plt.rcParams['text.usetex'] = False
        # Ensure fallback font family is set if the serif one was too specific for non-LaTeX
        plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']

# 设置matplotlib中文支持和美观样式
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 设置现代化配色方案
COLORS = {
    'classical': '#2E86AB',     # 深蓝色 - 经典方法
    'naive_ml': '#A23B72',      # 深紫红 - 朴素ML
    'ppi': '#F18F01',           # 橙色 - PPI方法
    'true_value': '#C73E1D',    # 红色 - 真实值
    'gray': '#6C757D',          # 灰色 - 辅助元素
    'light_gray': '#E9ECEF'     # 浅灰 - 背景
}

class AcademicVisualizer:
    """学术级可视化类"""
    
    def __init__(self, style='whitegrid', context='paper'):
        sns.set_style(style)
        sns.set_context(context, font_scale=1.2)
        self.fig_count = 0
        
    def plot_confidence_intervals_comparison(self, 
                                          results: Dict,
                                          true_value: float,
                                          save_path: str = None) -> plt.Figure:
        """
        绘制置信区间比较图
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        methods = ['Classical', 'Naive ML', 'PPI']
        colors = [COLORS['classical'], COLORS['naive_ml'], COLORS['ppi']]
        
        y_positions = np.arange(len(methods))
        
        for i, (method, color) in enumerate(zip(methods, colors)):
            method_key = method.lower().replace(' ', '_')
            if method_key in results:
                estimate = results[method_key]['estimate']
                ci_lower, ci_upper = results[method_key]['ci']
                
                # 绘制置信区间
                ax.errorbar(estimate, y_positions[i], 
                           xerr=[[estimate - ci_lower], [ci_upper - estimate]],
                           fmt='o', color=color, markersize=10, 
                           capsize=8, capthick=3, linewidth=3,
                           label=method)
                
                # 添加数值标签
                ax.text(estimate, y_positions[i] + 0.15, 
                       f'{estimate:.3f}\n[{ci_lower:.3f}, {ci_upper:.3f}]',
                       ha='center', va='bottom', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2))
        
        # 绘制真实值线
        ax.axvline(true_value, color=COLORS['true_value'], linestyle='--', 
                  linewidth=3, alpha=0.8, label=f'True Value: {true_value:.3f}')
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(methods)
        ax.set_xlabel('Estimated Value', fontsize=14, fontweight='bold')
        ax.set_title('Confidence Intervals Comparison\nPrediction-Powered Inference vs Baseline Methods', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 美化边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('#333333')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def plot_coverage_rate_analysis(self, 
                                   coverage_data: Dict,
                                   sample_sizes: List[int],
                                   save_path: str = None) -> plt.Figure:
        """
        绘制覆盖率分析图
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：覆盖率 vs 样本量
        methods = ['classical', 'naive_ml', 'ppi']
        method_names = ['Classical', 'Naive ML', 'PPI']
        colors = [COLORS['classical'], COLORS['naive_ml'], COLORS['ppi']]
        
        for method, name, color in zip(methods, method_names, colors):
            if method in coverage_data:
                coverage_rates = coverage_data[method]
                ax1.plot(sample_sizes, coverage_rates, 'o-', 
                        color=color, linewidth=3, markersize=8, 
                        label=name, alpha=0.8)
        
        ax1.axhline(0.95, color=COLORS['true_value'], linestyle='--', 
                   linewidth=2, alpha=0.7, label='Target Coverage (95%)')
        ax1.set_xlabel('Labeled Sample Size', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Coverage Rate', fontsize=12, fontweight='bold')
        ax1.set_title('Coverage Rate vs Sample Size', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.8, 1.0)
        
        # 右图：置信区间宽度比较
        if 'ci_widths' in coverage_data:
            ci_data = coverage_data['ci_widths']
            x = np.arange(len(method_names))
            width = 0.25
            
            for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
                if method in ci_data:
                    heights = ci_data[method]
                    ax2.bar(x[i], np.mean(heights), width, 
                           color=color, alpha=0.7, label=name,
                           yerr=np.std(heights), capsize=5)
            
            ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Average CI Width', fontsize=12, fontweight='bold')
            ax2.set_title('Confidence Interval Width Comparison', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(method_names)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def plot_bias_variance_analysis(self, 
                                   simulation_results: Dict,
                                   save_path: str = None) -> plt.Figure:
        """
        绘制偏差-方差分析图
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        methods = ['classical', 'naive_ml', 'ppi']
        method_names = ['Classical', 'Naive ML', 'PPI']
        colors = [COLORS['classical'], COLORS['naive_ml'], COLORS['ppi']]
        
        # 1. 估计值分布
        for method, name, color in zip(methods, method_names, colors):
            if f'{method}_estimates' in simulation_results:
                estimates = simulation_results[f'{method}_estimates']
                ax1.hist(estimates, bins=30, alpha=0.6, color=color, 
                        label=name, density=True)
        
        true_value = simulation_results.get('true_value', 0)
        ax1.axvline(true_value, color=COLORS['true_value'], linestyle='--', 
                   linewidth=2, label=f'True Value: {true_value:.3f}')
        ax1.set_xlabel('Estimated Value', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Distribution of Estimates', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 偏差分析
        bias_data = []
        for method, name in zip(methods, method_names):
            if f'{method}_estimates' in simulation_results:
                estimates = simulation_results[f'{method}_estimates']
                bias = np.mean(estimates) - true_value
                bias_data.append(bias)
        
        bars = ax2.bar(method_names, bias_data, color=colors, alpha=0.7)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_ylabel('Bias', fontsize=12)
        ax2.set_title('Bias Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, bias in zip(bars, bias_data):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001 * np.sign(height),
                    f'{bias:.4f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 3. 方差分析
        variance_data = []
        for method, name in zip(methods, method_names):
            if f'{method}_estimates' in simulation_results:
                estimates = simulation_results[f'{method}_estimates']
                variance = np.var(estimates)
                variance_data.append(variance)
        
        bars = ax3.bar(method_names, variance_data, color=colors, alpha=0.7)
        ax3.set_ylabel('Variance', fontsize=12)
        ax3.set_title('Variance Comparison', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        for bar, var in zip(bars, variance_data):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{var:.4f}', ha='center', va='bottom')
        
        # 4. MSE分析
        mse_data = []
        for method, name in zip(methods, method_names):
            if f'{method}_estimates' in simulation_results:
                estimates = simulation_results[f'{method}_estimates']
                mse = np.mean((estimates - true_value)**2)
                mse_data.append(mse)
        
        bars = ax4.bar(method_names, mse_data, color=colors, alpha=0.7)
        ax4.set_ylabel('Mean Squared Error', fontsize=12)
        ax4.set_title('MSE Comparison', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        for bar, mse in zip(bars, mse_data):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{mse:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def plot_data_overview(self, 
                          X: np.ndarray, 
                          y: np.ndarray,
                          feature_names: List[str],
                          split_info: Dict,
                          save_path: str = None) -> plt.Figure:
        """
        绘制数据概览图
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 数据分割饼图
        ax1 = fig.add_subplot(gs[0, 0])
        sizes = [len(split_info['X_pretrain']), 
                len(split_info['X_labeled']), 
                len(split_info['X_unlabeled'])]
        labels = ['Pretrain', 'Labeled', 'Unlabeled']
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax1.set_title('Data Split Overview', fontsize=14, fontweight='bold')
        
        # 2. 目标变量分布
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(y, bins=30, color=COLORS['ppi'], alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Cognitive Change Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3-6. 主要特征分布（选择前4个特征）
        for i in range(min(4, len(feature_names))):
            ax = fig.add_subplot(gs[0, 2 + i//2]) if i < 2 else fig.add_subplot(gs[1, (i-2)])
            feature_data = X[:, i]
            ax.hist(feature_data, bins=20, color=colors[i % len(colors)], 
                   alpha=0.7, edgecolor='black')
            ax.set_xlabel(feature_names[i])
            ax.set_ylabel('Frequency')
            ax.set_title(f'{feature_names[i]} Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 7. 特征相关性热力图
        ax7 = fig.add_subplot(gs[1:, 2:])
        df = pd.DataFrame(X, columns=feature_names)
        correlation_matrix = df.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', ax=ax7, 
                   cbar_kws={'label': 'Correlation Coefficient'})
        ax7.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        # 8. 样本量对比柱状图
        ax8 = fig.add_subplot(gs[2, 0:2])
        dataset_names = ['Pretrain', 'Labeled', 'Unlabeled']
        dataset_sizes = sizes
        bars = ax8.bar(dataset_names, dataset_sizes, color=colors, alpha=0.8)
        ax8.set_ylabel('Sample Size')
        ax8.set_title('Dataset Size Comparison', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, size in zip(bars, dataset_sizes):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + max(dataset_sizes) * 0.01,
                    f'{size}', ha='center', va='bottom', fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def create_method_comparison_table(self, results: Dict) -> pd.DataFrame:
        """
        创建方法比较表格
        """
        comparison_data = []
        
        methods = ['classical', 'naive_ml', 'ppi']
        method_names = ['Classical (Labeled Only)', 'Naive ML (All Predictions)', 'PPI (Corrected)']
        
        for method, name in zip(methods, method_names):
            if method in results:
                result = results[method]
                comparison_data.append({
                    'Method': name,
                    'Estimate': f"{result['estimate']:.4f}",
                    'Standard Error': f"{result['se']:.4f}",
                    'CI Lower': f"{result['ci'][0]:.4f}",
                    'CI Upper': f"{result['ci'][1]:.4f}",
                    'CI Width': f"{result['ci_width']:.4f}"
                })
        
        return pd.DataFrame(comparison_data)
    
    def plot_real_vs_predicted(self,
                               y_true: np.ndarray,
                               y_predicted: np.ndarray,
                               title: str = '真实值 vs. 预测值',
                               save_path: str = None) -> plt.Figure:
        """
        绘制真实值与预测值的散点图。

        参数:
            y_true (np.ndarray): 真实标签/值。
            y_predicted (np.ndarray): 模型预测的标签/值。
            title (str): 图表标题。
            save_path (str, optional): 保存图像的路径。如果为None，则不保存。

        返回:
            plt.Figure: Matplotlib Figure 对象。
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_true, y_predicted, alpha=0.6, edgecolors='k', color=COLORS['ppi'], label='预测点')
        
        # 添加 y=x 参考线
        # 获取当前x和y轴的限制，以确保参考线覆盖整个数据范围
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        
        # 计算一个合适的全局限制，使参考线美观
        min_val = np.min(np.concatenate([y_true, y_predicted]))
        max_val = np.max(np.concatenate([y_true, y_predicted]))
        plot_lims = [min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val)]

        ax.plot(plot_lims, plot_lims, color=COLORS['true_value'], linestyle='--', linewidth=2, label='理想情况 (真实值 = 预测值)')
        ax.set_xlim(plot_lims)
        ax.set_ylim(plot_lims)
        
        ax.set_xlabel('真实值', fontsize=14, fontweight='bold')
        ax.set_ylabel('预测值', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # 美化边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('#333333')
            
        plt.tight_layout()
        
        if save_path:
            self.fig_count += 1 # Ensure fig_count is incremented if used for unique names
            actual_save_path = save_path.replace('.png', f'_{self.fig_count}.png') if '{count}' in save_path else save_path
            plt.savefig(actual_save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"图表已保存到 {actual_save_path}")
            
        return fig 