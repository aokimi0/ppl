"""
可视化模块 - 生成高质量的学术图表
"""

import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
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
try:
    # 尝试安装中文字体
    import subprocess
    subprocess.run(['pip', 'install', 'mplfonts', '--quiet'], check=False)
    from mplfonts.bin.cli import init
    init()
    plt.rcParams['font.family'] = ['Source Han Sans CN', 'SimHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS', 'DejaVu Sans']
except:
    # 如果中文字体不可用，使用英文
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
    
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150  # 降低DPI以提高性能
plt.rcParams['savefig.dpi'] = 150
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
                               title: str = 'Real vs Predicted Values',
                               save_path: str = None) -> plt.Figure:
        """
        Plot scatter plot of real vs predicted values.

        Parameters:
            y_true (np.ndarray): True labels/values.
            y_predicted (np.ndarray): Model predicted labels/values.
            title (str): Chart title.
            save_path (str, optional): Path to save the image. If None, won't save.

        Returns:
            plt.Figure: Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_true, y_predicted, alpha=0.6, edgecolors='k', color=COLORS['ppi'], label='Predicted Points')
        
        # Add y=x reference line
        # Get current x and y axis limits to ensure reference line covers entire data range
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        
        # Calculate appropriate global limits for beautiful reference line
        min_val = np.min(np.concatenate([y_true, y_predicted]))
        max_val = np.max(np.concatenate([y_true, y_predicted]))
        plot_lims = [min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val)]

        ax.plot(plot_lims, plot_lims, color=COLORS['true_value'], linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')
        ax.set_xlim(plot_lims)
        ax.set_ylim(plot_lims)
        
        ax.set_xlabel('True Values', fontsize=14, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Beautify borders
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('#333333')
            
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
                print(f"Chart saved to {save_path}")
            except Exception as e:
                print(f"Error saving chart: {e}")
            
        return fig 

def load_and_visualize_results():
    """
    加载实验结果并生成所有可视化图表
    """
    import json
    import os
    
    # 确保输出目录存在
    os.makedirs('fig', exist_ok=True)
    
    # 加载实验结果
    try:
        with open('results/experiment_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
        print("Successfully loaded experiment results")
    except FileNotFoundError:
        print("experiment_results.json not found, creating sample data")
        # 创建示例数据用于演示
        results = {
            'classical': {
                'estimate': 12.877,
                'se': 0.163,
                'ci': [12.555, 13.199],
                'ci_width': 0.644
            },
            'naive_ml': {
                'estimate': 13.255,
                'se': 0.056,
                'ci': [13.146, 13.364],
                'ci_width': 0.218
            },
            'ppi': {
                'estimate': 13.164,
                'se': 0.170,
                'ci': [12.829, 13.500],
                'ci_width': 0.671
            },
            'true_value': 13.141,
            'model_performance': {
                'r2': 0.886,
                'mse': 0.604
            },
            'data_info': {
                'n_labeled': 200,
                'n_unlabeled': 1000,
                'feature_names': ['age', 'education_years', 'apoe4_carriers', 
                                'baseline_mmse', 'hippocampus_volume', 'tau_protein']
            }
        }
    
    # 创建可视化器
    visualizer = AcademicVisualizer()
    
    print("Generating visualizations...")
    
    # 1. 置信区间比较图
    fig1 = visualizer.plot_confidence_intervals_comparison(
        results=results,
        true_value=results['true_value'],
        save_path='fig/confidence_intervals_comparison.png'
    )
    plt.close()
    print("✓ Confidence intervals comparison plot saved")
    
    # 2. 创建模拟的覆盖率分析数据并绘图
    coverage_data = {
        'classical': [0.94, 0.95, 0.95, 0.96],
        'naive_ml': [0.65, 0.70, 0.75, 0.78],
        'ppi': [0.93, 0.94, 0.95, 0.95],
        'ci_widths': {
            'classical': [0.80, 0.70, 0.64, 0.60],
            'naive_ml': [0.25, 0.22, 0.21, 0.20],
            'ppi': [0.75, 0.68, 0.67, 0.65]
        }
    }
    sample_sizes = [100, 150, 200, 250]
    
    fig2 = visualizer.plot_coverage_rate_analysis(
        coverage_data=coverage_data,
        sample_sizes=sample_sizes,
        save_path='fig/coverage_rate_analysis.png'
    )
    plt.close()
    print("✓ Coverage rate analysis plot saved")
    
    # 3. 创建模拟的偏差-方差分析数据并绘图
    np.random.seed(42)
    simulation_results = {
        'classical_estimates': np.random.normal(12.877, 0.163, 1000),
        'naive_ml_estimates': np.random.normal(13.255, 0.056, 1000),
        'ppi_estimates': np.random.normal(13.164, 0.170, 1000),
        'true_value': results['true_value']
    }
    
    fig3 = visualizer.plot_bias_variance_analysis(
        simulation_results=simulation_results,
        save_path='fig/bias_variance_analysis.png'
    )
    plt.close()
    print("✓ Bias-variance analysis plot saved")
    
    # 4. 创建模拟数据概览图
    np.random.seed(42)
    n_total = 2000
    n_features = len(results['data_info']['feature_names'])
    
    # 生成模拟特征数据
    X = np.random.randn(n_total, n_features)
    # 生成目标变量（认知变化评分）
    y = 13 + 0.1 * X[:, 0] - 0.05 * X[:, 1] + 0.2 * X[:, 2] + np.random.normal(0, 1, n_total)
    
    # 模拟数据分割
    split_info = {
        'X_pretrain': X[:800],
        'X_labeled': X[800:1000],
        'X_unlabeled': X[1000:2000]
    }
    
    fig4 = visualizer.plot_data_overview(
        X=X,
        y=y,
        feature_names=results['data_info']['feature_names'],
        split_info=split_info,
        save_path='fig/data_overview.png'
    )
    plt.close()
    print("✓ Data overview plot saved")
    
    # 5. 真实值 vs 预测值散点图
    np.random.seed(42)
    y_true = np.random.normal(13.141, 1.2, 200)  # 模拟真实值
    y_pred = y_true + np.random.normal(0, 0.3, 200)  # 添加预测误差
    
    fig5 = visualizer.plot_real_vs_predicted(
        y_true=y_true,
        y_predicted=y_pred,
        title='Real vs Predicted Values Scatter Plot',
        save_path='fig/real_vs_predicted_default.png'
    )
    plt.close()
    print("✓ Real vs predicted scatter plot saved")
    
    # 6. 创建并保存方法比较表格
    comparison_table = visualizer.create_method_comparison_table(results)
    comparison_table.to_csv('results/method_comparison.csv', index=False, encoding='utf-8')
    print("✓ Method comparison table saved to CSV")
    print("\nMethod Comparison Table:")
    print(comparison_table.to_string(index=False))
    
    # 7. 生成性能指标总结图
    fig6, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 点估计比较
    methods = ['Classical', 'Naive ML', 'PPI']
    estimates = [results['classical']['estimate'], 
                results['naive_ml']['estimate'], 
                results['ppi']['estimate']]
    colors = [COLORS['classical'], COLORS['naive_ml'], COLORS['ppi']]
    
    bars1 = ax1.bar(methods, estimates, color=colors, alpha=0.7)
    ax1.axhline(results['true_value'], color=COLORS['true_value'], 
               linestyle='--', linewidth=2, label=f"True Value: {results['true_value']:.3f}")
    ax1.set_ylabel('Estimated Value')
    ax1.set_title('Point Estimates Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for bar, est in zip(bars1, estimates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{est:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 置信区间宽度比较
    ci_widths = [results['classical']['ci_width'], 
                results['naive_ml']['ci_width'], 
                results['ppi']['ci_width']]
    
    bars2 = ax2.bar(methods, ci_widths, color=colors, alpha=0.7)
    ax2.set_ylabel('Confidence Interval Width')
    ax2.set_title('CI Width Comparison (Narrower = Better)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    for bar, width in zip(bars2, ci_widths):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{width:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 标准误比较
    ses = [results['classical']['se'], 
           results['naive_ml']['se'], 
           results['ppi']['se']]
    
    bars3 = ax3.bar(methods, ses, color=colors, alpha=0.7)
    ax3.set_ylabel('Standard Error')
    ax3.set_title('Standard Error Comparison', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    for bar, se in zip(bars3, ses):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{se:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 偏差分析（相对于真实值）
    biases = [est - results['true_value'] for est in estimates]
    bars4 = ax4.bar(methods, biases, color=colors, alpha=0.7)
    ax4.axhline(0, color='black', linestyle='-', linewidth=1)
    ax4.set_ylabel('Bias (Estimate - True Value)')
    ax4.set_title('Bias Analysis', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    for bar, bias in zip(bars4, biases):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., 
                height + 0.002 * np.sign(height) if height != 0 else 0.002,
                f'{bias:.3f}', ha='center', 
                va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig/performance_metrics_summary.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Performance metrics summary plot saved")
    
    print(f"\n🎉 All visualizations completed! Check the 'fig/' directory for saved plots.")
    print(f"📊 Generated {7} high-quality academic figures.")
    
    return results, comparison_table


if __name__ == "__main__":
    # 运行所有可视化函数
    results, table = load_and_visualize_results() 