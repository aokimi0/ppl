#!/usr/bin/env python3
"""
预测驱动推断(PPI)实验主程序
Prediction-Powered Inference Experiment Main Script

作者：数据科学导论实验小组
日期：2025年1月
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from src.experiments import ExperimentRunner
from src.visualization import AcademicVisualizer
import matplotlib.pyplot as plt

# 忽略一些不重要的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def setup_directories():
    """创建必要的目录"""
    directories = ['fig', 'data', 'results']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"目录 '{dir_name}' 已准备就绪")

def print_banner():
    """打印程序横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║        预测驱动推断 (Prediction-Powered Inference)          ║
    ║                     实验评估系统                           ║
    ║                                                            ║
    ║  基于预训练模型和无标注数据的统计推断框架实验               ║
    ║  参考: Angelopoulos et al. (2023) Science                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_experiment_info():
    """打印实验信息"""
    info = """
    实验概述:
    ========
    
    目标: 比较预测驱动推断(PPI)与传统统计推断方法的性能
    
    方法对比:
    1. 经典方法 (Classical)     - 仅使用少量标注数据
    2. 朴素ML方法 (Naive ML)    - 直接使用所有预测值
    3. PPI方法 (PPI)           - 校正偏差的预测驱动推断
    
    评估指标:
    - 置信区间覆盖率
    - 置信区间宽度  
    - 估计偏差
    - 统计效率
    
    数据模拟:
    - 模拟阿尔茨海默病研究数据 (类似ADNI数据集)
    - 包含年龄、教育、APOE4基因型等特征
    - 目标变量: 认知变化评分
    """
    print(info)

def main():
    """主函数"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行PPI实验评估')
    parser.add_argument('--quick', action='store_true', 
                       help='快速运行模式(减少计算量)')
    parser.add_argument('--output-dir', default='fig',
                       help='输出目录 (默认: fig)')
    parser.add_argument('--no-plots', action='store_true',
                       help='不生成图表')
    
    args = parser.parse_args()
    
    # 打印横幅和信息
    print_banner()
    print_experiment_info()
    
    # 设置目录
    setup_directories()
    
    try:
        print("\n" + "="*80)
        print("开始实验执行...")
        print("="*80)
        
        # 创建实验运行器
        runner = ExperimentRunner(output_dir=args.output_dir)
        
        # 运行实验
        print("\n1. 运行主要PPI对比实验...")
        results = runner.run_complete_evaluation()
        
        # 打印结果摘要
        print_results_summary(results)
        
        # 生成报告数据
        print("\n2. 生成实验报告数据...")
        generate_report_data(results, args.output_dir)
        
        print("\n" + "="*80)
        print("实验完成!")
        print(f"结果图表保存在: {args.output_dir}/")
        print(f"报告数据保存在: results/")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n实验被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def print_results_summary(results):
    """打印结果摘要"""
    main_result = results['main_experiment']
    
    print("\n" + "-"*60)
    print("实验结果摘要")
    print("-"*60)
    
    # 基本信息
    print(f"数据集信息:")
    print(f"  - 标注集大小: {main_result['data_info']['n_labeled']}")
    print(f"  - 未标注集大小: {main_result['data_info']['n_unlabeled']}")
    print(f"  - 预测模型R²: {main_result['model_performance']['r2']:.3f}")
    
    print(f"\n真实值: {main_result['true_value']:.4f}")
    
    # 方法比较
    print(f"\n方法比较:")
    print(f"{'方法':<15} {'估计值':<10} {'标准误':<10} {'置信区间':<20} {'偏差':<10} {'覆盖':<6}")
    print("-" * 80)
    
    methods = [
        ('Classical', 'classical'),
        ('Naive ML', 'naive_ml'), 
        ('PPI', 'ppi')
    ]
    
    for name, key in methods:
        result = main_result[key]
        metrics = main_result[f'{key}_metrics']
        
        ci_str = f"[{result['ci'][0]:.3f}, {result['ci'][1]:.3f}]"
        coverage = "✓" if metrics['coverage'] else "✗"
        
        print(f"{name:<15} {result['estimate']:<10.4f} {result['se']:<10.4f} "
              f"{ci_str:<20} {metrics['bias']:<10.4f} {coverage:<6}")
    
    # 性能提升
    classical_width = main_result['classical']['ci_width']
    ppi_width = main_result['ppi']['ci_width']
    improvement = (classical_width - ppi_width) / classical_width * 100
    
    print(f"\nPPI性能提升:")
    print(f"  - 置信区间宽度减少: {improvement:.1f}%")
    print(f"  - 相对效率提升: {classical_width/ppi_width:.2f}x")

def generate_report_data(results, output_dir):
    """生成报告所需的数据文件"""
    import json
    import pandas as pd
    
    # 确保results目录存在
    Path('results').mkdir(exist_ok=True)
    
    main_result = results['main_experiment']
    
    # 1. 保存主要结果为JSON
    # 需要处理numpy数组
    def convert_numpy(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # 简化结果用于JSON保存
    json_result = {
        'classical': main_result['classical'],
        'naive_ml': main_result['naive_ml'],
        'ppi': main_result['ppi'],
        'true_value': main_result['true_value'],
        'model_performance': main_result['model_performance'],
        'data_info': main_result['data_info']
    }
    
    # 添加指标
    for method in ['classical', 'naive_ml', 'ppi']:
        json_result[f'{method}_metrics'] = main_result[f'{method}_metrics']
    
    json_result = convert_numpy(json_result)
    
    with open('results/experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    
    # 2. 创建方法比较表
    visualizer = AcademicVisualizer()
    comparison_table = visualizer.create_method_comparison_table(main_result)
    comparison_table.to_csv('results/method_comparison.csv', index=False)
    
    print("实验数据已保存:")
    print("  - results/experiment_results.json")
    print("  - results/method_comparison.csv")

if __name__ == "__main__":
    main() 