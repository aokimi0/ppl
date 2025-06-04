"""
实验运行模块 - 执行PPI vs 基准方法的比较实验
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
from sklearn.metrics import mean_squared_error, r2_score

from .ppi_inference import PPIInference, BaselineComparison
from .data_generator import SyntheticDataGenerator, DataSplitter, PretrainedModel
from .visualization import AcademicVisualizer

class PPIExperiment:
    """PPI实验运行器"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = {}
        
    def run_single_experiment(self, n_total: int = 2000, n_labeled: int = 200,
                            n_unlabeled: int = 1000, n_pretrain: int = 500,
                            imperfection_level: float = 0.15,
                            simulation_seed: Optional[int] = None) -> Dict:
        """运行单次PPI实验"""
        # Use simulation_seed if provided, otherwise use instance's random_state
        current_random_state = simulation_seed if simulation_seed is not None else self.random_state
        
        if simulation_seed is not None:
            print(f"  Running single experiment iteration with seed: {current_random_state}, n_labeled={n_labeled}")
        else:
            print(f"开始实验: n_labeled={n_labeled}, imperfection={imperfection_level:.2f}, state={current_random_state}")
        
        # 生成数据
        data_generator = SyntheticDataGenerator(random_state=current_random_state)
        X, y, feature_names = data_generator.generate_medical_data(n_total=n_total, n_features=6)
        
        # 分割数据
        splitter = DataSplitter(random_state=current_random_state)
        split_data = splitter.split_for_ppi(X, y, n_labeled, n_unlabeled, n_pretrain)
        
        # 训练预测模型
        pretrained_model = PretrainedModel(
            imperfection_level=imperfection_level,
            random_state=current_random_state
        )
        pretrained_model.fit(split_data['X_pretrain'], split_data['y_pretrain'])
        
        # 生成预测值
        yhat_labeled = pretrained_model.predict(split_data['X_labeled'])
        yhat_unlabeled = pretrained_model.predict(split_data['X_unlabeled'])
        
        # 评估预测模型
        pred_r2 = r2_score(split_data['y_labeled'], yhat_labeled)
        pred_mse = mean_squared_error(split_data['y_labeled'], yhat_labeled)
        # Suppress print for simulation runs unless it's a main run
        if simulation_seed is None:
             print(f"预测模型性能: R² = {pred_r2:.3f}, MSE = {pred_mse:.3f}")
        
        # 真实均值
        true_mean = np.mean(split_data['y_unlabeled']) # True mean from current split's unlabeled ground truth
        
        # 三种方法
        baseline = BaselineComparison(alpha=0.05)
        
        # 经典方法
        classical_result = baseline.classical_inference(split_data['y_labeled'])
        
        # 朴素ML方法
        all_predictions = np.concatenate([yhat_labeled, yhat_unlabeled])
        naive_ml_result = baseline.naive_ml_inference(all_predictions)
        
        # PPI方法
        ppi = PPIInference(alpha=0.05)
        ppi_estimate, ppi_se, ppi_ci = ppi.estimate_mean_ppi(
            split_data['y_labeled'], yhat_labeled, yhat_unlabeled
        )
        
        ppi_result = {
            'estimate': ppi_estimate,
            'se': ppi_se,
            'ci': ppi_ci,
            'ci_width': ppi_ci[1] - ppi_ci[0]
        }
        
        # 整理结果
        results = {
            'classical': classical_result,
            'naive_ml': naive_ml_result,
            'ppi': ppi_result,
            'true_value': true_mean,
            'model_performance': {'r2': pred_r2, 'mse': pred_mse},
            'data_info': {'n_labeled': n_labeled, 'n_unlabeled': n_unlabeled,
                         'feature_names': feature_names},
            'split_data': split_data,
            'X': X, 'y': y
        }
        
        # 计算指标
        for method_name, method_result in [('classical', classical_result), 
                                         ('naive_ml', naive_ml_result), 
                                         ('ppi', ppi_result)]:
            bias = method_result['estimate'] - true_mean
            coverage = (method_result['ci'][0] <= true_mean <= method_result['ci'][1])
            
            results[f'{method_name}_metrics'] = {
                'bias': bias,
                'coverage': coverage,
                'ci_width': method_result['ci_width']
            }
        
        return results

    def run_coverage_simulation(self, 
                                n_labeled_sizes: List[int], 
                                n_simulations: int, 
                                n_total: int = 2000, 
                                n_unlabeled_base: int = 1000, # Base N_unlabeled, n_labeled will change
                                n_pretrain: int = 500,
                                imperfection_level: float = 0.15,
                                alpha: float = 0.05) -> Dict:
        """运行覆盖率模拟实验"""
        print(f"  Starting coverage simulation: {n_simulations} runs per n_labeled size.")
        
        all_results = {
            'n_labeled_sizes': n_labeled_sizes,
            'classical_coverage': np.zeros(len(n_labeled_sizes)),
            'naive_ml_coverage': np.zeros(len(n_labeled_sizes)),
            'ppi_coverage': np.zeros(len(n_labeled_sizes)),
            'classical_avg_width': np.zeros(len(n_labeled_sizes)),
            'naive_ml_avg_width': np.zeros(len(n_labeled_sizes)),
            'ppi_avg_width': np.zeros(len(n_labeled_sizes)),
        }

        for i, n_labeled_current in enumerate(n_labeled_sizes):
            # Adjust n_unlabeled if necessary, e.g., n_unlabeled = n_total - n_labeled - n_pretrain
            # For simplicity, let's keep n_unlabeled_base fixed unless n_labeled makes it problematic
            current_n_unlabeled = n_unlabeled_base 
            if n_labeled_current + n_unlabeled_base + n_pretrain > n_total:
                # Adjust n_unlabeled if total is exceeded, ensuring positive pretrain and unlabeled
                current_n_unlabeled = max(50, n_total - n_labeled_current - n_pretrain) 
                if current_n_unlabeled + n_labeled_current + n_pretrain > n_total : # if still not fitting, adjust pretrain
                    current_n_pretrain = max(50, n_total - n_labeled_current - current_n_unlabeled)
                else:
                    current_n_pretrain = n_pretrain
            else:
                 current_n_pretrain = n_pretrain


            print(f"    Simulating for n_labeled = {n_labeled_current} (N_unlabeled={current_n_unlabeled}, N_pretrain={current_n_pretrain})")

            sim_classical_coverage = []
            sim_naive_ml_coverage = []
            sim_ppi_coverage = []
            sim_classical_widths = []
            sim_naive_ml_widths = []
            sim_ppi_widths = []

            for sim_idx in range(n_simulations):
                # Ensure each simulation is independent by providing a unique seed
                simulation_seed = self.random_state + sim_idx * len(n_labeled_sizes) + i 

                # Pass all necessary parameters for a single run
                single_run_res = self.run_single_experiment(
                    n_total=n_total, 
                    n_labeled=n_labeled_current,
                    n_unlabeled=current_n_unlabeled, 
                    n_pretrain=current_n_pretrain,
                    imperfection_level=imperfection_level,
                    simulation_seed=simulation_seed # Pass the varying seed
                )
                
                sim_classical_coverage.append(single_run_res['classical_metrics']['coverage'])
                sim_naive_ml_coverage.append(single_run_res['naive_ml_metrics']['coverage'])
                sim_ppi_coverage.append(single_run_res['ppi_metrics']['coverage'])
                
                sim_classical_widths.append(single_run_res['classical_metrics']['ci_width'])
                sim_naive_ml_widths.append(single_run_res['naive_ml_metrics']['ci_width'])
                sim_ppi_widths.append(single_run_res['ppi_metrics']['ci_width'])

            all_results['classical_coverage'][i] = np.mean(sim_classical_coverage)
            all_results['naive_ml_coverage'][i] = np.mean(sim_naive_ml_coverage)
            all_results['ppi_coverage'][i] = np.mean(sim_ppi_coverage)
            all_results['classical_avg_width'][i] = np.mean(sim_classical_widths)
            all_results['naive_ml_avg_width'][i] = np.mean(sim_naive_ml_widths)
            all_results['ppi_avg_width'][i] = np.mean(sim_ppi_widths)
            
        return all_results

    def run_bias_variance_analysis(self, 
                                   n_simulations: int, 
                                   n_total: int = 2000, 
                                   n_labeled: int = 200,
                                   n_unlabeled: int = 1000, 
                                   n_pretrain: int = 500,
                                   imperfection_level: float = 0.15) -> Dict:
        """运行偏差-方差分析模拟"""
        print(f"  Starting bias-variance analysis: {n_simulations} runs.")
        
        classical_estimates = []
        naive_ml_estimates = []
        ppi_estimates = []
        true_values = [] # To confirm consistency or average if it varies due to sampling

        for sim_idx in range(n_simulations):
            simulation_seed = self.random_state + sim_idx # Ensure independence

            single_run_res = self.run_single_experiment(
                n_total=n_total, 
                n_labeled=n_labeled,
                n_unlabeled=n_unlabeled, 
                n_pretrain=n_pretrain,
                imperfection_level=imperfection_level,
                simulation_seed=simulation_seed
            )
            
            classical_estimates.append(single_run_res['classical']['estimate'])
            naive_ml_estimates.append(single_run_res['naive_ml']['estimate'])
            ppi_estimates.append(single_run_res['ppi']['estimate'])
            true_values.append(single_run_res['true_value'])
            
        # It's generally assumed true_value is fixed or we average it if generated stochastically per run
        # For this setup, true_value comes from np.mean(split_data['y_unlabeled']), so it can vary slightly
        # if X,y generation is part of each simulation run.
        # Let's report the average true value observed across simulations.
        avg_true_value = np.mean(true_values) if true_values else None

        return {
            'true_value': avg_true_value, # Or the first true_value if parameters are such that it's fixed
            'classical_estimates': np.array(classical_estimates),
            'naive_ml_estimates': np.array(naive_ml_estimates),
            'ppi_estimates': np.array(ppi_estimates)
        }

class ExperimentRunner:
    """实验运行管理器"""
    
    def __init__(self, output_dir: str = "fig"):
        self.output_dir = output_dir
        self.visualizer = AcademicVisualizer()
        self.experiment = PPIExperiment()
        self.results_summary = {} # Initialize results_summary
        
    def run_complete_evaluation(self) -> Dict:
        """运行完整评估"""
        print("开始完整的PPI评估实验")
        
        # 主要实验 (单次运行详细结果)
        print("  Running single detailed experiment...")
        main_result = self.experiment.run_single_experiment(
            n_labeled=200, n_unlabeled=1000, n_pretrain=500, imperfection_level=0.15
        )
        self.results_summary['main_experiment'] = main_result

        # 覆盖率分析模拟
        print("  Running coverage simulations...")
        if 'coverage_simulation_data' not in self.results_summary.get('main_experiment', {}):
            coverage_sim_data = self.experiment.run_coverage_simulation(
                n_labeled_sizes=[50, 100, 200, 400], 
                n_simulations=100, # For speed; recommend 200-500+ for smoother plots
                n_total=2000, 
                n_unlabeled_base=1000, # Base N_unlabeled
                n_pretrain=500,
                imperfection_level=0.15
            )
            self.results_summary['main_experiment']['coverage_simulation_data'] = coverage_sim_data
        else:
            print("    Skipping coverage_simulation, data already present.")
            # coverage_sim_data = self.results_summary['main_experiment']['coverage_simulation_data'] # No need to reassign

        # 偏差-方差分析模拟
        print("  Running bias-variance simulations...")
        if 'bias_variance_data' not in self.results_summary.get('main_experiment', {}):
            bias_var_data = self.experiment.run_bias_variance_analysis(
                n_simulations=200, # For speed; recommend 500-1000+ for smoother plots
                n_total=2000,
                n_labeled=200,
                n_unlabeled=1000,
                n_pretrain=500,
                imperfection_level=0.15
            )
            self.results_summary['main_experiment']['bias_variance_data'] = bias_var_data
        else:
            print("    Skipping bias_variance_analysis, data already present.")
            # bias_var_data = self.results_summary['main_experiment']['bias_variance_data'] # No need to reassign

        # 生成图表 (包括新的模拟图表)
        self._generate_plots(self.results_summary['main_experiment'])
        
        return self.results_summary # Return the full summary
    
    def _generate_plots(self, results_data: Dict):
        """生成图表"""
        print(f"  Generating plots in {self.output_dir}...")
        # 置信区间比较 (来自单次运行)
        self.visualizer.plot_confidence_intervals_comparison(
            results_data, results_data['true_value'],
            save_path=f"{self.output_dir}/confidence_intervals_comparison.png"
        )
        print(f"    - Confidence intervals comparison plot saved.")
        
        # 数据概览 (来自单次运行)
        if 'X' in results_data and 'y' in results_data and 'data_info' in results_data and 'split_data' in results_data:
            self.visualizer.plot_data_overview(
                results_data['X'], results_data['y'],
                results_data['data_info']['feature_names'],
                results_data['split_data'], # This was a dict, plot_data_overview expects split_info Dict
                save_path=f"{self.output_dir}/data_overview.png"
            )
            print(f"    - Data overview plot saved.")
        else:
            print("    - Skipping data overview plot due to missing data.")

        # 覆盖率分析图
        if 'coverage_simulation_data' in results_data:
            coverage_data = results_data['coverage_simulation_data']
            if 'n_labeled_sizes' in coverage_data: # Ensure sample sizes are in the data
                self.visualizer.plot_coverage_rate_analysis(
                    coverage_data,
                    sample_sizes=coverage_data['n_labeled_sizes'],
                    save_path=f"{self.output_dir}/coverage_rate_analysis.png"
                )
                print(f"    - Coverage rate analysis plot saved.")
            else:
                print("    - Skipping coverage rate plot: 'n_labeled_sizes' missing in coverage_simulation_data.")
        else:
            print("    - Skipping coverage rate plot: 'coverage_simulation_data' missing.")

        # 偏差-方差分析图
        if 'bias_variance_data' in results_data:
            self.visualizer.plot_bias_variance_analysis(
                results_data['bias_variance_data'],
                save_path=f"{self.output_dir}/bias_variance_analysis.png"
            )
            print(f"    - Bias-variance analysis plot saved.")
        else:
            print("    - Skipping bias-variance plot: 'bias_variance_data' missing.")
        
        print(f"图表已保存到 {self.output_dir}/") 