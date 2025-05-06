#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 设置绘图风格 - 更现代美观的风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')
sns.set_context('notebook', font_scale=1.2)

# 禁用最大图表警告
plt.rcParams.update({'figure.max_open_warning': 0})

# 设置输出图片路径
FIG_DIR = './fig'
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

# 加载数据
def load_data(file_path='./data/Crash_Analysis_System_(CAS)_data.csv', sample_size=None):
    print(f"Loading dataset: {file_path}")
    
    # 加载完整数据集 - 移除样本大小限制
    df_sample = pd.read_csv(file_path)
    print(f"Full dataset size: {df_sample.shape}")
    
    # 查看列名和数据类型
    print("\nDataset columns and types:")
    print(df_sample.dtypes)
    
    # 查看缺失值情况
    print("\nMissing values summary:")
    missing_data = df_sample.isnull().sum()
    missing_percent = (missing_data / len(df_sample)) * 100
    missing_info = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percent(%)': missing_percent
    })
    missing_info = missing_info[missing_info['Missing Count'] > 0]
    print(f"Number of columns with missing values: {len(missing_info)}")
    
    # 打印前10个缺失值最多的列
    print("\nTop 10 columns with most missing values:")
    print(missing_info.sort_values('Missing Count', ascending=False).head(10))
    
    return df_sample

# 数据探索：基本信息
def explore_basic_info(df):
    print("\nBasic dataset information:")
    print(f"- Rows: {df.shape[0]}")
    print(f"- Columns: {df.shape[1]}")
    
    # 事故严重程度分布
    if 'crashSeverity' in df.columns:
        print("\nCrash severity distribution:")
        severity_counts = df['crashSeverity'].value_counts()
        print(severity_counts)
        
        # 绘制事故严重程度分布图 - 使用饼图更直观
        plt.figure(figsize=(10, 8))
        
        # 使用更漂亮的颜色
        colors = sns.color_palette('viridis', len(severity_counts))
        
        plt.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=colors, explode=[0.1 if x in ['Fatal', 'Serious'] else 0 for x in severity_counts.index])
        plt.title('Crash Severity Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')  # 确保饼图是圆的
        
        # 添加图例，更清晰地显示各类别
        plt.legend(severity_counts.index, loc="best", bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/crash_severity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 年份分布
    if 'crashYear' in df.columns:
        print("\nCrash year distribution:")
        year_counts = df['crashYear'].value_counts().sort_index()
        print(year_counts)
        
        # 绘制年份分布图 - 完全重写这部分
        plt.figure(figsize=(14, 8))
        
        # 使用matplotlib的bar，而不是seaborn的barplot，以便更好地控制x轴标签
        years = year_counts.index.tolist()
        counts = year_counts.values.tolist()
        
        # 创建柱状图
        bars = plt.bar(years, counts, color='steelblue', width=0.7)
        
        # 添加数据标签，但只为主要年份添加标签以避免拥挤
        for bar, year, count in zip(bars, years, counts):
            # 只为部分值添加标签，避免图表过于拥挤
            if count > np.percentile(counts, 75) or year % 5 == 0:
                plt.text(year, count + max(counts)*0.01, f'{count:,}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 添加图表样式
        plt.title('Crash Count by Year', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Number of Crashes', fontsize=14)
        
        # 确保x轴包含所有年份并且以适当的间隔显示
        plt.xticks(years)
        if len(years) > 10:
            # 如果年份太多，只显示部分年份的标签
            plt.xticks(years[::2])  # 每隔一年显示标签
        
        plt.grid(True, alpha=0.3)
        
        # 设置y轴从0开始
        plt.ylim(0, max(counts) * 1.1)
        
        # 增强边框
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
            
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/crash_year_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 为特定区域绘制事故趋势图表
        print("\n为特定区域创建年度事故趋势图表...")
        
        # 按区域和年份分组计数
        region_year_counts = df.groupby(['region', 'crashYear']).size().reset_index(name='crash_count')
        
        # 指定需要绘图的区域
        regions_to_plot = [
            'Gisborne Region', 
            'Tasman Region', 
            'Otago Region', 
            'Southland Region'
        ]
        
        # 为每个区域创建完整的时间序列图表
        for region in regions_to_plot:
            # 过滤该区域的数据
            region_data = region_year_counts[region_year_counts['region'] == region]
            
            if len(region_data) == 0:
                print(f"警告: 未找到 {region} 的数据")
                continue
            
            # 创建图表
            plt.figure(figsize=(12, 8))
            
            # 绘制时间序列
            plt.plot(region_data['crashYear'], region_data['crash_count'], 
                    marker='o', linestyle='-', linewidth=2, markersize=8)
            
            # 设置图表标题和标签
            plt.title(f'Annual Crash Count Changes in {region} (Complete Data)', fontsize=16)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Crash Count', fontsize=14)
            
            # 确保X轴包含所有年份
            all_years = list(range(2000, 2025))
            plt.xticks(all_years, rotation=45)
            plt.xlim(min(all_years), max(all_years))
            
            # 添加网格线和紧凑布局
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图表
            save_path = f'{FIG_DIR}/region_{region.replace(" ", "_")}_complete_data.png'
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            print(f"已创建图表: {save_path}")
        
        print("区域图表创建完成")

# 问题1数据探索：雨天夜间无路灯事故
def explore_question1(df):
    """
    问题1：雨天夜间无路灯的事故是否更容易导致重伤或死亡？
    根据requirement.md的建模思路进行分析
    """
    print("\nQuestion 1 Analysis: Rainy, Night-time, and No Street Light Crashes")
    
    # 检查相关列是否存在
    relevant_columns = ['weatherA', 'weatherB', 'light', 'streetLight', 'crashSeverity']
    missing_columns = [col for col in relevant_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return
    
    # 1.1 数据准备 - 根据requirement.md中的建模思路
    # 识别雨天条件 - 使用更精确的方法
    df['rain'] = (df['weatherA'].str.contains('rain', case=False, na=False) | 
                  df['weatherB'].str.contains('rain', case=False, na=False)).astype(int)
    print(f"Rainy crashes: {df['rain'].sum()} ({df['rain'].mean()*100:.2f}%)")
    
    # 识别夜间条件
    df['night'] = df['light'].str.contains('Dark', case=False, na=False).astype(int)
    print(f"Night-time crashes: {df['night'].sum()} ({df['night'].mean()*100:.2f}%)")
    
    # 识别无路灯条件
    df['no_light'] = (df['streetLight'].str.contains('Null', case=False, na=False) | 
                      df['streetLight'].str.contains('Off', case=False, na=False)).astype(int)
    print(f"No street light crashes: {df['no_light'].sum()} ({df['no_light'].mean()*100:.2f}%)")
    
    # 创建组合条件
    df['rain_night_no_light'] = ((df['rain'] == 1) & (df['night'] == 1) & (df['no_light'] == 1)).astype(int)
    print(f"Rainy + Night + No Light crashes: {df['rain_night_no_light'].sum()} ({df['rain_night_no_light'].mean()*100:.2f}%)")
    
    # 定义严重事故（重伤或死亡）- 二元分类变量Y
    df['severe_crash'] = df['crashSeverity'].str.contains('Fatal|Serious', case=False, na=False).astype(int)
    print(f"Total severe crashes: {df['severe_crash'].sum()} ({df['severe_crash'].mean()*100:.2f}%)")
    
    # 创建交互项
    df['rain_night'] = df['rain'] * df['night']
    df['rain_no_light'] = df['rain'] * df['no_light']
    df['night_no_light'] = df['night'] * df['no_light']
    
    # 各条件下的严重事故率
    print("\nSevere crash rates under different conditions:")
    condition_columns = ['rain', 'night', 'no_light', 'rain_night', 'rain_no_light', 'night_no_light', 'rain_night_no_light']
    
    # 创建条件-严重事故对比DataFrame
    condition_stats = []
    
    for col in condition_columns:
        # 当条件存在时的严重事故率
        condition_true = df[df[col] == 1]['severe_crash'].mean() * 100
        # 当条件不存在时的严重事故率
        condition_false = df[df[col] == 0]['severe_crash'].mean() * 100
        # 差异及比例
        diff = condition_true - condition_false
        ratio = condition_true / condition_false if condition_false > 0 else float('inf')
        
        # 收集统计
        stat_row = {
            'Condition': col.replace('_', ' ').title(),
            'Crashes': df[col].sum(),
            'Severe Crash Rate (%)': condition_true,
            'Other Conditions Rate (%)': condition_false,
            'Difference (pp)': diff,
            'Ratio': ratio
        }
        condition_stats.append(stat_row)
    
    condition_stats_df = pd.DataFrame(condition_stats)
    print(condition_stats_df)
    
    # 绘制条件下的严重事故率对比图
    plt.figure(figsize=(12, 8))
    
    # 创建分组条形图
    x = np.arange(len(condition_stats_df))
    width = 0.35
    
    # 使用对比鲜明的颜色
    condition_rate_bars = plt.bar(x - width/2, condition_stats_df['Severe Crash Rate (%)'], 
                                  width, label='With Condition', color='#E63946')
    other_rate_bars = plt.bar(x + width/2, condition_stats_df['Other Conditions Rate (%)'], 
                             width, label='Without Condition', color='#457B9D')
    
    # 添加图表元素
    plt.title('Severe Crash Rate (%) Under Different Conditions', fontsize=16, fontweight='bold')
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel('Severe Crash Rate (%)', fontsize=14)
    plt.xticks(x, condition_stats_df['Condition'], rotation=45, ha='right')
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # 添加数据标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    add_labels(condition_rate_bars)
    add_labels(other_rate_bars)
    
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/severe_crash_rate_by_condition.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1.2 统计建模 - 二元逻辑回归
    print("\nLogistic Regression Analysis for Severe Crash Probability:")
    
    try:
        # 准备模型变量
        X = df[['rain', 'night', 'no_light', 'rain_night', 'rain_no_light', 'night_no_light', 'rain_night_no_light']]
        X = sm.add_constant(X)  # 添加截距项
        y = df['severe_crash']
        
        # 使用statsmodels进行逻辑回归
        model = sm.Logit(y, X)
        result = model.fit(disp=0)  # disp=0抑制迭代输出
        
        # 打印模型摘要
        print(result.summary())
        
        # 提取并展示关键系数
        coefficients = result.params
        odds_ratios = np.exp(coefficients)
        conf_int = result.conf_int()
        conf_int_odds = np.exp(conf_int)
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'Coefficient': coefficients,
            'Odds Ratio': odds_ratios,
            'p-value': result.pvalues,
            'CI 2.5%': conf_int[0],
            'CI 97.5%': conf_int[1],
            'OR CI 2.5%': conf_int_odds[0],
            'OR CI 97.5%': conf_int_odds[1]
        })
        
        print("\nLogistic Regression Results (Odds Ratios):")
        print(results_df[['Odds Ratio', 'p-value', 'OR CI 2.5%', 'OR CI 97.5%']])
        
        # 逻辑回归系数可视化
        plt.figure(figsize=(12, 8))
        
        # 排除常数项
        coef_df = results_df.iloc[1:].copy()
        coef_df.index = [idx.replace('_', ' ').title() for idx in coef_df.index]
        
        # 创建系数森林图
        y_pos = np.arange(len(coef_df))
        
        # 绘制水平条形图 - 使用更专业的配色方案
        bars = plt.barh(y_pos, coef_df['Coefficient'], 
                      xerr=[coef_df['Coefficient'] - coef_df['CI 2.5%'], 
                            coef_df['CI 97.5%'] - coef_df['Coefficient']],
                      align='center', alpha=0.8, color='#4E79A7', 
                      error_kw=dict(ecolor='#F28E2B', lw=2, capsize=5, capthick=2))
        
        # 添加零线
        plt.axvline(x=0, color='#E15759', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # 设置y轴标签 - 改进字体和样式
        plt.yticks(y_pos, coef_df.index, fontsize=12)
        
        # 添加标题和标签 - 使用更专业的标题格式
        plt.title('Logistic Regression Coefficients with 95% Confidence Intervals', 
                  fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Coefficient Value (Log Odds Ratio)', fontsize=14, labelpad=10)
        
        # 添加系数值标签 - 改进标签格式和位置
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width + 0.02 if width >= 0 else width - 0.08
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', va='center', fontsize=11, fontweight='bold',
                    color='black' if width >= 0 else 'white')
        
        # 改进图表边框和背景
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.2)
        plt.gca().spines['bottom'].set_linewidth(1.2)
        plt.grid(axis='x', alpha=0.2, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/logistic_regression_coefficients.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 可视化比值比（Odds Ratios）
        plt.figure(figsize=(12, 8))
        
        # 绘制比值比森林图
        y_pos = np.arange(len(coef_df))
        
        # 使用更专业的颜色方案和设计
        plt.barh(y_pos, coef_df['Odds Ratio'], 
                xerr=[coef_df['Odds Ratio'] - coef_df['OR CI 2.5%'], 
                      coef_df['OR CI 97.5%'] - coef_df['Odds Ratio']],
                align='center', alpha=0.8, color='#5A9BD4', 
                error_kw=dict(ecolor='#FF9D45', lw=2, capsize=5, capthick=2))
        
        # 添加1线（无效应）
        plt.axvline(x=1, color='#E15759', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # 设置y轴标签 - 改进字体
        plt.yticks(y_pos, coef_df.index, fontsize=12)
        
        # 设置x轴为对数尺度 - 增加刻度标签可读性
        plt.xscale('log')
        plt.xticks([0.3, 0.5, 0.7, 1, 1.5, 2, 3], 
                   ['0.3', '0.5', '0.7', '1.0', '1.5', '2.0', '3.0'], 
                   fontsize=11)
        
        # 添加标题和标签 - 更专业的格式
        plt.title('Odds Ratios with 95% Confidence Intervals (Log Scale)', 
                  fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Odds Ratio', fontsize=14, labelpad=10)
        
        # 添加参考区域和标注 - 更清晰的标记
        plt.axvspan(0.95, 1.05, alpha=0.1, color='#9C9C9C')
        plt.text(0.96, max(y_pos) + 0.5, 'No significant effect', 
                 fontsize=11, color='#666666', fontweight='bold')
        
        # 为每个条形添加数值标签
        for i, odds in enumerate(coef_df['Odds Ratio']):
            if odds < 1:
                x_pos = odds * 0.85
                h_align = 'right'
            else:
                x_pos = odds * 1.1
                h_align = 'left'
            
            plt.text(x_pos, i, f'{odds:.2f}', va='center', ha=h_align, 
                     fontsize=11, fontweight='bold', color='#333333')
        
        # 改进图表边框和背景
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.2)
        plt.gca().spines['bottom'].set_linewidth(1.2)
        plt.grid(axis='x', alpha=0.2, linestyle='--')
        
        # 添加图例说明条件的意义
        plt.figtext(0.5, 0.01, 
                    "Values < 1: Reduced odds of severe crash | Values > 1: Increased odds of severe crash", 
                    ha='center', fontsize=11, style='italic')
        
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/logistic_regression_odds_ratios.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 解释模型结果
        significant_predictors = results_df[results_df['p-value'] < 0.05]
        
        print("\nStatistically significant predictors (p < 0.05):")
        if not significant_predictors.empty:
            for idx, row in significant_predictors.iterrows():
                effect = "increases" if row['Odds Ratio'] > 1 else "decreases"
                print(f"- {idx.replace('_', ' ').title()} {effect} the odds of severe crash by a factor of {row['Odds Ratio']:.2f} (95% CI: {row['OR CI 2.5%']:.2f}-{row['OR CI 97.5%']:.2f})")
        else:
            print("No statistically significant predictors found.")
            
    except Exception as e:
        print(f"Error in logistic regression analysis: {e}")

# 主函数
if __name__ == "__main__":
    # 加载数据集样本
    sample_df = load_data()
    
    # 探索基本信息
    explore_basic_info(sample_df)
    
    # 问题1探索
    explore_question1(sample_df)
    
    print("\nData exploration completed!") 