#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import statsmodels.api as sm
from scipy import stats

# 设置绘图风格
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')

# 设置输出图片路径
FIG_DIR = './fig'
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

# 加载数据
def load_data(file_path='./data/Crash_Analysis_System_(CAS)_data.csv'):
    print(f"正在加载数据集: {file_path}")
    
    # 选择与车辆类型和年份相关的列
    columns = ['crashYear', 'carStationWagon', 'motorcycle', 'suv', 
               'taxi', 'truck', 'bus', 'vanOrUtility', 'otherVehicleType', 
               'bicycle', 'moped', 'pedestrian', 'unknownVehicleType']
    
    # 读取数据
    df = pd.read_csv(file_path, usecols=columns)
    print(f"数据集大小: {df.shape}")
    
    # 打印年份分布情况
    year_counts = df['crashYear'].value_counts().sort_index()
    print("\n年份分布情况:")
    print(year_counts)
    
    return df

# Mann-Kendall趋势检验
def mann_kendall_test(timeseries):
    """执行Mann-Kendall趋势检验"""
    n = len(timeseries)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(timeseries[j] - timeseries[i])
    
    # 计算方差
    var_s = (n*(n-1)*(2*n+5))/18
    
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    # 计算双尾p值
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # 趋势方向
    trend = "Increasing" if z > 0 else "Decreasing" if z < 0 else "No Trend"
    
    return {
        'trend': trend,
        'p_value': p,
        'significant': p < 0.05
    }

# 问题3分析：车辆类型参与事故的趋势
def analyze_vehicle_trends(df):
    """
    问题3：随着时间的推移，某些类型的车辆是否更频繁地参与事故？
    """
    print("\n问题3分析: 车辆类型参与事故的时间趋势")
    
    # 检查必要的列是否存在
    if 'crashYear' not in df.columns:
        print("数据集缺少必要的'crashYear'列")
        return
    
    # 识别车辆类型列
    vehicle_columns = [col for col in df.columns if col != 'crashYear']
    
    # 验证车辆类型列存在
    if not vehicle_columns:
        print("数据集缺少车辆类型相关列")
        return
    
    print(f"分析中的车辆类型: {', '.join(vehicle_columns)}")
    
    # 按年份统计各类型车辆参与事故的数量
    yearly_vehicle_counts = df.groupby('crashYear')[vehicle_columns].sum()
    
    # 诊断输出年份范围
    min_year = yearly_vehicle_counts.index.min()
    max_year = yearly_vehicle_counts.index.max()
    print(f"\n年份范围: {min_year} - {max_year}")
    print(f"可用年份: {sorted(yearly_vehicle_counts.index.unique())}")
    
    # 确保x轴显示到2024年
    plot_max_year = 2024
    
    # 计算每年的总事故数涉及的车辆
    yearly_vehicle_counts['total_vehicles'] = yearly_vehicle_counts.sum(axis=1)
    
    # 计算每种车辆类型占比
    vehicle_proportions = yearly_vehicle_counts.copy()
    for col in vehicle_columns:
        vehicle_proportions[col] = yearly_vehicle_counts[col] / yearly_vehicle_counts['total_vehicles'] * 100
    
    # 绘制车辆类型随时间的变化趋势
    plt.figure(figsize=(14, 8))
    for col in vehicle_columns:
        plt.plot(yearly_vehicle_counts.index, yearly_vehicle_counts[col], marker='o', label=col)
    
    plt.title('Annual Crash Count by Vehicle Type (2000-2024)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Crash Count', fontsize=14)
    plt.legend(title='Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # 确保x轴显示所有年份，并且标注清晰，强制显示到2024年
    plt.xlim(min_year - 0.5, plot_max_year + 0.5)
    plt.xticks(range(min_year, plot_max_year + 1, 2))  # 每隔2年标注一次
    
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/vehicle_type_trends.png', dpi=300)
    plt.close()
    
    # 绘制车辆类型占比随时间的变化趋势
    plt.figure(figsize=(14, 8))
    for col in vehicle_columns:
        plt.plot(vehicle_proportions.index, vehicle_proportions[col], marker='o', label=col)
    
    plt.title('Annual Proportion of Crashes by Vehicle Type (2000-2024)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Proportion (%)', fontsize=14)
    plt.legend(title='Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # 确保x轴显示所有年份，并且标注清晰，强制显示到2024年
    plt.xlim(min_year - 0.5, plot_max_year + 0.5)
    plt.xticks(range(min_year, plot_max_year + 1, 2))  # 每隔2年标注一次
    
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/vehicle_type_proportion_trends.png', dpi=300)
    plt.close()
    
    # 堆叠面积图显示车辆类型构成随时间变化
    plt.figure(figsize=(14, 8))
    # 确保数据索引完整从数据年份最小值到2024
    full_years = pd.Index(range(min_year, plot_max_year + 1))
    proportion_data = vehicle_proportions[vehicle_columns].reindex(full_years)
    proportion_data = proportion_data.fillna(method='ffill')  # 使用前向填充处理缺失年份
    
    proportion_data.plot.area(figsize=(14, 8))
    plt.title('Annual Proportion of Crashes by Vehicle Type (2000-2024)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Proportion (%)', fontsize=14)
    plt.legend(title='Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # 确保x轴显示所有年份，显示到2024年
    plt.xlim(min_year, plot_max_year)
    plt.xticks(range(min_year, plot_max_year + 1, 2))  # 每隔2年标注一次
    
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/vehicle_type_area_chart.png', dpi=300)
    plt.close()
    
    # Mann-Kendall趋势检验
    print("\nMann-Kendall趋势检验结果:")
    for col in vehicle_columns:
        # 检验绝对数量趋势
        count_trend = mann_kendall_test(yearly_vehicle_counts[col].values)
        # 检验比例趋势
        proportion_trend = mann_kendall_test(vehicle_proportions[col].values)
        
        print(f"\n车辆类型: {col}")
        print(f"  绝对数量趋势: {count_trend['trend']}" + 
              (f" (p={count_trend['p_value']:.4f}, 显著)" if count_trend['significant'] else f" (p={count_trend['p_value']:.4f}, 不显著)"))
        print(f"  占比趋势: {proportion_trend['trend']}" + 
              (f" (p={proportion_trend['p_value']:.4f}, 显著)" if proportion_trend['significant'] else f" (p={proportion_trend['p_value']:.4f}, 不显著)"))
    
    # 绘制显著趋势的车辆类型详细图
    significant_vehicles = []
    for col in vehicle_columns:
        proportion_trend = mann_kendall_test(vehicle_proportions[col].values)
        if proportion_trend['significant']:
            significant_vehicles.append((col, proportion_trend['trend']))
    
    # 添加统计不显著但需要展示的车型
    additional_vehicles = ['bicycle', 'moped', 'otherVehicleType', 'carStationWagon']
    for vehicle in additional_vehicles:
        if vehicle in vehicle_columns and not any(v[0] == vehicle for v in significant_vehicles):
            proportion_trend = mann_kendall_test(vehicle_proportions[vehicle].values)
            significant_vehicles.append((vehicle, proportion_trend['trend']))
    
    if significant_vehicles:
        print("\n展示的车辆类型趋势:")
        for vehicle, trend in significant_vehicles:
            print(f"  {vehicle}: {trend}趋势")
            
            # 为每个需要展示的车辆类型绘制单独的图表
            plt.figure(figsize=(10, 6))
            
            # 绘制数据点
            plt.scatter(vehicle_proportions.index, vehicle_proportions[vehicle], color='blue')
            
            # 添加趋势线（线性回归）
            x = vehicle_proportions.index.astype(int)
            y = vehicle_proportions[vehicle].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            # 创建包含原始数据年份到2024年的完整范围
            x_trend = np.array(range(min_year, plot_max_year + 1))
            y_trend = p(x_trend)
            plt.plot(x_trend, y_trend, "r--", linewidth=2)
            
            # 添加趋势方向和斜率
            slope = z[0]
            trend_text = "Increasing" if slope > 0 else "Decreasing"
            plt.annotate(f'Trend Slope: {slope:.4f}% / year ({trend_text})', 
                        xy=(0.05, 0.95), 
                        xycoords='axes fraction',
                        backgroundcolor='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # 确保标题显示到2024年
            plt.title(f'Trend of {vehicle} in Crashes (2000-2024)', fontsize=14)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Proportion (%)', fontsize=12)
            plt.grid(True)
            
            # 确保x轴显示所有年份，显示到2024年
            plt.xlim(min_year - 0.5, plot_max_year + 0.5)
            plt.xticks(range(min_year, plot_max_year + 1, 4))  # 每隔4年标注一次以避免拥挤
            
            plt.tight_layout()
            plt.savefig(f'{FIG_DIR}/trend_{vehicle}.png', dpi=300)
            plt.close()
    
    # 创建区域总体趋势图
    plt.figure(figsize=(14, 10))
    # 创建包含所有区域事故数据的图表
    plt.title('Vehicle Types Involved in Crashes (2000-2024)', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Crash Count', fontsize=14)
    
    # 绘制所有区域的事故趋势
    cmap = plt.cm.get_cmap('tab20', len(vehicle_columns))
    for i, col in enumerate(vehicle_columns):
        plt.plot(yearly_vehicle_counts.index, yearly_vehicle_counts[col], 
                 marker='o', linestyle='-', linewidth=2, markersize=5,
                 color=cmap(i), label=col)
    
    # 确保X轴包含所有年份到2024
    plt.xlim(min_year, plot_max_year)
    plt.xticks(range(min_year, plot_max_year + 1, 2), rotation=45)
    
    # 添加图例
    plt.legend(title='Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加网格线
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f'{FIG_DIR}/vehicle_types_trend_overview.png', dpi=300)
    plt.close()
    
    print("\n所有车辆类型趋势图生成完成，包括到2024年数据")

# 主函数
if __name__ == "__main__":
    # 加载数据
    df = load_data()
    
    # 分析车辆类型趋势
    analyze_vehicle_trends(df)
    
    print("\n分析完成") 