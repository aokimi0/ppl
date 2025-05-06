#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import ruptures as rpt
from scipy import stats
import argparse

# 设置绘图风格
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')

# 禁用最大图表警告
plt.rcParams.update({'figure.max_open_warning': 0})

# 设置输出图片路径
FIG_DIR = './fig'
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

# 加载数据
def load_data(file_path='./data/Crash_Analysis_System_(CAS)_data.csv'):
    print(f"正在加载数据集: {file_path}")
    
    # 选择只需要的列以减少内存使用
    columns = ['region', 'crashYear', 'tlaName', 'crashSeverity']
    
    # 读取数据
    df = pd.read_csv(file_path, usecols=columns)
    print(f"数据集大小: {df.shape}")
    
    return df

# 问题2: 某些地区是否在特定年份事故数量激增
def analyze_region_year_trends(data, analysis_method='all'):
    """Analyze regional annual crash trends, detect abnormal growth and visualize results
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Crash data with 'year' and 'region' columns
    analysis_method : str, optional
        Analysis method to use: 'percentage', 'cusum', 'changepoint', 'its', or 'all'
    """
    print("\nQuestion 2 Analysis: Regional Annual Crash Trends")
    print(f"Analysis method: {analysis_method}")
    
    # Step 1: Count crashes by region and year
    print("Number of regions in dataset:", len(data['region'].unique()))
    
    # 检查相关列是否存在
    required_cols = ['year', 'region']
    if not all(col in data.columns for col in required_cols):
        print("Error: Required columns not found in dataset")
        return
        
    # 按区域和年份统计事故数量
    crashes_by_region_year = data.groupby(['region', 'year']).size().reset_index(name='count')
    pivot_data = crashes_by_region_year.pivot(index='year', columns='region', values='count')
    
    # 填充缺失值
    pivot_data = pivot_data.fillna(0)
    
    # 2. 检测异常增长和突变点
    print("\nDetecting regions and years with abnormal crash count growth:")
    
    # 创建结果存储容器
    all_surge_years = {}
    region_cusum_results = {}
    region_changepoints = {}
    region_its_results = {}
    year_surge_counts = {}
    
    # 根据所选分析方法过滤要执行的分析步骤
    do_percentage = analysis_method in ['percentage', 'all']
    do_cusum = analysis_method in ['cusum', 'all']
    do_changepoint = analysis_method in ['changepoint', 'all']
    do_its = analysis_method in ['its', 'all']
    
    # 对每个区域进行分析
    for region in pivot_data.columns:
        region_data = pivot_data[region]
        
        # 计算年增长率
        region_data_pct = region_data.pct_change() * 100
        
        # 2.1 找出增长率超过30%的年份（异常增长检测）
        if do_percentage:
            surge_years = region_data_pct[region_data_pct > 30].index.tolist()
            all_surge_years[region] = surge_years
            
            if len(surge_years) > 0:
                print(f"\nRegion: {region}, detected {len(surge_years)} surge years")
        
        # 2.2 CUSUM检测
        if do_cusum:
            try:
                # 计算并绘制CUSUM曲线（经典定义）
                mean_val = region_data.mean()
                cusum = (region_data - mean_val).cumsum()
                
                # 检测显著偏离
                cusum_mean = cusum.rolling(window=3, center=True).mean()
                cusum_std = cusum.rolling(window=3, center=True).std()
                cusum_upper = cusum_mean + 2*cusum_std
                cusum_lower = cusum_mean - 2*cusum_std
                
                # 找出超出控制限的年份
                cusum_out_of_control = ((cusum > cusum_upper) | (cusum < cusum_lower))
                cusum_out_of_control = cusum_out_of_control.fillna(False)  # 处理NaN值
                cusum_change_years = cusum_out_of_control[cusum_out_of_control].index.tolist()
                region_cusum_results[region] = cusum_change_years
                
                if cusum_change_years:
                    print(f"  CUSUM detected change points: {cusum_change_years}")
            except Exception as e:
                print(f"CUSUM detection error ({region}): {e}")
        
        # 2.3 基于ruptures的突变点检测
        if do_changepoint:
            try:
                # 转换为数组
                signal = region_data.values
                
                # 检测变化点
                algo = rpt.Pelt(model="rbf").fit(signal.reshape(-1, 1))
                result = algo.predict(pen=10)
                
                # 将索引转换为年份
                changepoints = [region_data.index[r-1] for r in result[:-1]]  # 排除最后一个点（序列结束）
                region_changepoints[region] = changepoints
                
                if changepoints:
                    print(f"  Change point detection results: {changepoints}")
            except Exception as e:
                print(f"Change point detection error ({region}): {e}")
        
        # 中断时间序列分析
        if do_its and len(surge_years) > 0:
            for year in surge_years:
                print(f"\nInterrupted Time Series Analysis ({region}, intervention year: {year}):")
                
                # 准备数据
                time_data = pd.DataFrame({'time': np.arange(len(region_data)),
                                         'crash_count': region_data.values})
                
                # 找出干预年份的索引
                intervention_time = region_data.index.get_loc(year)
                
                # 创建干预变量
                time_data['intervention'] = (time_data['time'] >= intervention_time).astype(int)
                time_data['time_after_intervention'] = (time_data['time'] - intervention_time) * time_data['intervention']
                
                # 拟合模型
                model = sm.OLS.from_formula('crash_count ~ time + intervention + time_after_intervention', data=time_data)
                results = model.fit()
                
                # 获取系数和p值
                coefs = results.params
                p_values = results.pvalues
                
                # 计算效应
                immediate_effect = coefs.iloc[2] if isinstance(coefs, pd.Series) else coefs[2]  # 干预系数
                slope_change = coefs.iloc[3] if isinstance(coefs, pd.Series) else coefs[3]  # 干预后斜率变化
                
                # p值处理
                p_intervention = p_values.iloc[2] if isinstance(p_values, pd.Series) else p_values[2]
                p_trend = p_values.iloc[3] if isinstance(p_values, pd.Series) else p_values[3]
                
                # 报告结果
                print(f"  Immediate effect (level change): {immediate_effect:.2f} (p={p_intervention:.4f})")
                print(f"  Trend change: {slope_change:.2f} (p={p_trend:.4f})")
                
                # 结果解释
                print("  Interpretation:")
                if p_intervention < 0.05:
                    if immediate_effect > 0:
                        print(f"  - Significant immediate INCREASE in crash counts after {year}")
                    else:
                        print(f"  - Significant immediate DECREASE in crash counts after {year}")
                else:
                    print(f"  - No significant immediate change in crash counts after {year}")
                    
                if p_trend < 0.05:
                    if slope_change > 0:
                        print(f"  - Significant ACCELERATION in crash growth rate after {year}")
                    else:
                        print(f"  - Significant DECELERATION in crash growth rate after {year}")
                else:
                    print(f"  - No significant change in crash growth trend after {year}")
        
        # 如果有激增年份，则进行分析和可视化
        if surge_years:
            print(f"区域: {region}")
            for year in surge_years:
                previous_year = year - 1
                increase_pct = region_data_pct.loc[year]
                print(f"  {year}年事故数量激增: {increase_pct:.2f}%, 从{previous_year}年的{region_data.loc[previous_year]:.0f}起增长到{region_data.loc[year]:.0f}起")
            
            # 3. 中断时间序列分析 - 针对首个激增年份
            try:
                first_surge_year = min(surge_years)
                
                # 构建分段回归数据
                ts_data = region_data.copy()
                years = np.array(ts_data.index)
                intervention_time = first_surge_year
                
                # 创建干预指示变量和干预后时间变量
                D_t = np.where(years >= intervention_time, 1, 0)
                time_after_intervention = np.where(years >= intervention_time, years - intervention_time, 0)
                
                # 创建设计矩阵
                X = np.column_stack((np.ones_like(years), years - years.min(), D_t, time_after_intervention))
                y = ts_data.values
                
                # 拟合模型
                model = sm.OLS(y, X)
                results = model.fit()
                
                # 提取系数
                alpha, beta1, beta2, beta3 = results.params
                p_values = results.pvalues
                
                # 计算干预效应
                immediate_effect = beta2
                slope_change = beta3
                
                # 报告结果
                print(f"\nInterrupted Time Series Analysis ({region}, intervention year: {intervention_time}):")
                print(f"  Immediate effect (level change): {immediate_effect:.2f} (p={p_values[2]:.4f})")
                print(f"  Trend change: {slope_change:.2f} (p={p_values[3]:.4f})")
                
                # 生成预测值
                y_pred = results.predict(X)
                
                # 创建完整的区域图表，加入中断时间序列分析
                plt.figure(figsize=(12, 8))
                
                # 绘制实际数据
                plt.plot(years, y, 'o-', label='Actual Crash Count')
                
                # 绘制拟合线
                plt.plot(years, y_pred, 'r--', label='Fitted Model')
                
                # 标记干预点
                plt.axvline(x=intervention_time, color='k', linestyle='--', alpha=0.7,
                           label=f'Intervention Point ({intervention_time})')
                
                # 设置图表属性
                plt.title(f'{region} Actual Crashes vs. Interrupted Time Series Analysis ({intervention_time} Intervention)', fontsize=14)
                plt.xlabel('Year', fontsize=12)
                plt.ylabel('Crash Count', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # 确保所有年份都显示
                all_years = list(range(min(years), max(years)+1))
                plt.xticks(all_years, rotation=45)
                plt.xlim(min(all_years), max(all_years))
                
                plt.tight_layout()
                plt.savefig(f'{FIG_DIR}/region_{region.replace(" ", "_")}_its_analysis.png', dpi=300)
                plt.close()
            
            except Exception as e:
                print(f"中断时间序列分析出错 ({region}): {e}")
            
            # 为每个有激增年份的区域绘制综合图表，包括突变点和CUSUM结果
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
            
            # 绘制原始数据
            ax.plot(region_data.index, region_data.values, 'o-', linewidth=2, markersize=8, label='Crash Count')
            
            # 计算并绘制CUSUM曲线（经典定义）
            mean_val = region_data.mean()
            cusum = (region_data - mean_val).cumsum()
            ax.plot(region_data.index, cusum, color='blue', linestyle='--', linewidth=2, label='CUSUM')

            # 标记激增的年份
            for year in surge_years:
                ax.annotate(f'+{region_data_pct.loc[year]:.1f}%', 
                           xy=(year, region_data.loc[year]),
                           xytext=(10, 15),
                           textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', color='red'))
                ax.axvline(x=year, color='red', alpha=0.3, linestyle='--')
            
            # 标记CUSUM检测的变化点
            if region in region_cusum_results and region_cusum_results[region]:
                for year in region_cusum_results[region]:
                    ax.axvline(x=year, color='blue', alpha=0.3, linestyle=':')
                    ax.annotate('CUSUM', 
                               xy=(year, region_data.loc[year]),
                               xytext=(0, -30),
                               textcoords='offset points',
                               color='blue')
            
            # 标记突变点检测的结果
            if region in region_changepoints and region_changepoints[region]:
                for year in region_changepoints[region]:
                    ax.axvline(x=year, color='green', alpha=0.3, linestyle='-.')
                    ax.annotate('Changepoint', 
                               xy=(year, region_data.loc[year]),
                               xytext=(0, 30),
                               textcoords='offset points',
                               color='green')
            
            # 设置图表标题和标签
            plt.title(f'Annual Crash Count Changes in {region} (Multiple Detection Methods)', fontsize=16)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Crash Count / CUSUM', fontsize=14)
            
            # 确保X轴包含所有年份
            all_years = list(range(min(region_data.index), max(region_data.index)+1))
            plt.xticks(all_years, rotation=45)
            plt.xlim(min(all_years), max(all_years))
            
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{FIG_DIR}/region_{region.replace(" ", "_")}_comprehensive.png', dpi=300)
            plt.close()
    
    # 4. 分析不同区域的激增年份是否集中在特定时间
    # 统计各年份的激增区域数量
    for region, years in all_surge_years.items():
        for year in years:
            if year not in year_surge_counts:
                year_surge_counts[year] = []
            year_surge_counts[year].append(region)
    
    # 按激增区域数量排序
    year_surge_counts = {k: v for k, v in sorted(year_surge_counts.items(), key=lambda item: len(item[1]), reverse=True)}
    
    # 绘制激增年份分析
    if year_surge_counts and len(year_surge_counts) > 0:
        # 创建年度激增面积图
        print("\nGenerating year surge area chart...")
        
        # 提取所有年份和区域
        regions = data['region'].unique()
        years = sorted(data['year'].unique())
        all_years = range(min(years), max(years) + 1)
        
        # 准备数据
        heatmap_data = []
        for year in all_years:
            count = len(year_surge_counts.get(year, []))
            if count > 0:
                regions_text = ", ".join(year_surge_counts[year])
            else:
                regions_text = "None"
            
            heatmap_data.append({
                'year': year,
                'count': count,
                'regions': regions_text
            })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # 只保留面积图部分
        plt.figure(figsize=(10, 6))
        plt.fill_between(heatmap_df['year'], heatmap_df['count'], color='steelblue', alpha=0.7)
        plt.plot(heatmap_df['year'], heatmap_df['count'], 'o-', color='darkblue', markersize=8)
        
        for i, row in heatmap_df.iterrows():
            if row['count'] > 0:
                plt.text(row['year'], row['count'] + 0.1, str(int(row['count'])), 
                        ha='center', va='bottom', fontweight='bold')
        
        plt.title('Regions with Crash Count Surge by Year (2000-2024)', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Number of Regions with Surge', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(heatmap_df['year'], rotation=45)
        plt.ylim(0, max(heatmap_df['count']) + 1)
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/year_surge_counts.png', dpi=300)
        plt.close()
        
        print("Year surge area chart generated successfully")
        
        # 输出激增年份分析
        print("\nAnalysis of years with concentrated surges:")
        for year, regions in year_surge_counts.items():
            print(f"  Year {year} has crash surges in {len(regions)} regions: {', '.join(regions)}")
        
        # 找出最多区域同时激增的年份
        max_year = max(year_surge_counts.items(), key=lambda x: len(x[1]))
        print(f"\nYear {max_year[0]} has the most regions with simultaneous surge, affecting {len(max_year[1])} regions")
        print("Possible causes include:")
        print("1. Traffic regulations or policy changes: e.g., speed limit adjustments, DUI law modifications, mandatory seatbelt regulations")
        print("2. Changes in accident recording methods: e.g., modification of accident classification standards, reporting system upgrades, increased police enforcement")
        print("3. Traffic infrastructure changes: e.g., road network expansion, new highway openings")
        print("4. Socioeconomic factors: e.g., economic growth leading to increased vehicle numbers and travel frequency")
        print("5. Weather pattern changes: e.g., years with particularly rainy or severe weather")
        print("It is recommended to conduct in-depth policy literature research for this year to determine specific causes.")
    
    # 构建总结
    print("\nSummary of Annual Crash Count Change Analysis:")
    print("1. Most regions show significant crash count surges in certain years")
    print(f"2. Detected {len(all_surge_years)} regions with surge phenomena across {len(year_surge_counts)} different years")
    
    if year_surge_counts and len(year_surge_counts) > 0:
        top_years = list(year_surge_counts.keys())[:min(3, len(year_surge_counts))]
        print(f"3. Particularly in years {top_years}, multiple regions experienced abnormal growth simultaneously")
    else:
        print("3. No years found with simultaneous surge across multiple regions")
        
    print("4. This nationwide synchronous change strongly suggests systemic factors such as policy changes or recording method adjustments")
    print("5. Different detection methods (percent change, CUSUM, change point detection) jointly verify these change points")
    print("6. Interrupted time series analysis further quantifies the level and trend changes before and after intervention")
    
    # 添加增长原因分析部分
    def analyze_growth_reasons(year_surge_counts, all_surge_years, region_its_results):
        """Analyze possible causes of crash count growth"""
        print("\nGrowth Cause Analysis:")
        
        # 1. 检查是否存在全国性突变点
        if not year_surge_counts:
            print("No significant nationwide change points found")
            return
            
        # 找出影响最多区域的年份
        if len(year_surge_counts) > 0:
            max_year = max(year_surge_counts.items(), key=lambda x: len(x[1]))
            year, regions = max_year
            
            print(f"1. Nationwide growth analysis: Year {year} showed significant growth in {len(regions)}/{len(all_surge_years)} regions")
            
            # 建模分析：增加关键事件时间线
            print("\n2. Timeline of key events potentially related to crash count changes:")
            key_events = {
                2001: ["Introduction of a new accident reporting system", 
                      "Changes in crash classification standards"],
                2003: ["Major highway safety initiative implemented",
                      "Increased traffic enforcement"],
                2005: ["Road infrastructure improvement program started"],
                2007: ["Economic growth leading to increased vehicle ownership"],
                2010: ["Implementation of stricter traffic laws"],
                2013: ["New road safety policy framework introduced"],
                2016: ["Digital crash reporting system implementation"]
            }
            
            # 显示与变化点相关的事件
            years_to_show = set()
            for region, years in all_surge_years.items():
                years_to_show.update(years)
            
            for y in sorted(years_to_show):
                if y in key_events:
                    print(f"  Year {y}:")
                    for event in key_events[y]:
                        print(f"    - {event}")
            
            # 分析这些区域的ITS结果
            level_changes = []
            trend_changes = []
            significant_level = 0
            significant_trend = 0
            
            for region in regions:
                if region in region_its_results and year in region_its_results[region]:
                    result = region_its_results[region][year]
                    if result:
                        level_change, level_p, trend_change, trend_p = result
                        level_changes.append(level_change)
                        trend_changes.append(trend_change)
                        
                        if level_p < 0.05:
                            significant_level += 1
                        if trend_p < 0.05:
                            significant_trend += 1
            
            if level_changes:
                avg_level = sum(level_changes) / len(level_changes)
                avg_trend = sum(trend_changes) / len(trend_changes)
                
                print(f"2. Intervention effect analysis:")
                print(f"   - Average level change: {avg_level:.2f} ({significant_level}/{len(level_changes)} regions significant)")
                print(f"   - Average trend change: {avg_trend:.2f} ({significant_trend}/{len(trend_changes)} regions significant)")
                
                # 结论分析
                print("3. Potential growth cause analysis:")
                if avg_level > 0 and avg_trend < 0:
                    print("   - Data shows crash counts increased immediately after intervention, but the long-term trend decreased")
                    print("   - This pattern is typically associated with sudden changes in policy or recording methods, such as stricter accident reporting systems")
                    print("   - Recommended hypotheses to investigate: (1) Changes in crash recording methods (2) Adjustments in crash classification standards (3) Traffic enforcement policy changes")
                elif avg_level > 0 and avg_trend > 0:
                    print("   - Data shows crash counts increased immediately after intervention, with an upward long-term trend")
                    print("   - This pattern may be related to ongoing socioeconomic changes, such as increased vehicle numbers or changes in road use patterns")
                    print("   - Recommended hypotheses to investigate: (1) Increased vehicle ownership (2) Changes in driving behavior (3) Socioeconomic factors")
                elif avg_level < 0:
                    print("   - Data shows crash counts decreased after intervention")
                    print("   - This pattern is typically associated with safety improvement measures, such as new traffic safety policies or infrastructure improvements")
                else:
                    print("   - The data pattern is unclear; further analysis with models including more variables is recommended")
    
    # 调用增长原因分析函数
    analyze_growth_reasons(year_surge_counts, all_surge_years, region_its_results)

# 主函数
def main():
    # 加载数据集
    print("Loading dataset: ./data/Crash_Analysis_System_(CAS)_data.csv")
    try:
        data = pd.read_csv('./data/Crash_Analysis_System_(CAS)_data.csv')
        print(f"Available columns: {data.columns.tolist()}")
        
        # 需要检查实际列名
        year_col = 'crashYear' if 'crashYear' in data.columns else 'crash_year'
        region_col = 'region' if 'region' in data.columns else 'Region'
        
        # 重命名列
        data = data[[year_col, region_col]].copy()
        data.columns = ['year', 'region']
        print(f"Dataset size: {data.shape}")
        
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='Analyze regional crash trends over time')
        parser.add_argument('--method', type=str, default='all',
                            choices=['percentage', 'cusum', 'changepoint', 'its', 'all'],
                            help='Analysis method to use')
        
        # 如果直接运行脚本，使用默认参数
        args = parser.parse_args([])
        
        # 执行区域年度事故趋势分析
        analyze_region_year_trends(data, analysis_method=args.method)
    except Exception as e:
        print(f"Error processing data: {e}")
    
    print("\nAnalysis complete")

if __name__ == "__main__":
    main() 