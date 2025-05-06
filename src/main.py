#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
新西兰交通事故分析项目主程序
整合现有的数据探索和分析模块
"""

import os
import logging
import shutil
from pathlib import Path
import kagglehub

from data_exploration import explore_data
from region_year_analysis import analyze_region_year
from vehicle_trend_analysis import analyze_vehicle_trends

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """创建必要的目录结构"""
    dirs = ['data', 'fig']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info("目录结构创建完成")

def download_dataset():
    """下载数据集并移动到data目录"""
    try:
        logger.info("开始下载数据集...")
        # 下载数据集
        temp_path = kagglehub.dataset_download("maryamrahmani/crash-analysis-system-cas-data-new-zealand")
        
        # 获取下载目录中的所有文件
        source_dir = Path(temp_path)
        target_dir = Path('data')
        
        # 移动所有文件到data目录
        for file_path in source_dir.glob('*'):
            target_path = target_dir / file_path.name
            shutil.copy2(file_path, target_path)
            logger.info(f"已复制文件: {file_path.name} 到 data/")
        
        logger.info("数据集已保存到data/目录")
        return str(target_dir)
    except Exception as e:
        logger.error(f"数据集下载或移动失败: {e}")
        raise

def main():
    """主函数"""
    try:
        # 1. 创建目录结构
        setup_directories()
        
        # 2. 下载数据集
        dataset_path = download_dataset()
        
        # 3. 数据探索
        logger.info("开始数据探索...")
        explore_data()
        
        # 4. 区域年度分析
        logger.info("开始区域年度分析...")
        analyze_region_year()
        
        # 5. 车辆趋势分析
        logger.info("开始车辆趋势分析...")
        analyze_vehicle_trends()
        
        logger.info("所有分析完成！结果已保存到相应目录")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main() 