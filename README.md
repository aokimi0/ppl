# 预测驱动推断 (Prediction-Powered Inference) 实验

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 项目概述

本项目实现并评估了**预测驱动推断（Prediction-Powered Inference, PPI）**框架，这是一种结合少量高质量标注数据、大量未标注数据和预训练机器学习模型进行统计推断的先进方法。

### 核心思想

在现代数据科学中，我们常常面临这样的挑战：
- 🔸 **标注数据稀缺且昂贵** - 获取高质量的标注数据成本高昂
- 🔸 **未标注数据丰富** - 但直接使用ML预测进行统计推断可能产生偏差
- 🔸 **预训练模型可用** - 但其预测存在不确定性

PPI方法巧妙地解决了这一矛盾，通过校正机制消除ML预测的系统性偏差，在保证统计有效性的同时大幅提升推断效率。

## 方法对比

本项目比较了三种统计推断方法：

| 方法 | 描述 | 优势 | 劣势 |
|------|------|------|------|
| **经典方法** | 仅使用少量标注数据 | 统计理论保证 | 样本量小，功效低 |
| **朴素ML方法** | 直接使用所有预测值 | 利用大量数据 | 可能产生严重偏差 |
| **PPI方法** | 校正偏差的预测驱动推断 | 兼顾效率与有效性 | 依赖预测质量 |

## 实验设计

### 数据模拟
- **医学场景**: 模拟阿尔茨海默病研究数据（类似ADNI数据集）
- **特征变量**: 年龄、教育年限、APOE4基因型、认知评分等
- **目标变量**: 12个月认知变化评分
- **数据分割**: 预训练集 → 标注集（小）+ 未标注集（大）

### 评估指标
- ✅ **置信区间覆盖率** - 统计有效性
- ✅ **置信区间宽度** - 推断精度  
- ✅ **估计偏差** - 准确性
- ✅ **相对效率** - 方法效率对比

## 项目结构

```
.
├── src/                           # 源代码
│   ├── __init__.py               # 包初始化
│   ├── ppi_inference.py          # PPI核心算法实现
│   ├── data_generator.py         # 数据生成与预处理
│   ├── visualization.py          # 学术级可视化
│   ├── experiments.py            # 实验运行框架
│   └── main.py                   # 主程序入口
├── fig/                          # 图表输出目录
├── results/                      # 实验结果数据
├── reference/                    # 参考文献和模板
├── style/                        # LaTeX样式文件
├── requirements.txt              # Python依赖
├── requirement.md                # 项目需求文档
├── solution.md                   # 解决方案说明
└── README.md                     # 项目说明（本文件）
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd crash

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行实验

```bash
# 运行完整实验
python src/main.py

# 快速模式（减少计算量）
python src/main.py --quick

# 指定输出目录
python src/main.py --output-dir results/figures
```

### 3. 查看结果

实验完成后，查看生成的文件：
- `fig/confidence_intervals_comparison.png` - 置信区间比较图
- `fig/data_overview.png` - 数据概览图  
- `results/experiment_results.json` - 详细实验结果
- `results/method_comparison.csv` - 方法比较表

## 核心算法

### PPI估计量

对于均值估计，PPI方法的核心公式为：

```
μ̂_PPI = (1/N)∑f(X_i') - (1/n)∑[f(X_j) - Y_j]
```

其中：
- `f(X_i')` 是未标注数据的预测值
- `f(X_j) - Y_j` 是标注数据上的预测误差（校正项）

### 方差估计

PPI估计量的渐近方差为：

```
Var(μ̂_PPI) = σ²/n + σ²_ε/n - σ²_ε/N
```

其中 `σ²_ε` 是预测误差的方差。

## 预期结果

基于理论分析和实验，预期结果包括：

1. **PPI方法优势明显**：
   - 置信区间宽度比经典方法缩小 20-40%
   - 保持接近名义覆盖率（95%）
   - 偏差控制良好

2. **朴素ML方法问题**：
   - 可能产生显著偏差
   - 置信区间覆盖率不准确

3. **效率提升显著**：
   - 相当于将有效样本量提升 2-5 倍
   - 在保证统计有效性前提下大幅提升推断精度

## 技术特色

### 🔬 严谨的统计理论
- 基于渐近统计理论的置信区间构建
- 多种偏差校正机制
- 覆盖率保证

### 🎨 学术级可视化
- 现代化配色方案
- 高分辨率图表输出
- 符合顶级期刊标准

### 🛠️ 模块化设计
- 清晰的代码结构
- 易于扩展和修改
- 完整的文档注释

### 📊 全面的评估
- 多维度性能指标
- 统计显著性检验
- 稳健性分析

## 参考文献

1. Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T. (2023). **Prediction-powered inference**. *Science*, 382(6671), 669-674.

2. 更多相关文献请参见 `reference/reference.bib`

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目负责人：数据科学导论实验小组
- 邮箱：[your-email@example.com]
- 项目主页：[项目GitHub链接]

---

*本项目是数据科学导论课程的实验项目，旨在探索和评估现代统计推断方法在实际数据科学场景中的应用效果。* 