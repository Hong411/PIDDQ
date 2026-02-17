# -*- coding: utf-8 -*-
"""
Centralized matplotlib configuration for consistent project-wide plotting
"""

import matplotlib.pyplot as plt

def set_plot_style():
    """Configure global matplotlib rcParams"""
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 18,
        'font.weight': 'bold',
        'axes.linewidth': 2,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'figure.autolayout': True,  # Enable automatic layout adjustments
        'figure.dpi': 300           # High resolution figures
    })
    
def set_natcomsci_style():
    plt.rcParams.update({
        'font.sans-serif': ['Arial', 'Helvetica'], # 使用要求字体
        'font.size': 7,                           # 正文文字大小 (pt)
        'axes.titlesize': 7,                      # 子图标题大小
        'axes.labelsize': 7,                      # 坐标轴标签大小
        'xtick.labelsize': 6,                     # X轴刻度标签，可略小于正文
        'ytick.labelsize': 6,                     # Y轴刻度标签
        'legend.fontsize': 6,                     # 图例字体大小
        'figure.dpi': 300,                        # 导出时DPI，影响嵌入的位图
        'savefig.dpi': 300,
        'pdf.fonttype': 42,                       # 确保字体嵌入PDF并可编辑 (重要!)
        'ps.fonttype': 42,
        'figure.constrained_layout.use': True     # 自动调整布局，避免标签重叠
    })

