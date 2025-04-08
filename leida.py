import numpy as np
import matplotlib.pyplot as plt

# ===================
# 数据结构配置
# ===================
positions = [
    ('CO', 'MAE'), ('CO', 'RMSE'), ('CO', 'R²'),
    ('NOx', 'MAE'), ('NOx', 'RMSE'), ('NOx', 'R²')
]
labels = [f"{metric} ({gas})" for gas, metric in positions]

data = {
    'CIDA':   [0.8055, 1.1027, 0.3418, 2.9546, 3.7277, 0.6503],
    'SAD':    [0.6410, 0.8488, 0.5224, 2.2573, 2.7953, 0.6945],
    'VDI':    [0.6678, 0.8960, 0.4651, 2.1998, 2.7770, 0.7160],
    'NU':     [0.6686, 0.8862, 0.3028, 2.2970, 2.8905, 0.6487],
    'DGCR':   [0.5583, 0.7416, 0.6481, 1.5544, 1.9596, 0.8635]
}

# ===================
# 智能归一化函数
# ===================
def normalize(data_dict):
    ranges = {
        'CO': {
            'MAE': (0.5, 1.7, 'reverse'),
            'RMSE': (0.7, 2.1, 'reverse'),
            'R²': (0.3, 0.65, 'forward')
        },
        'NOx': {
            'MAE': (1.5, 5.8, 'reverse'),
            'RMSE': (2.0, 7.0, 'reverse'),
            'R²': (0.59, 0.86, 'forward')
        }
    }
    
    normalized = {}
    for method, values in data_dict.items():
        normalized_values = []
        for i, val in enumerate(values):
            gas, metric = positions[i]
            vmin, vmax, dir_type = ranges[gas][metric]
            
            if dir_type == 'reverse':
                norm = (vmax - val) / (vmax - vmin)
            else:
                norm = (val - vmin) / (vmax - vmin)
                
            normalized_values.append(max(0, min(1, norm)))
        
        normalized_values.append(normalized_values[0])
        normalized[method] = normalized_values
    return normalized

# ===================
# 优化后的雷达图可视化函数
# ===================
def plot_combined_radar(normalized_data):
    plt.figure(dpi=300)
    plt.rc('font', family='Times New Roman')
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # 角度计算
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    # 视觉样式配置
    colors = plt.cm.tab10(np.linspace(0, 1, len(normalized_data)))
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    
    # 绘制每个方法
    for (method, values), color, ls in zip(normalized_data.items(), colors, line_styles):
        ax.plot(angles, values, color=color, linestyle=ls, linewidth=3,
                label=method, marker='o', markersize=8, zorder=3)
        ax.fill(angles, values, color=color, alpha=0.05, zorder=2)
    
    # 坐标轴设置
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=20)
    
    # 径向网格优化
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8], 
                 labels=['20%', '40%', '60%', '80%'], 
                 angle=45, 
                 fontsize=14,
                 color='gray')
    
    # 图例和边框设置
    plt.legend(loc='best', bbox_to_anchor=(1.35, 0.3), fontsize=18, framealpha=0.95)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('gray')
    ax.spines['polar'].set_linewidth(0.8)
    ax.grid(True, linestyle='--', alpha=0.8, linewidth=0.8)
    
    plt.tight_layout()

# ===================
# 执行流程
# ===================
if __name__ == "__main__":
    normalized_data = normalize(data)
    plot_combined_radar(normalized_data)
    
    # 保存文件（请确认路径存在）
    plt.savefig("/root/dh/2025/OODCO/figs/leida.png", bbox_inches='tight')
    plt.savefig("/root/dh/2025/OODCO/figs/leida.eps", format='eps', bbox_inches='tight')
    plt.savefig("/root/dh/2025/OODCO/figs/leida.pdf", format='eps', bbox_inches='tight')