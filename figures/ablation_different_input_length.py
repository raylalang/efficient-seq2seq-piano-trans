# Model, Sec. per Token, Onset F-Score
# baseline, 21.82, 96.57
# Local Enc. Attn. , 36.28, 96.45
# Hierarchical Pool , 38.12, 96.13
# Local Enc. & Dec. Attn. , 41.38, 95.20


# {
#     "baseline": 21.82,
#     "Local Enc. Attn.": 38.12,
#     "Hierarchical Pool": 41.38,
    
# }

import matplotlib.pyplot as plt
import numpy as np

# baseline, local encoder attention, local encoder and hierarchical pooling, local encoder & decoder attention and hierarchical pooling
# 数据
# models = ['Baseline', 'Enc-LocAttn', 'Enc-LocAttn +\n Dec-LocAttn', 'Enc-LocAttn +\n Dec-LocAttn +\n EncDec-HybridAttn', 'Enc-LocAttn +\n Dec-LocAttn +\n EncDec-HybridAttn + \n Fixed Pooling', 'Enc-LocAttn +\n Dec-LocAttn +\n EncDec-HybridAttn + \n Hierarchical Pooling' ]
models = ["Baseline", "V1", "V2", "V3", "V4", "V5"]
# tokens_per_sec = [21.82, 36.28, 38.12, 41.38]

batch_size=64
tokens_per_sec_baseline = [38.04, 22.25, 14.84]
memory_usage_baseline = [4.91, 10.95, 19.01]

tokens_per_sec_v5 = [80.75, 66.74, 53.92]
memory_usage_v5 = [2.89, 7.17, 10.15]

# baseline, local encoder attention, local encoder and hierarchical pooling, local encoder & decoder attention and hierarchical pooling
# tokens_per_sec = [20.86, , 23.76, 30.40]
# batch_size 64
# [,, , ,48.34]
# batchsize 256
# [10.48, , , 17.71]
#  batch_size 320
# [ 8.60, , 8.83, 13.04]
onset_f_score = [96.86, 96.72, 96.76, 96.57, 96.19, 96.60]

# 创建图形和主坐标轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 设置x轴位置
x_pos = np.arange(len(models))

# 绘制第一条线 - Tokens per Second
color1 = 'tab:blue'
# ax1.set_xlabel('Model Category', fontsize=12, fontweight='bold')
ax1.set_ylabel('Inference Speed (iterations/s)', color=color1, fontsize=15, fontweight='bold')
line1 = ax1.plot(x_pos, tokens_per_sec, color=color1, marker='o', linewidth=2, 
                 markersize=8, label='Inference Speed')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 65)  # 设置y轴范围以便更好地显示速度

# 创建第二个y轴
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Onset F-Score (%)', color=color2, fontsize=15, fontweight='bold')
line2 = ax2.plot(x_pos, onset_f_score, color=color2, marker='s', linewidth=2, 
                 markersize=8, label='Onset F-Score')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(80, 100)  # 设置y轴范围以便更好地显示F-score

# 创建第三个y轴
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # 将第三个y轴向右移动60个点
color3 = 'tab:green'
ax3.set_ylabel('GPU Memory Usage (GB)', color=color3, fontsize=15, fontweight='bold')
line3 = ax3.plot(x_pos, memory_usage, color=color3, marker='^', linewidth=2, 
                markersize=8, label='GPU Memory Usage')
ax3.tick_params(axis='y', labelcolor=color3)
ax3.set_ylim(6, 22)  # 设置y轴范围以便更好地显示内存使用情况

# 设置x轴标签
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, rotation=0, ha='center', fontsize=15) #, rotation=15

# 添加数值标签
for i, (tps, ofs) in enumerate(zip(tokens_per_sec, onset_f_score)):
    ax1.annotate(f'{tps:.2f}', (i, tps), textcoords="offset points", 
                xytext=(0,10), ha='center', color=color1, fontweight='bold')
    ax2.annotate(f'{ofs:.2f}', (i, ofs), textcoords="offset points", 
                xytext=(0,-15), ha='center', color=color2, fontweight='bold')
    ax3.annotate(f'{memory_usage[i]:.2f}', (i, memory_usage[i]), textcoords="offset points", 
                xytext=(0,10), ha='center', color=color3, fontweight='bold')

# 添加标题
# plt.title('Inference Speed vs Transcription Performance',  # Model Performance: 
#           fontsize=18, fontweight='bold', pad=20)

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='center right', 
           bbox_to_anchor=(0.98, 0.9))

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
plt.savefig('figures/tokens_per_second_v2.pdf', dpi=300, bbox_inches='tight')

# 打印数据摘要
print("数据摘要:")
print("-" * 50)
for i, model in enumerate(models):
    print(f"{model:20s} | {tokens_per_sec[i]:8.3f} tokens/sec | {onset_f_score[i]:6.2f} F-score")