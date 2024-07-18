import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ['MLP/30chs', 'CNN-GRU/30chs', 'EEGNet/30chs', 'AITST/30chs', 'MD-CAT/30chs', 'Ours/30chs', 'Ours(EQVA)/62+30chs']
OA = [85.37, 92.43, 38.52, 83.60, 96.93, 97.81, 98.61]
OA_SEED = [94.55, 97.05, 44.75, 80.37, 97.62, 99.60, 99.89]
OA_FACED = [59.02, 79.19, 20.65, 92.85, 94.96, 92.68, 94.96]
BA = [62.89, 81.13, 23.90, 91.66, 95.25, 93.44, 95.49]

x = np.arange(len(methods))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - 1.5*width, OA, width, label='OA')
rects2 = ax.bar(x - 0.5*width, OA_SEED, width, label='OA-SEED')
rects3 = ax.bar(x + 0.5*width, OA_FACED, width, label='OA-FACED')
rects4 = ax.bar(x + 1.5*width, BA, width, label='BA')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Methods')
ax.set_ylabel('Performance (%)')
ax.set_title('Comparison of Methods by Different Metrics')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()

# Add bar labels
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

plt.show()