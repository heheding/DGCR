import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import *

# 读取五个模型的预测结果csv文件和标签csv文件
df_cida = pd.read_csv('/root/dh/2024/OODCo/out/CIDA.csv').values
df_vdi = pd.read_csv('/root/dh/2024/OODCo/out/VDI.csv').values
df_sad = pd.read_csv('/root/dh/2024/OODCo/out/SAD.csv').values
df_nu = pd.read_csv('/root/dh/2024/OODCo/out/NU.csv').values
df_dgcr = pd.read_csv('/root/dh/2025/OODCO/out/DGCR.csv').values
df_label = pd.read_csv('/root/dh/2024/OODCo/out/Label.csv').values
df_label, df_cida, df_dgcr, df_sad, df_vdi, df_nu = df_label.squeeze(-1), df_cida.squeeze(-1), df_dgcr.squeeze(-1), df_sad.squeeze(-1), df_vdi.squeeze(-1), df_nu.squeeze(-1)

# 调用函数进行绘图
errplt(df_label, df_cida, df_dgcr, df_sad, df_vdi, df_nu)

curves(df_label, df_cida, df_dgcr, df_sad, df_vdi, df_nu)

curves2(df_label, df_dgcr)

plot_boxplot(df_label, df_cida, df_dgcr, df_sad, df_vdi, df_nu)

scatter(df_label, df_cida, df_dgcr, df_sad, df_vdi, df_nu)
