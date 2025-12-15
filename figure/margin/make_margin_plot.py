###########################################################################################################################
# margin_on_training.py를 실행한 뒤, 생성된 csv 파일을 연결하여 그래프를 그리는 파일입니다. 
# 생성된 csv 파일이 없으면, 동작하지 못합니다. 
###########################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# CSV 파일 경로
csv_path_effi = './eff_stats_per_epoch.csv' #margin_on_training.py의 train_output_dir/아래에 있는 csv 파일 경로로 바꾸어야함
csv_path_our  = './margin_stats_per_epoch.csv'

# 저장할 그래프 경로
out_dir  = './plot'
os.makedirs(out_dir, exist_ok=True)
save_path = os.path.join(out_dir, 'margin_per_epoch_comparison.png')

# 전체 폰트 설정
plt.rcParams['font.family']  = 'serif'
plt.rcParams['font.serif']   = ['Times New Roman'] + plt.rcParams.get('font.serif', [])
plt.rcParams['font.size']    = 25

# x축 고정값
fixed_x = [1, 20, 40, 60, 80, 100]

# 정규화 없이 raw 값만 반환
def prepare_series(csv_path):
    df   = pd.read_csv(csv_path)
    df0  = df.iloc[[0]].copy()
    df10 = df.iloc[1::20].copy()
    dfp  = pd.concat([df0, df10], ignore_index=True)
    n    = min(len(dfp), len(fixed_x))
    x    = fixed_x[:n]
    ytr  = dfp['train_mean'].iloc[:n].values
    yte  = dfp['test_mean'].iloc[:n].values
    return x, ytr, yte

# 데이터 준비
x_e, ytr_e, yte_e = prepare_series(csv_path_effi)
x_o, ytr_o, yte_o = prepare_series(csv_path_our)

# 전체 범위에서 y축 최소/최대값 계산
y_min = min(np.min(ytr_e), np.min(yte_e), np.min(ytr_o), np.min(yte_o))
y_max = max(np.max(ytr_e), np.max(yte_e), np.max(ytr_o), np.max(yte_o))
y_margin = 0.05 * (y_max - y_min)
y_lower, y_upper = y_min - y_margin, y_max + y_margin

# ================================
# EfficientAD 그래프 저장
# ================================
fig_e, ax_e = plt.subplots(figsize=(9, 7))  # 단일 그래프 사이즈 조정
ax_e.plot(x_e, ytr_e, marker='o', markersize=12, linewidth=5, color="#FF9AA9")
ax_e.plot(x_e, yte_e, marker='s', markersize=12, linewidth=5, color="#74C5F8", linestyle='--')
ax_e.plot([0.68, 0.73], [0.23, 0.23], color='#FF9AA9', linewidth=6, transform=ax_e.transAxes, solid_capstyle='round')
ax_e.text(0.75, 0.25, 'Train mean', transform=ax_e.transAxes, ha='left', va='top', fontsize=28)
ax_e.plot([0.68, 0.73], [0.165, 0.165], color='#74C5F8', linewidth=6, transform=ax_e.transAxes, solid_capstyle='round')
ax_e.text(0.75, 0.185, 'Test mean', transform=ax_e.transAxes, ha='left', va='top', fontsize=28)
ax_e.set_title('EfficientAD')
ax_e.set_xlabel('Epoch')
ax_e.set_ylabel('Margin', fontsize=28)
ax_e.set_xticks(fixed_x)
ax_e.set_yticks([])
ax_e.set_ylim(y_lower, y_upper)  # y축 고정
ax_e.grid(True, color='#EEEEEE')
for spine in ax_e.spines.values():
    spine.set_edgecolor("#A8A8A8")
    spine.set_linewidth(1)
fig_e.tight_layout()
fig_e.savefig(os.path.join(out_dir, 'figure_2_eff_no_scale.pdf'), dpi=600)


# ================================
# Ours
# ================================
fig_o, ax_o = plt.subplots(figsize=(9, 7))  # 동일 사이즈
ax_o.plot(x_o, ytr_o, marker='o', markersize=12, linewidth=5, color="#FF9AA9")
ax_o.plot(x_o, yte_o, marker='s', markersize=12, linewidth=5, color="#74C5F8", linestyle='--')
ax_o.plot([0.68, 0.73], [0.23, 0.23], color='#FF9AA9', linewidth=6, transform=ax_o.transAxes, solid_capstyle='round')
ax_o.text(0.75, 0.25, 'Train mean', transform=ax_o.transAxes, ha='left', va='top', fontsize=28)
ax_o.plot([0.68, 0.73], [0.165, 0.165], color='#74C5F8', linewidth=6, transform=ax_o.transAxes, solid_capstyle='round')
ax_o.text(0.77, 0.185, 'Test mean', transform=ax_o.transAxes, ha='left', va='top', fontsize=28)
ax_o.set_title('FALCON')
ax_o.set_xlabel('Epoch')
ax_o.set_ylabel('Margin', fontsize=28)
ax_o.set_xticks(fixed_x)
ax_o.set_yticks([])
ax_o.set_ylim(y_lower, y_upper)  # y축 고정
ax_o.grid(True, color='#EEEEEE')
for spine in ax_o.spines.values():
    spine.set_edgecolor("#A8A8A8")
    spine.set_linewidth(1)
fig_o.tight_layout()
fig_o.savefig(os.path.join(out_dir, 'figure_2_our_no_scale.pdf'), dpi=600)
