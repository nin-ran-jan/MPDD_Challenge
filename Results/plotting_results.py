attention_fusion_lstm = 0.800677966101695
attention_fusion_mlp = 0.4531897265948633
early_fusion_lstm = 0.7393939393939394
early_fusion_mlp = 0.57
late_fusion_lstm = 0.7185501066098081
late_fusion_mlp = 0.3980861244019139
late_fusion_svm = 0.44

import matplotlib.pyplot as plt

# Labels and Scores
models = ['Early Fusion MLP', 'Late Fusion MLP', 'Attention Fusion MLP', 'Late Fusion SVM']
f1_scores = [early_fusion_mlp, late_fusion_mlp, attention_fusion_mlp, late_fusion_svm]

# Plotting
plt.figure(figsize=(8, 5))
bars = plt.bar(models, f1_scores, color=['skyblue', 'lightcoral', 'mediumpurple', 'lightgreen'])

# Annotate values on top
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')

plt.ylabel('Macro F1 Score')
plt.title('Macro F1 Scores: MLPs and SVM (Late Fusion)')
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('mlp_svm_f1_scores.png', dpi=300)
plt.close()


import matplotlib.pyplot as plt

# Labels and Scores for LSTM models
models = ['Early Fusion LSTM', 'Late Fusion LSTM', 'Attention Fusion LSTM']
f1_scores = [early_fusion_lstm, late_fusion_lstm, attention_fusion_lstm]

# Plotting
plt.figure(figsize=(8, 5))
bars = plt.bar(models, f1_scores, color=['skyblue', 'lightcoral', 'mediumpurple'])

# Annotate values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')

plt.ylabel('Macro F1 Score')
plt.title('Macro F1 Scores: LSTM Models')
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure
plt.savefig('lstm_f1_scores.png', dpi=300)
plt.close()
