# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.metrics import (
#     roc_curve,
#     roc_auc_score,
#     precision_recall_curve,
#     accuracy_score,
#     f1_score,
#     auc
# )

# # =====================================================
# # CONFIGURATION
# # =====================================================

# BASE_DIR = "model_outputs"

# MODEL_FILES = {
#     "Logistic Regression": f"{BASE_DIR}/logisticregression_outputs/test_predictions.csv",
#     "KNN": f"{BASE_DIR}/knn_outputs/test_predictions_knn.csv",
#     "Random Forest": f"{BASE_DIR}/random_forest_strict/test_predictions.csv",
#     "XGBoost": f"{BASE_DIR}/xgboost_outputs/test_predictions_xgb.csv"
# }

# OUTPUT_DIR = "comparative_outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # =====================================================
# # LOAD ALL MODEL OUTPUTS
# # =====================================================

# model_outputs = {}

# for model_name, file_path in MODEL_FILES.items():
#     df = pd.read_csv(file_path)

#     # Standard columns (as you confirmed)
#     y_true = df["actual"]
#     y_proba = df["probability"]

#     model_outputs[model_name] = {
#         "y_true": y_true,
#         "y_proba": y_proba
#     }

# print(" All model prediction files loaded successfully")

# # =====================================================
# # ROC CURVE COMPARISON (ALL MODELS IN ONE IMAGE)
# # =====================================================

# plt.figure(figsize=(8, 6))

# for model_name, data in model_outputs.items():
#     fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
#     auc_score = roc_auc_score(data["y_true"], data["y_proba"])

#     plt.plot(
#         fpr,
#         tpr,
#         linewidth=2,
#         label=f"{model_name} (AUC = {auc_score:.3f})"
#     )

# plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve Comparison – All Models")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/roc_curve_comparison.png", dpi=300)
# plt.show()

# # =====================================================
# # PRECISION–RECALL CURVE COMPARISON
# # =====================================================

# plt.figure(figsize=(8, 6))

# for model_name, data in model_outputs.items():
#     precision, recall, _ = precision_recall_curve(
#         data["y_true"], data["y_proba"]
#     )
#     pr_auc = auc(recall, precision)

#     plt.plot(
#         recall,
#         precision,
#         linewidth=2,
#         label=f"{model_name} (AUC = {pr_auc:.3f})"
#     )

# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision–Recall Curve Comparison")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/pr_curve_comparison.png", dpi=300)
# plt.show()

# # =====================================================
# # ACCURACY / F1 / ROC-AUC BAR COMPARISON
# # =====================================================

# rows = []

# for model_name, data in model_outputs.items():
#     y_pred = (data["y_proba"] >= 0.5).astype(int)

#     rows.append({
#         "Model": model_name,
#         "Accuracy": accuracy_score(data["y_true"], y_pred),
#         "F1 Score": f1_score(data["y_true"], y_pred),
#         "ROC-AUC": roc_auc_score(data["y_true"], data["y_proba"])
#     })

# summary_df = pd.DataFrame(rows)

# summary_df.set_index("Model").plot(
#     kind="bar",
#     figsize=(9, 6)
# )

# plt.title("Model Performance Comparison")
# plt.ylabel("Score")
# plt.ylim(0, 1)
# plt.grid(axis="y", alpha=0.3)
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/model_comparison_bar.png", dpi=300)
# plt.show()

# summary_df.to_csv(
#     f"{OUTPUT_DIR}/model_comparison_summary.csv",
#     index=False
# )

# print("\n FINAL COMPARISON SUMMARY\n")
# print(summary_df)


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import (
#     confusion_matrix,
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score
# )

# # ======================================================
# # MODEL OUTPUT CSV PATHS
# # ======================================================

# model_files = {
#     "Logistic Regression": "model_outputs/logisticregression_outputs/test_predictions.csv",
#     "KNN": "model_outputs/knn_outputs/test_predictions_knn.csv",
#     "Random Forest": "model_outputs/random_forest_strict/test_predictions.csv",
#     "XGBoost": "model_outputs/xgboost_outputs/test_predictions_xgb.csv"
# }

# # ======================================================
# # COLLECT RESULTS
# # ======================================================

# confusion_rows = ["TP", "FP", "TN", "FN"]
# metric_rows = ["Accuracy", "Precision", "Recall", "F1-score", "AUROC"]

# confusion_data = {}
# metric_data = {}

# for model, path in model_files.items():
#     df = pd.read_csv(path)

#     y_true = df["actual"]
#     y_pred = df["predicted"]
#     y_prob = df["probability"]

#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

#     confusion_data[model] = [tp, fp, tn, fn]
#     metric_data[model] = [
#         round(accuracy_score(y_true, y_pred), 2),
#         round(precision_score(y_true, y_pred), 2),
#         round(recall_score(y_true, y_pred), 2),
#         round(f1_score(y_true, y_pred), 2),
#         round(roc_auc_score(y_true, y_prob), 2)
#     ]

# conf_df = pd.DataFrame(confusion_data, index=confusion_rows)
# metric_df = pd.DataFrame(metric_data, index=metric_rows)

# # 👇 Spacer row to create TWO BLOCKS
# separator = pd.DataFrame(
#     [[""] * len(model_files)],
#     columns=model_files.keys(),
#     index=["Evaluation Metrics"]
# )

# final_df = pd.concat([conf_df, separator, metric_df])

# # ======================================================
# # DRAW TABLE IMAGE
# # ======================================================

# fig, ax = plt.subplots(figsize=(16, 8))
# ax.axis("off")

# table = ax.table(
#     cellText=final_df.values,
#     rowLabels=final_df.index,
#     colLabels=final_df.columns,
#     cellLoc="center",
#     loc="center"
# )

# table.auto_set_font_size(False)
# table.set_fontsize(11)
# table.scale(1.3, 1.8)

# n_rows, n_cols = final_df.shape

# # ======================================================
# # STYLING (LIKE RESEARCH PAPER)
# # ======================================================

# for (row, col), cell in table.get_celld().items():
#     cell.set_edgecolor("black")

#     # Header row
#     if row == 0:
#         cell.set_facecolor("#D9D9D9")
#         cell.set_text_props(weight="bold")
#         cell.set_linewidth(1.8)

#     # Separator row (between blocks)
#     elif final_df.index[row - 1] == "Evaluation Metrics":
#         cell.set_text_props(weight="bold")
#         cell.set_facecolor("#F2F2F2")
#         cell.set_linewidth(2.2)
#         if col == -1:
#             cell.get_text().set_text("Evaluation Metrics")

#     else:
#         cell.set_linewidth(1.2)

# # ======================================================
# # STRONG OUTER BORDER
# # ======================================================

# for col in range(n_cols):
#     table[(0, col)].set_linewidth(2.0)
#     table[(n_rows, col)].set_linewidth(2.0)

# for row in range(n_rows + 1):
#     table[(row, 0)].set_linewidth(2.0)
#     table[(row, n_cols - 1)].set_linewidth(2.0)

# # ======================================================
# # TITLES
# # ======================================================

# plt.text(
#     0.5, 0.96,
#     "Table: Comparative Analysis of Models using Confusion Matrix and Evaluation Metrics",
#     ha="center",
#     va="center",
#     fontsize=14,
#     weight="bold",
#     transform=ax.transAxes
# )

# # ======================================================
# # SAVE IMAGE
# # ======================================================

# plt.savefig(
#     "comparative_outputs/model_comparison_table_blocks.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()



# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score
# )

# # ============================================================
# # OUTPUT DIRECTORY
# # ============================================================
# OUTPUT_DIR = "comparative_outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ============================================================
# # LOAD MODEL OUTPUT FILES
# # ============================================================
# model_files = {
#     "Logistic Regression": "model_outputs/logisticregression_outputs/test_predictions.csv",
#     "KNN": "model_outputs/knn_outputs/test_predictions_knn.csv",
#     "Random Forest": "model_outputs/random_forest_strict/test_predictions.csv",
#     "XGBoost": "model_outputs/xgboost_outputs/test_predictions_xgb.csv"
# }

# # ============================================================
# # COMPUTE EVALUATION METRICS
# # ============================================================
# rows = []

# for model_name, file_path in model_files.items():
#     df = pd.read_csv(file_path)

#     y_true = df["actual"]
#     y_pred = df["predicted"]
#     y_proba = df["probability"]

#     rows.append([
#         model_name,
#         accuracy_score(y_true, y_pred),
#         precision_score(y_true, y_pred),
#         recall_score(y_true, y_pred),
#         f1_score(y_true, y_pred),
#         roc_auc_score(y_true, y_proba)
#     ])

# metrics_df = pd.DataFrame(
#     rows,
#     columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
# )

# metrics_df.set_index("Model", inplace=True)
# metrics_df = metrics_df.round(3)

# # ============================================================
# # CREATE TABLE IMAGE
# # ============================================================
# fig, ax = plt.subplots(figsize=(11, 4))
# ax.axis("off")

# table = ax.table(
#     cellText=metrics_df.values,
#     colLabels=metrics_df.columns,
#     rowLabels=metrics_df.index,
#     cellLoc="center",
#     loc="center"
# )

# # ============================================================
# # STYLING
# # ============================================================
# table.auto_set_font_size(False)
# table.set_fontsize(11)
# table.scale(1, 1.8)

# # Header styling
# for col in range(len(metrics_df.columns)):
#     cell = table[(0, col)]
#     cell.set_facecolor("#1f2937")   # dark slate
#     cell.set_text_props(color="white", weight="bold")

# # Row label styling
# for row in range(1, len(metrics_df.index) + 1):
#     cell = table[(row, -1)]
#     cell.set_facecolor("#374151")
#     cell.set_text_props(color="white", weight="bold")

# # Cell color gradients (performance-based)
# for row in range(1, len(metrics_df.index) + 1):
#     for col in range(len(metrics_df.columns)):
#         value = metrics_df.iloc[row - 1, col]

#         if value >= 0.85:
#             color = "#bbf7d0"  # green
#         elif value >= 0.70:
#             color = "#fef9c3"  # yellow
#         else:
#             color = "#fecaca"  # red

#         table[(row, col)].set_facecolor(color)

# # ============================================================
# # TITLE
# # ============================================================
# plt.title(
#     "Model Evaluation Metrics Comparison",
#     fontsize=14,
#     weight="bold",
#     pad=20
# )

# # ============================================================
# # SAVE IMAGE
# # ============================================================
# plt.savefig(
#     os.path.join(OUTPUT_DIR, "evaluation_metrics_comparison_table.png"),
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.show()

# print("✅ Evaluation metrics table saved successfully!")
