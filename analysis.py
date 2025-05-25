import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# load saved accuracy scores and test data
accuracies = joblib.load('ml_pickles/accuracies.pkl')
X_test_scaled, y_test = joblib.load('ml_pickles/test_data.pkl')

# mapping of model names to their saved files
model_files = {
    'Random Forest': 'ml_pickles/random_forest_model.pkl',
    'KNN': 'ml_pickles/knn_model.pkl',
    'Logistic Regression': 'ml_pickles/logistic_regression_model.pkl',
    'MLP Neural Net': 'ml_pickles/mlp_neural_net_model.pkl'
}

# plot accuracy scores for all models
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# plot confusion matrices side by side for all models
fig, axes = plt.subplots(1, len(model_files), figsize=(15, 5))
fig.suptitle('Confusion Matrices Comparison', fontsize=16)

for ax, (name, model_file) in zip(axes, model_files.items()):
    model = joblib.load(model_file)
    y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title(name)
    ax.grid(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# calculate classification report metrics for each model
metrics = ['precision', 'recall', 'f1-score']
classes = ['low', 'medium', 'high']

report_dfs = {}

for name, model_file in model_files.items():
    model = joblib.load(model_file)
    y_pred = model.predict(X_test_scaled)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    try:
        class_scores = report_df.loc[classes, metrics]
        report_dfs[name] = class_scores
    except KeyError:
        print(f'Warning: Some labels missing for model {name}. Skipping.')
        continue

# plot bar charts for each metric comparing all models
for metric in metrics:
    plt.figure(figsize=(15, 6))
    width = 0.2
    x = range(len(classes))
    
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    
    for i, (model_name, df) in enumerate(report_dfs.items()):
        values = df[metric].values
        plt.bar([pos + offsets[i] for pos in x], values, width=width, label=model_name)
    
    plt.xticks(ticks=x, labels=classes)
    plt.ylim(0, 1)
    plt.ylabel(metric.capitalize())
    plt.title(f'Comparison of {metric} Per Class For All Models')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()