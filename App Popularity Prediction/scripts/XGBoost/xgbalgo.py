#!/usr/bin/python3
import xgboost as xgb
from pandas import read_csv
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score


# Function to determine the value mapping
def determine_mapping(values):
    unique_values = sorted(dataset['Minimum Installs'].unique())
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    return mapping

dataset = read_csv("./preprocessed.csv")

# Get the mapping dictionary
value_mapping = determine_mapping(dataset['Minimum Installs'])

# Use the replace method to map the values in the 'values' column
dataset['Minimum Installs'] = dataset['Minimum Installs'].replace(value_mapping)
unique_values = dataset['Minimum Installs'].unique()
sorted_unique_values = sorted(unique_values)

# Define the target variable
target = 'Minimum Installs'

# List of predictors (exclude the target column)
predictors = [col for col in dataset.columns if col != 'Minimum Installs']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset[predictors],
                                                    dataset['Minimum Installs'],
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=dataset['Minimum Installs'])

# Train the final model with the best parameters
final_final_xgb_model = XGBClassifier(
    n_estimators=5000,
    max_depth=13,
    min_child_weight=12,
    gamma=0.9,
    subsample=0.9,
    colsample_bytree=0.6,
    reg_alpha=0.01,
    device = "cuda",
    tree_method = "hist",
    learning_rate=0.01,
    nthread=64,
    seed=27
)
final_final_xgb_model.fit(X_train, y_train)

# Evaluate the final model on the test data
y_pred = final_final_xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Data:", accuracy)

# Train
y_train_pred = final_final_xgb_model.predict(X_train)
test_accuracy = accuracy_score(y_train, y_train_pred)
print("Train accuracy:", test_accuracy)

# Confusion Matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_mat)

# Classification Report
class_report = classification_report(y_test, y_pred, output_dict=True)
print("Classification Report:\n", class_report)

# Initialize an array to store FPR, TPR, and AUC for each class
class_fpr = dict()
class_tpr = dict()
class_roc_auc = dict()

# Iterate over each class
for i in range(n_classes):  # n_classes is the number of classes
    fpr, tpr, _ = roc_curve(y_test[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    class_fpr[i] = fpr
    class_tpr[i] = tpr
    class_roc_auc[i] = roc_auc

# Print FPR, TPR, and AUC for each class
for i in range(n_classes):
    print(f"Class {i} - FPR: {class_fpr[i]}, TPR: {class_tpr[i]}, AUC: {class_roc_auc[i]}")

micro_fpr, micro_tpr, _ = roc_curve(y_test.ravel(), y_probs.ravel())
micro_roc_auc = auc(micro_fpr, micro_tpr)
macro_roc_auc = roc_auc_score(y_test, y_probs, average='macro')

print("Micro-Averaged AUC:", micro_roc_auc)
print("Macro-Averaged AUC:", macro_roc_auc)