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

# Handle missing values in y_train (choose one of the methods mentioned above)
# For example, removing rows with missing values:
X_train = X_train[~y_train.isnull()]
y_train = y_train.dropna()

# Define the XGBoost Classifier
xgb_classifier = XGBClassifier(
    learning_rate =0.2,
    n_estimators=70,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    device = "cuda",
    tree_method = "hist",
    nthread=64,
    seed=27
)

# Define a parameter grid for Grid Search
param_grid = {
    'max_depth':[5, 8, 10, 13, 15, 18, 20, 22, 25],
    'min_child_weight':[4,5,6,8,10,12]
}

# Create a Grid Search with cross-validation
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, scoring='accuracy', cv=5, verbose=2, return_train_score=True)

# Fit the Grid Search to your training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params_1 = grid_search.best_params_
best_score_1 = grid_search.best_score_
print("Best Parameters:", best_params_1)
print("Best Accuracy Score:", best_score_1)

# Print the scores for all parameter combinations
cv_results = grid_search.cv_results_
for mean_test_score, std_test_score, mean_train_score, params in zip(
    cv_results['mean_test_score'],
    cv_results['std_test_score'],
    cv_results['mean_train_score'],
    cv_results['params']
):
    print("Parameters:", params)
    print(f"Mean Test Score: {mean_test_score:.4f}")
    print(f"Std Test Score: {std_test_score:.4f}")
    print(f"Mean Train Score: {mean_train_score:.4f}")
    

# Define the XGBoost Classifier
xgb_classifier = XGBClassifier(
    learning_rate=0.2,
    n_estimators=70,
    max_depth=best_params_1['max_depth'],
    min_child_weight=best_params_1['min_child_weight'],
    device = "cuda",
    tree_method = "hist",
    subsample=0.8,
    colsample_bytree=0.8,
    nthread=64,
    seed=27
)

# Define a parameter grid for Grid Search
param_grid = {
    'gamma':[i/10.0 for i in range(0,11)]
}

# Create a Grid Search with cross-validation
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, scoring='accuracy', cv=5, verbose=2, return_train_score=True)

# Fit the Grid Search to your training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params_2 = grid_search.best_params_
best_score_2 = grid_search.best_score_
print("Best Parameters:", best_params_2)
print("Best Accuracy Score:", best_score_2)

# Print the accuracy for all parameter combinations
results = grid_search.cv_results_
for params, accuracy in zip(results['params'], results['mean_test_score']):
    print(f"Parameters: {params}, Accuracy: {accuracy:.4f}")

# Define the XGBoost Classifier
xgb_classifier = XGBClassifier(
    learning_rate =0.1,
    n_estimators=100,
    max_depth=best_params_1['max_depth'],
    min_child_weight=best_params_1['min_child_weight'],
    gamma=best_params_2['gamma'],
    device = "cuda",
    tree_method = "hist",
    subsample=0.8,
    colsample_bytree=0.8,
    nthread=64,
    seed=27
)

# Define a parameter grid for Grid Search
param_grid = {
    'subsample':[i/10.0 for i in range(5,10)],
    'colsample_bytree':[i/10.0 for i in range(3,10)]
}

# Create a Grid Search with cross-validation
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, scoring='accuracy', cv=5, verbose=2, return_train_score=True)

# Fit the Grid Search to your training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params_3 = grid_search.best_params_
best_score_3 = grid_search.best_score_
print("Best Parameters:", best_params_3)
print("Best Accuracy Score:", best_score_3)

# Print the scores for all parameter combinations
cv_results = grid_search.cv_results_
for mean_test_score, std_test_score, mean_train_score, params in zip(
    cv_results['mean_test_score'],
    cv_results['std_test_score'],
    cv_results['mean_train_score'],
    cv_results['params']
):
    print("Parameters:", params)
    print(f"Mean Test Score: {mean_test_score:.4f}")
    print(f"Std Test Score: {std_test_score:.4f}")
    print(f"Mean Train Score: {mean_train_score:.4f}")

# Define the XGBoost Classifier
xgb_classifier = XGBClassifier(
    learning_rate =0.1,
    n_estimators=100,
    max_depth=best_params_1['max_depth'],
    min_child_weight=best_params_1['min_child_weight'],
    gamma=best_params_2['gamma'],
    subsample=best_params_3['subsample'],
    colsample_bytree=best_params_3['colsample_bytree'],
    device = "cuda",
    tree_method = "hist",
    nthread=64,
    seed=27
)

# Define a parameter grid for Grid Search
param_grid = {
    'reg_alpha':[1e-5, 1e-2,0.05, 0.1, 0.5, 1, 100]
}

# Create a Grid Search with cross-validation
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, scoring='accuracy', cv=5, verbose=2, return_train_score=True)

# Fit the Grid Search to your training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params_4 = grid_search.best_params_
best_score_4 = grid_search.best_score_
print("Best Parameters:", best_params_4)
print("Best Accuracy Score:", best_score_4)

# Print the scores for all parameter combinations
cv_results = grid_search.cv_results_
for mean_test_score, std_test_score, mean_train_score, params in zip(
    cv_results['mean_test_score'],
    cv_results['std_test_score'],
    cv_results['mean_train_score'],
    cv_results['params']
):
    print("Parameters:", params)
    print(f"Mean Test Score: {mean_test_score:.4f}")
    print(f"Std Test Score: {std_test_score:.4f}")
    print(f"Mean Train Score: {mean_train_score:.4f}")
    
# Function to determine the value mapping
def determine_mapping(values):
    unique_values = sorted(dataset['Minimum Installs'].unique())
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    return mapping

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
final_xgb_model = XGBClassifier(
    n_estimators=1000,
    max_depth=best_params_1['max_depth'],
    min_child_weight=best_params_1['min_child_weight'],
    gamma=best_params_2['gamma'],
    subsample=best_params_3['subsample'],
    colsample_bytree=best_params_3['colsample_bytree'],
    reg_alpha=best_params_4['reg_alpha'],
    device = "cuda",
    tree_method = "hist",
    learning_rate=0.1,
    nthread=64,
    seed=27
)
final_xgb_model.fit(X_train, y_train)

# Evaluate the final model on the test data
y_pred = final_xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Data:", accuracy)

# Train the final model with the best parameters
final_final_xgb_model = XGBClassifier(
    n_estimators=5000,
    max_depth=best_params_1['max_depth'],
    min_child_weight=best_params_1['min_child_weight'],
    gamma=best_params_2['gamma'],
    subsample=best_params_3['subsample'],
    colsample_bytree=best_params_3['colsample_bytree'],
    reg_alpha=best_params_4['reg_alpha'],
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