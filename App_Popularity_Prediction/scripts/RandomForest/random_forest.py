#!/usr/bin/python3

from pandas                  import read_csv
from sklearn                 import tree
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score, classification_report, confusion_matrix
import gc

data = read_csv("./sampled_data_1000000.csv")

# Split the data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(data.drop('Minimum Installs', axis=1),
                                                 data['Minimum Installs'],
                                                 test_size=0.2,
                                                 random_state=42,
                                                 stratify=data['Minimum Installs']  # Stratified split for class balance
                                                 )

stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

# Define a parameter grid for Grid Search
param_grid = {
    'max_depth': [5, 10, 15, 18, 20, 22, 23],
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=stratified_cv, n_jobs=-1, verbose=2, return_train_score=True)
grid_search.fit(X_train, y_train)


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

# Get the best max_depth from Grid Search
best_max_depth = grid_search.best_params_['max_depth']
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

del rf_model
del grid_search
del results
gc.collect()

#----------------------------------------------------------------------------------------

rf_model = RandomForestClassifier(max_depth=best_max_depth, random_state=42, n_jobs=-1)

#Retrain Again
param_grid2 = {
    'n_estimators': [50, 100, 500, 1000]
}

grid_search2 = GridSearchCV(estimator=rf_model, param_grid=param_grid2, cv=stratified_cv, n_jobs=-1, verbose=2, return_train_score=True)
grid_search2.fit(X_train, y_train)

# Print the scores for all parameter combinations
cv_results = grid_search2.cv_results_
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

# Get the best n_estimators from Grid Search
best_n_estimators = grid_search2.best_params_['n_estimators']
best_params = grid_search2.best_params_
print("Best Parameters:", best_params)

del rf_model
del grid_search2
del results
gc.collect()

#----------------------------------------------------------------------------------------

rf_model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=42, n_jobs=-1)

#Retrain Again
param_grid3 = {
    'min_samples_split': [2, 5, 8]
}

grid_search3 = GridSearchCV(estimator=rf_model, param_grid=param_grid3, cv=stratified_cv, n_jobs=-1, verbose=2, return_train_score=True)
grid_search3.fit(X_train, y_train)

# Print the scores for all parameter combinations
cv_results = grid_search3.cv_results_
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

# Get the best n_estimators from Grid Search
best_min_sample_split = grid_search3.best_params_['min_samples_split']
best_params = grid_search3.best_params_
print("Best Parameters:", best_params)

del rf_model
del grid_search3
del results
gc.collect()

#----------------------------------------------------------------------------------------

rf_model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, min_samples_split=best_min_sample_split, random_state=42, n_jobs=-1)

#Retrain Again
param_grid4 = {
    'min_samples_leaf': [1, 2, 4]
}

grid_search4 = GridSearchCV(estimator=rf_model, param_grid=param_grid4, cv=stratified_cv, n_jobs=-1, verbose=2, return_train_score=True)
grid_search4.fit(X_train, y_train)

# Print the scores for all parameter combinations
cv_results = grid_search4.cv_results_
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

# Get the best parameters from Grid Search
best_params = grid_search4.best_params_
print("Best Parameters:", best_params)

#----------------------------------------------------------------------------------------

# Create a Random Forest model with the best parameters
best_rf_model = RandomForestClassifier(random_state=42, **best_params, n_jobs=-1)

# Train the model with the best parameters on the entire training set
best_rf_model.fit(X_train, y_train)

# Test
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Train
y_train_pred = best_rf_model.predict(X_train)
test_accuracy = accuracy_score(y_train, y_train_pred)
print("Train accuracy:", test_accuracy)

# Generate classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Extract feature importances
feature_importances = rf_model.feature_importances_

# Print feature importances
print("Feature Importances:")
for feature, importance in zip(X_train.columns, feature_importances):
    print(f"{feature}: {importance:.4f}")
