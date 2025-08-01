import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostRegressor
plt.close('all')


# Load and Prepare Data

url = "https://raw.githubusercontent.com/chandanverma07/DataSets/master/Concrete_Data.csv"
df = pd.read_csv(url)
X = df.drop(columns=['CMS'])
y = df['CMS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Utility Functions
def evaluate_model(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n{label}:\nR² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    return y_pred, r2, rmse, mae

def plot_actual_vs_predicted(y_test, y_pred, title, color):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', color=color)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Concrete Strength (MPa)")
    plt.ylabel("Predicted Concrete Strength (MPa)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_residuals(y_test, y_pred, title, color):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', color=color)
    plt.axhline(0, color='blue', linestyle='--')
    plt.xlabel("Predicted Concrete Strength")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.grid(True)
    plt.show()

# Model Dictionary
models = {
    "Linear": LinearRegression(),
    "Ridge": GridSearchCV(Ridge(), {"alpha": [0.001, 0.01, 0.1, 1, 10]}, cv=5),
    "Lasso": GridSearchCV(Lasso(max_iter=10000), {"alpha": [0.001, 0.01, 0.1, 1, 10]}, cv=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GBT": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "Cat GBT": CatBoostRegressor(n_estimators=100, random_state=42),
    "Elastic Net": ElasticNet(random_state=42),
}

# Train, Evaluate, and Plot
r2s, rmses, maes, model_names = [], [], [], []

for name, model in models.items():
    model.fit(X_train, y_train)
    best_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model

    y_pred, r2, rmse, mae = evaluate_model(best_model, X_test, y_test, name)
    r2s.append(r2)
    rmses.append(rmse)
    maes.append(mae)
    model_names.append(name)

    plot_actual_vs_predicted(y_test, y_pred, f"{name} - Actual vs Predicted", 'lightblue')
    plot_residuals(y_test, y_pred, f"{name} - Residuals", 'coral')

# ---------------------------
# 5. Error Metric Comparison
# ---------------------------
x = np.arange(len(model_names))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - width, r2s, width=width, label='R²', color='skyblue')
plt.bar(x,         rmses, width=width, label='RMSE', color='orange')
plt.bar(x + width, maes, width=width, label='MAE', color='green')
plt.xticks(x, model_names)
plt.ylabel("Metric Value")
plt.title("Model Error Comparison (R², RMSE, MAE)")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()





# #fitting the lr
# lr = LinearRegression()
# lr.fit(X_train, y_train)
#
# #predict and evaluate
# y_pred = lr.predict(X_test)
#
# #print out the r2. rmse, mae to judge accuracy
# r2 = r2_score(y_test, y_pred)
# print("R² Score:", r2)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("RMSE:", rmse)
# mae = mean_absolute_error(y_test, y_pred)
# print("MAE:", mae)
#
# #plot the model's actual vs predicted strength
# plt.close('all')
#
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, color='skyblue', edgecolor='black', alpha=0.6)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel("Actual Concrete Strength (MPa)")
# plt.ylabel("Predicted Concrete Strength (MPa)")
# plt.title("Actual vs Predicted Concrete Strength")
# plt.grid(True)
# plt.show()
#
# # Print intercept
# print("Intercept:", lr.intercept_)
#
# # Print coefficients with their corresponding feature names
# print("\nFeature Coefficients:")
# for name, coef in zip(X.columns, lr.coef_):
#     print(f"{name}: {coef:.4f}")
#
# # Calculate residuals
# residuals = y_test - y_pred
#
# # Plot residuals vs predicted values
# plt.figure(figsize=(8, 6))
# plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', color='coral')
# plt.axhline(y=0, color='blue', linestyle='--', linewidth=2)
# plt.xlabel("Predicted Concrete Strength (MPa)")
# plt.ylabel("Residuals (Actual - Predicted)")
# plt.title("Residuals vs Predicted")
# plt.grid(True)
# plt.show()
#
# ## Plot Feature Importance as a Bar Chart
#
# # Get feature names and coefficients
# features = X.columns
# coefficients = lr.coef_
#
# # Sort by absolute value of coefficients
# sorted_indices = np.argsort(np.abs(coefficients))[::-1]
# features_sorted = features[sorted_indices]
# coefficients_sorted = coefficients[sorted_indices]
#
# # Plot
# plt.figure(figsize=(10, 6))
# plt.barh(features_sorted, coefficients_sorted, color='skyblue', edgecolor='black')
# plt.xlabel("Coefficient Value (MPa/unit)")
# plt.title("Feature Importance (Linear Regression Coefficients)")
# plt.gca().invert_yaxis()  # Highest importance at top
# plt.grid(True, axis='x')
# plt.show()
#
# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
#
# # Define a grid of alpha values to try
# alpha_values = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
#
# # Calibrate and find best alpha value
# ridge = Ridge()
# ridge_cv = GridSearchCV(ridge, alpha_values, cv=5, scoring='r2')
# ridge_cv.fit(X_train, y_train)
#
# print("Best alpha for Ridge:", ridge_cv.best_params_['alpha'])
# print("Best R² (CV):", ridge_cv.best_score_)
#
# ridge = ridge_cv.best_estimator_
# y_ridge = ridge.predict(X_test)
#
# #plot the ridge regression
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_ridge, color='lightgreen', edgecolor='black', alpha=0.6)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel("Actual Concrete Strength (MPa)")
# plt.ylabel("Predicted Concrete Strength (MPa)")
# plt.title("Actual vs Predicted - Ridge Regression")
# plt.grid(True)
# plt.show()
#
# # Calculate residuals for Ridge
# residuals_ridge = y_test - y_ridge
#
# plt.figure(figsize=(8, 6))
# plt.scatter(y_ridge, residuals_ridge, alpha=0.6, edgecolors='black', color='lightgreen')
# plt.axhline(y=0, color='blue', linestyle='--', linewidth=2)
# plt.xlabel("Predicted Concrete Strength (MPa) - Ridge")
# plt.ylabel("Residuals (Actual - Predicted)")
# plt.title("Residuals vs Predicted - Ridge Regression")
# plt.grid(True)
# plt.show()
#
#
# # Fit Lasso Regression
# lasso = Lasso(max_iter=10000)
# lasso_cv = GridSearchCV(lasso, alpha_values, cv=5, scoring='r2')
# lasso_cv.fit(X_train, y_train)
#
# print("Best alpha for Lasso:", lasso_cv.best_params_['alpha'])
# print("Best R² (CV):", lasso_cv.best_score_)
#
# lasso = lasso_cv.best_estimator_
# y_lasso = lasso.predict(X_test)
# #Plot Lasso regression
#
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_lasso, color='salmon', edgecolor='black', alpha=0.6)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel("Actual Concrete Strength (MPa)")
# plt.ylabel("Predicted Concrete Strength (MPa)")
# plt.title("Actual vs Predicted - Lasso Regression")
# plt.grid(True)
# plt.show()
#
# # Calculate residuals for Lasso
# residuals_lasso = y_test - y_lasso
#
# plt.figure(figsize=(8, 6))
# plt.scatter(y_lasso, residuals_lasso, alpha=0.6, edgecolors='black', color='salmon')
# plt.axhline(y=0, color='blue', linestyle='--', linewidth=2)
# plt.xlabel("Predicted Concrete Strength (MPa) - Lasso")
# plt.ylabel("Residuals (Actual - Predicted)")
# plt.title("Residuals vs Predicted - Lasso Regression")
# plt.grid(True)
# plt.show()
#
# # Fit Random Forest
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
#
# # Predict
# y_rf = rf.predict(X_test)
#
# # Evaluate Random Forest
# r2_rf = r2_score(y_test, y_rf)
# rmse_rf = np.sqrt(mean_squared_error(y_test, y_rf))
# mae_rf = mean_absolute_error(y_test, y_rf)
#
# #Plot actual vs predicted of random forest
# print("\nRandom Forest Regressor:")
# print(f"R² Score: {r2_rf:.4f}")
# print(f"RMSE:     {rmse_rf:.4f}")
# print(f"MAE:      {mae_rf:.4f}")
#
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_rf, color='gold', edgecolor='black', alpha=0.6)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel("Actual Concrete Strength (MPa)")
# plt.ylabel("Predicted Concrete Strength (MPa)")
# plt.title("Actual vs Predicted - Random Forest")
# plt.grid(True)
# plt.show()
#
# # Get feature names and model coefficients
# feature_names = X.columns.to_numpy()
# coefs_linear = lr.coef_
# coefs_ridge = ridge.coef_
# coefs_lasso = lasso.coef_
#
# # Sort indices based on absolute value of linear regression coefficients
# sorted_idx = np.argsort(np.abs(coefs_linear))[::-1]
#
# # Apply sorting to all arrays
# feature_names_sorted = feature_names[sorted_idx]
# coefs_linear_sorted = coefs_linear[sorted_idx]
# coefs_ridge_sorted = coefs_ridge[sorted_idx]
# coefs_lasso_sorted = coefs_lasso[sorted_idx]
#
# # Plot
# plt.figure(figsize=(10, 6))
# x = np.arange(len(feature_names_sorted))
#
# plt.bar(x - 0.2, coefs_linear_sorted, width=0.2, label='Linear', color='skyblue')
# plt.bar(x,       coefs_ridge_sorted,  width=0.2, label='Ridge',  color='orange')
# plt.bar(x + 0.2, coefs_lasso_sorted,  width=0.2, label='Lasso',  color='green')
#
# plt.xticks(x, feature_names_sorted, rotation=45)
# plt.ylabel("Coefficient Value")
# plt.title("Sorted Model Coefficient Comparison")
# plt.legend()
# plt.tight_layout()
# plt.grid(True, axis='y')
# plt.show()
#
# # Store the metrics
# model_names = ['Linear', 'Ridge', 'Lasso', 'Random Forest']
# r2_scores = [
#     r2_score(y_test, y_pred),
#     r2_score(y_test, ridge.predict(X_test)),
#     r2_score(y_test, lasso.predict(X_test)),
#     r2_rf
# ]
# rmse_scores = [
#     np.sqrt(mean_squared_error(y_test, y_pred)),
#     np.sqrt(mean_squared_error(y_test, ridge.predict(X_test))),
#     np.sqrt(mean_squared_error(y_test, lasso.predict(X_test))),
#     rmse_rf
# ]
# mae_scores = [
#     mean_absolute_error(y_test, y_pred),
#     mean_absolute_error(y_test, ridge.predict(X_test)),
#     mean_absolute_error(y_test, lasso.predict(X_test)),
#     mae_rf
# ]
#
# # 4. Plot comparison
# x = np.arange(len(model_names))  # x-axis positions
# width = 0.2
#
# plt.figure(figsize=(12, 6))
# plt.bar(x - width, r2_scores, width=width, label='R²', color='skyblue')
# plt.bar(x,         rmse_scores, width=width, label='RMSE', color='orange')
# plt.bar(x + width, mae_scores, width=width, label='MAE', color='green')
#
# plt.xticks(x, model_names)
# plt.ylabel("Metric Value")
# plt.title("Model Error Comparison (R², RMSE, MAE)")
# plt.legend()
# plt.grid(True, axis='y')
# plt.tight_layout()
# plt.show()
#
# # Get Random Forest importances and sort them using the same index as linear model
# coefs_rf = rf.feature_importances_
# coefs_rf_sorted = coefs_rf[sorted_idx]
#
# # Update plot
# plt.figure(figsize=(12, 6))
# x = np.arange(len(feature_names_sorted))
#
# plt.bar(x - 0.3, coefs_linear_sorted, width=0.2, label='Linear', color='skyblue')
# plt.bar(x - 0.1, coefs_ridge_sorted,  width=0.2, label='Ridge',  color='orange')
# plt.bar(x + 0.1, coefs_lasso_sorted,  width=0.2, label='Lasso',  color='green')
# plt.bar(x + 0.3, coefs_rf_sorted,     width=0.2, label='Random Forest (Importance)', color='gold')
#
# plt.xticks(x, feature_names_sorted, rotation=45)
# plt.ylabel("Coefficient / Importance Value")
# plt.title("Sorted Feature Comparison Across Models")
# plt.legend()
# plt.tight_layout()
# plt.grid(True, axis='y')
# plt.show()


