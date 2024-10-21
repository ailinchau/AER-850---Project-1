import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

#2.1
df=pd.read_csv("Project_1_Data.csv")
print(df)

#2.2

x=df['X'].values
y=df['Y'].values
z=df['Z'].values
step=df['Step'].values
plt.plot(step, x, label='X')
plt.plot(step, y, label='Y')
plt.plot(step, z, label='Z')
plt.title('Plot of Disassembling The Inverter')
plt.xlabel('Value')
plt.ylabel('Step')
plt.legend()
plt.show()

#2.3
correlation_matrix=df.corr()
sns.heatmap(correlation_matrix)
plt.title('Correlation Matrix')
plt.show()

#2.4
X=df[['X','Y','Z']]
y=df['Step']
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

my_scaler = StandardScaler()
my_scaler.fit(X_train)
scaled_data_train = my_scaler.transform(X_train)
scaled_data_train_df = pd.DataFrame(scaled_data_train, columns=[f"{col}_scaled" for col in X_train.columns])
X_train = scaled_data_train_df.join(X_train.reset_index(drop=True))
scaled_data_test = my_scaler.transform(X_test)
scaled_data_test_df = pd.DataFrame(scaled_data_test, columns=[f"{col}_scaled" for col in X_test.columns])
X_test = scaled_data_test_df.join(X_test.reset_index(drop=True))

#Random Forest
random_forest= RandomForestClassifier(random_state=42)
param_grid_rf= {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
    }
grid_search_rf= GridSearchCV(random_forest, param_grid_rf, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_model_rf=grid_search_rf.best_params_
print("Best Random Forest Model- Grid Search:", best_model_rf)
y_pred_rf= grid_search_rf.predict(X_test)

randomized_search_rf= RandomizedSearchCV(random_forest, param_distributions=param_grid_rf, cv=5, scoring='f1_weighted', n_jobs=-1)
randomized_search_rf.fit(X_train, y_train)
best_model_rf=randomized_search_rf.best_params_
print("Best Random Forest Model- Randomized Search:", best_model_rf)
y_pred_rs_rf= randomized_search_rf.predict(X_test)

#Support Vector Classifier
svc= SVC()
param_grid_svc={
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto']
    }
grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search_svc.fit(X_train, y_train)
best_model_svc=grid_search_svc.best_estimator_
print("Best SVC Model:", best_model_svc)
y_pred_svc= grid_search_svc.predict(X_test)

#Logistic Regression
logistic_reg = LogisticRegression()
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100]
    }  
grid_search_lr = GridSearchCV(logistic_reg, param_grid_lr, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_model_lr = grid_search_lr.best_estimator_
print("Best Logistic Regression Model:", best_model_lr)
y_pred_lr= grid_search_lr.predict(X_test)

#2.5
f1_rf=f1_score(y_test, y_pred_rf, average='weighted')
f1_svc=f1_score(y_test, y_pred_svc, average='weighted')
f1_lr=f1_score(y_test, y_pred_lr, average='weighted')
f1_rs_rf=f1_score(y_test, y_pred_rs_rf, average='weighted')

precision_rf=precision_score(y_test, y_pred_rf, average='weighted')
precision_svc=precision_score(y_test, y_pred_svc, average='weighted')
precision_lr=precision_score(y_test, y_pred_lr, average='weighted')
precision_rs_rf=precision_score(y_test, y_pred_rs_rf, average='weighted')

accuracy_rf=accuracy_score(y_test, y_pred_rf)
accuracy_svc=accuracy_score(y_test, y_pred_svc)
accuracy_lr=accuracy_score(y_test, y_pred_lr)
accuracy_rs_rf=accuracy_score(y_test, y_pred_rs_rf)

print(f'Random Forest - f1 score: {f1_rf}, precision: {precision_rf}, accuracy: {accuracy_rf}')
print(f'Support Vector Classifier - f1 score: {f1_svc}, precision: {precision_svc}, accuracy: {accuracy_svc}')
print(f'Logisitic Regression - f1 score: {f1_lr}, precision: {precision_lr}, accuracy: {accuracy_lr}')
print(f'Randomized Search Cv Random Forest - f1 score: {f1_rf}, precision: {precision_rf}, accuracy: {accuracy_rf}')

confusion_matrix_rf= confusion_matrix(y_test, y_pred_rf)
confusion_matrix_svc= confusion_matrix(y_test, y_pred_svc)
confusion_matrix_lr= confusion_matrix(y_test, y_pred_lr)
confusion_matrix_rs_rf= confusion_matrix(y_test, y_pred_rs_rf)

def plot_confusion_matrix(matrix, title):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
plot_confusion_matrix(confusion_matrix_rf, 'Confusion Matrix for Random Forest')
plot_confusion_matrix(confusion_matrix_svc, 'Confusion Matrix for Support Vector Classifier')
plot_confusion_matrix(confusion_matrix_lr, 'Confusion Matrix for Logistic Regression')
plot_confusion_matrix(confusion_matrix_rs_rf, 'Confusion Matrix for Randomized Search CV Random Forest')

#2.6

final_est= LogisticRegression(max_iter=5000)

estimators = [
    ('rf', RandomForestClassifier(max_features='sqrt', n_estimators=10, random_state=42)),
    ('svc', SVC(C=100, gamma='auto'))
    ]

stacked_model= StackingClassifier(estimators=estimators, final_estimator=final_est)
stacked_model.fit(X_train, y_train)

y_pred_stacked=stacked_model.predict(X_test)
f1_stacked=f1_score(y_test, y_pred_stacked, average='weighted')
accuracy_stacked=accuracy_score(y_test, y_pred_stacked)
precision_stacked=precision_score(y_test, y_pred_stacked, average='weighted')
print(f'Stacking Classifier - f1 score: {f1_stacked}, precision: {precision_stacked}, accuracy: {accuracy_stacked}')

confusion_matrix_stacked=confusion_matrix(y_test, y_pred_stacked)
plot_confusion_matrix(confusion_matrix_stacked, 'Confusion Matrix for Stacked Model')

#2.7
scaler=StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'scaler.joblib')

model=joblib.load('stacked_model.joblib')
scaler=joblib.load('scaler.joblib')

data_set= np.array([[9.375, 3.0625, 1.51], [6.995, 5.125, 0.3875], [0, 3.0625, 1.93],[9.4, 3, 1.8], [9.4, 3, 1.3]])

data_set=pd.DataFrame(data_set, columns=['X', 'Y', 'Z'])

scaled_data= my_scaler.transform(data_set)

predictions=model.predict(scaled_data)
for input_values, prediction in zip(data_set.values, predictions):
    print("Predicted maintenance steps for the given coordinates {imput_values}: {predictions}")
