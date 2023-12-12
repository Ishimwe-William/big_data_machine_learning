# Learning and Predicting
# Find the most accurate Model
# create a persisting Model
# use the persisting Model

# #################################################################### #
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
#
# data = pd.read_excel("Data.xlsx")
# # print(data.to_string())
#
# # Learning and Predicting
#
# x = data.drop(columns=['SNAMES', 'Total Marks', 'Marks /20', 'Grading'])
# y = data['Grading']
#
# # print(y.to_string())
# # print(x.to_string())
#
# model = DecisionTreeClassifier()
# model.fit(x.values, y)
#
# prediction = model.predict([[10, 10, 24, 13]])
# print(prediction)

# #################################################################### #

# import pandas as pd
#
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
#
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# import joblib
#
# # Load the dataset
# data = pd.read_excel("Data.xlsx")
#
# # Separate features (x) and target variable (y)
# x = data.drop(columns=['SNAMES', 'Total Marks', 'Marks /20', 'Grading'])
# y = data['Grading']
#
#
#
# # Loop through 5 iterations
# for i in range(5):
#     # Split the data into training and testing sets
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#
#     # Initialize models
#     dt_model = DecisionTreeClassifier()
#     svm_model = SVC(kernel='linear')
#     lr_model = LogisticRegression(solver='lbfgs', max_iter=10000)
#     rf_model = RandomForestClassifier(n_estimators=100)
#
#     # Train models
#     dt_model.fit(x_train, y_train)
#     svm_model.fit(x_train, y_train)
#     lr_model.fit(x_train, y_train)
#     rf_model.fit(x_train, y_train)
#
#     # Make predictions
#     dt_pred = dt_model.predict(x_test)
#     svm_pred = svm_model.predict(x_test)
#     lr_pred = lr_model.predict(x_test)
#     rf_pred = rf_model.predict(x_test)
#
#     # Calculate accuracies
#     dt_acc = accuracy_score(y_test, dt_pred)
#     svm_acc = accuracy_score(y_test, svm_pred)
#     lr_acc = accuracy_score(y_test, lr_pred)
#     rf_acc = accuracy_score(y_test, rf_pred)
#
#     # Print accuracies for each model
#     print(f"Iteration {i + 1}:")
#     print(f"DecisionTreeClassifier:\t{dt_acc * 100}%")
#     print(f"Support Vector Machine:\t{svm_acc * 100}%")
#     print(f"Logistic Regression:\t{lr_acc * 100}%")
#     print(f"Random Forest:\t\t{rf_acc * 100}%\n")
#

# ###################################################################### #

# import joblib
# import pandas as pd
# from sklearn.svm import SVC
#
# data = pd.read_excel("Data.xlsx")
#
# # Separate features (x) and target variable (y)
# x = data.drop(columns=['SNAMES', 'Total Marks', 'Marks /20', 'Grading'])
# y = data['Grading']
#
# svm_model = SVC(kernel='linear')
# svm_model.fit(x.values, y)
#
# joblib.dump(svm_model, "grade_classification_model")

# ######################################################################## #

import joblib
# User Inputs
quiz = int(input("Enter Quiz Marks /15: "))
assign = input("Enter Assignment Marks /15: ")
mid = int(input("Enter Mid Exam Marks Marks /30: "))
final = input("Enter Final Exam Marks /40: ")

model = joblib.load('grade_classification_model')
prediction = model.predict([[quiz, assign, mid, final]])
print("\n", prediction)

# ######################################################################## #
