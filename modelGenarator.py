import pandas as pd
import joblib


dataset = pd.read_csv('handData.csv')
x = dataset[['0','2', '3', '4', '6', '7', '8', '10', '11', '12', '14', '15', '16', '18', '19', '20']].values
y = dataset['letter'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.02)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

joblib.dump(classifier, 'hand_model_k5_2.pkl')

print(f'Final Training Accuracy: {classifier.score(X_train,y_train)*100}%')
print(f'Model Accuracy: {classifier.score(X_test,y_test)*100}%')

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

