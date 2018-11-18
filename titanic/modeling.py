from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from titanic import wranging

features_train, labels_train, features_test, labels_test = wranging.get_data()
clf = RandomForestClassifier(n_estimators=5, random_state=42)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test, pred)
print(acc)

