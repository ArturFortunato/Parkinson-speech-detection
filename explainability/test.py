from sklearn import datasets 
from sklearn import model_selection
from sklearn import ensemble

from lime import lime_tabular

import numpy as np
import sklearn
import lime

iris = datasets.load_iris()
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)
print(sklearn.metrics.accuracy_score(labels_test, rf.predict(test)))
explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)
i = np.random.randint(0, test.shape[0])
exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=2, top_labels=1)
print(exp)

exp.show_in_notebook(show_table=True, show_all=False)

