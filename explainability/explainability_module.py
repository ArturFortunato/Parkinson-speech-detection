import lime

class ExplainabilityModel:
    def test(self):
        print("Test")

    def generate_explanation(self):
        explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)
        i = np.random.randint(0, test.shape[0])
        exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=2, top_labels=1)
        #exp.show_in_notebook(show_table=True, show_all=False)

