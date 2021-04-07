from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from metrics import Metrics

class MLP:
    def __init__(self, hidden_layer_sizes, activation, alpha, max_iter, solver):
        self.mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes, 
                activation=activation, 
                alpha=alpha,
                max_iter=max_iter,
                solver=solver
        )
        self.trained = None

    # Returns x_train, x_test, y_train, y_test
    def split_train_test(self, x, y, test_size=0.1):
        return train_test_split(x, y, stratify=y, random_state=1, test_size=test_size)

    def fit(self, x_train, y_train):
        self.trained = self.mlp.fit(x_train, y_train)

    def __predict_prob(self, x_test):
        if self.trained is not None:
            return self.trained.predict_proba(x_test)
        
        return None

    def __threshold_filter(self, probabilities, threshold):
        # probability[1] is the probability to classify as PD
        return [1 if probability[1] >= threshold else 0 for probability in probabilities]

    def score(self, x_test, y_test, threshold, output_file, report_name):
        probabilities = self.__predict_prob(x_test)
        
        predicted = self.__threshold_filter(probabilities, threshold)
        
        metrics = Metrics(predicted, y_test.tolist())
    
        f = open(output_file, "a")
        f.write(metrics.generate_report(report_name, threshold))
        f.close()

        print("Report saved at {}".format(output_file))
