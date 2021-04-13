import random

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
    # Deprecated
    def split_train_test_old(self, x, y, test_size=0.1):
        return train_test_split(x, y, stratify=y, random_state=1, test_size=test_size)

    # Returns x_train, x_test, y_train, y_test
    def split_train_test(self, csv, test_size=0.1):
        participants = set(csv["name"])
        test_subjects  = random.sample(participants, round(test_size * len(participants)))
        train_subjects = [participant for participant in participants if participant not in test_subjects]
        
        train = csv.loc[csv['name'].isin(train_subjects)]
        test  = csv.loc[csv['name'].isin(test_subjects )]

        return train, test

    def fit(self, x_train, y_train):
        self.trained = self.mlp.fit(x_train, y_train)

    def __predict_prob(self, x_test):
        if self.trained is not None:
            return self.trained.predict_proba(x_test)
        
        return None

    # Averages the probabilities
    # yielded by the classification model
    def __participant_result(probabilities):
        final_probability = sum(probabilities) / len(probabilities)
        return 1 if final_probability >= threshold else 0

    def score(self, test, threshold, output_file, report_name):
        test_participants = set(test["name"])
        
        predicted = []
        labels = []
        for test_participant in test_participants:
            print("Participant {}".format(test_participant))
            print(test.loc[test['name'] == test_participant])
            participant_rows = self.__predict_prob(test.loc[test['name'] == test_participant])
            participant_x = participant_rows[[col for col in participant_rows.columns if col not in ['label']]]
            participant_y = participant_rows['label'][0]
            probabilities = self.__predict_prob(participant_x)
            participant_result = self.__participant_result(probabilities, threshold)
            predicted.append(participant_result)
            labels.append(participant_y)

        metrics = Metrics(predicted, labels)
    
        f = open(output_file, "a")
        f.write(metrics.generate_report(report_name, threshold))
        f.close()

        print("Report saved at {}".format(output_file))
