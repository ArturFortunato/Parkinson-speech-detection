import random
import os

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import pickle
import pandas as pd

from metrics import Metrics

SUBSETS_PATH = "../../subsets"

class MLP:
    def __init__(self, hidden_layer_sizes, mlp_params, dataset, experiment):
        self.activation = mlp_params['activation']
        self.alpha      = mlp_params['alpha']
        self.solver     = mlp_params['solver']
        self.max_iter   = mlp_params['max_iter'] 
        self.experiment = experiment
        self.dataset    = dataset

        self.mlp = MLPClassifier(
                hidden_layer_sizes = hidden_layer_sizes, 
                activation         = self.activation, 
                alpha              = self.alpha,
                max_iter           = self.max_iter,
                solver             = self.solver
        )

        if os.path.isfile('./pickles/{}/{}/{}.pkl'.format(self.experiment, self.dataset, self.generate_suffix())):
            with open('./pickles/{}/{}/{}.pkl'.format(self.experiment, self.dataset, self.generate_suffix()), 'rb') as filename:
                self.trained = pickle.load(filename)
        else:
            self.trained = None

    @staticmethod
    def save_csv(csv, path):
        csv.to_csv(path, sep=";", index=False)

    def split_train_test(self, csv, test_size=0.1):
        train_file = "{}/{}/{}/{}_train.csv".format(SUBSETS_PATH, self.experiment, self.dataset, self.generate_suffix())
        test_file  = "{}/{}/{}/{}_test.csv".format(SUBSETS_PATH, self.experiment, self.dataset, self.generate_suffix())

        if os.path.isfile(train_file) and os.path.isfile(test_file):
            train = pd.read_csv(train_file, sep=";")
            test= pd.read_csv(test_file, sep=";")

            return train, test
        
        participants = set(csv["name"])
        test_subjects  = random.sample(participants, round(test_size * len(participants)))
        train_subjects = [participant for participant in participants if participant not in test_subjects]
        
        train = csv.loc[csv['name'].isin(train_subjects)]
        test  = csv.loc[csv['name'].isin(test_subjects )]

        MLP.save_csv(train, "{}/{}/{}/{}_train.csv".format(SUBSETS_PATH, self.experiment, self.dataset, self.generate_suffix()))
        MLP.save_csv(test , "{}/{}/{}/{}_test.csv".format(SUBSETS_PATH, self.experiment, self.dataset, self.generate_suffix()))

        return train, test

    def generate_suffix(self):
        return "{}_{}_{}_{}".format(self.solver, self.alpha, self.max_iter, self.activation )

    def fit(self, x_train, y_train):
        if self.trained is not None:
            return

        self.trained = self.mlp.fit(x_train, y_train)

        with open('./pickles/{}/{}/{}.pkl'.format(self.experiment, self.dataset, self.generate_suffix()), 'wb') as filename:
            pickle.dump(self.mlp, filename)

    def __predict_prob(self, x_test):
        if self.trained is not None:
            return self.trained.predict_proba(x_test)
        
        return None

    # Averages the probabilities
    # yielded by the classification model
    def __participant_result(self, probabilities, threshold):
        healthy_prob = sum([i[0] for i in probabilities]) / len(probabilities)
        return 1 if healthy_prob <= threshold else 0

    def score(self, test, threshold, output_file, report_name):
        test_participants = set(test["name"])
        
        predicted = []
        labels = []
        for test_participant in test_participants:
            participant_rows = test.loc[test['name'] == test_participant]
            participant_x = participant_rows[[col for col in participant_rows.columns if col not in ['label', 'name', 'frameTime']]]
            participant_y = participant_rows['label'][participant_rows.index[0]]
            probabilities = self.__predict_prob(participant_x)
            participant_result = self.__participant_result(probabilities, threshold)
            predicted.append(participant_result)
            labels.append(participant_y)

        metrics = Metrics(predicted, labels)
    
        f = open(output_file, "w")
        f.write(metrics.generate_report(report_name, threshold))
        f.close()

        print("Report saved at {}".format(output_file))
