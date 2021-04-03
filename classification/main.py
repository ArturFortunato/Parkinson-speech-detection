import os

import pandas as pd

from mlp import MLP
from data_processing import DataProcessing

FEATURES     = '/afs/inesc-id.pt/home/aof/thesis/features'
REPORTS_PATH = "../reports"

# TODO Generalize this function so it can be used 
# in multiple ways during the same execution
def pre_process(input_csv, output_csv):
    if os.path.isfile(output_csv):
        return
    preparator = DataProcessing()
    preparator.zscore(input_csv, output_csv, columns_to_ignore=['name', 'label'])

def get_report_path(experiment, dataset, test_size, alpha, max_iter, act_funcion):
    if experiment == "baseline":
        return "{}/{}/{}/{}_{}_{}_{}.txt".format(REPORTS_PATH, experiment, dataset, test_size, alpha, max_iter, act_funcion)

def language_dependent(dataset, classification_threshold, report_name, mlp_params):
    input_csv  = '{}/{}/{}.csv'.format(FEATURES, dataset, dataset)
    normalized = '{}/{}/{}_normalized.csv'.format(FEATURES, dataset, dataset, dataset)

    report_path = get_report_path("baseline", dataset, mlp_params["test_size"], mlp_params["alpha"], mlp_params["max_iter"], mlp_params["act_function"]) 

    pre_process(input_csv, normalized)
    csv = pd.read_csv(normalized, sep=";")

    # hidden layer sizes is features + 1
    # csv.columns is features + 1 (features + label)
    mlp = MLP(hidden_layer_sizes=(len(csv.columns)), activation=mlp_params["act_function"], alpha=mlp_params["alpha"], max_iter=mlp_params["max_iter"])
    
    x = csv.loc[:, csv.columns != 'label']
    y = csv['label']

    x_train, x_test, y_train, y_test = mlp.split_train_test(x, y, test_size=mlp_params["test_size"])
    
    mlp.fit(x_train, y_train)

    mlp.score(x_test, y_test, classification_threshold, report_path, report_name)

def main():
    
    print("Language dependent experiments...\n")

    mlp_params = {
        "test_size": 0.1,
        "alpha": 0.001,
        "max_iter": 2000,
        "act_function": "tanh"
    }

    language_dependent('fralusopark', 0.5, "Fralusopark Baseline", mlp_params)
    language_dependent('gita',        0.5, "Gita baseline"       , mlp_params)
    language_dependent('mdvr_kcl',    0.5, "MDVR KCL baseline"   , mlp_params)

if __name__ == '__main__':
    main()
