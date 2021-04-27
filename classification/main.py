import os

import pandas as pd
from multiprocessing import Process

from mlp import MLP
from data_processing import DataProcessing

FEATURES     = '/afs/inesc-id.pt/home/aof/thesis/features'
REPORTS_PATH = "../reports"
DATASETS = ['fralusopark', 'gita', 'mdvr_kcl']

# TODO Generalize this function so it can be used 
# in multiple ways during the same execution
def pre_process(input_csv, output_csv):
    if os.path.isfile(output_csv):
        return
    print("Preprocessing {}".format(output_csv))

    preparator = DataProcessing()
    preparator.zscore(input_csv, output_csv, columns_to_ignore=['name', 'label', 'frameTime'])

def get_report_path(experiment, dataset, test_size, alpha, max_iter, act_funcion, solver):
    if   experiment == 'baseline' or experiment == 'independent':
        return "{}/{}/{}/{}_{}_{}_{}_{}.txt".format(REPORTS_PATH, experiment, dataset, solver, test_size, alpha, max_iter, act_funcion)

# set test to True to accelerate debugging
def generate_mlp_params_list(test=False):
    result = []
    alphas = [0.0001, 0.001, 0.01, 0.1]
    max_iters = [1000, 2000, 5000]
    solvers = ['lbfgs', 'sgd', 'adam']

    if test:
        return {'test_size': 0.1, 'alpha': 0.001, 'max_iter': 1000, 'solver': 'adam', 'act_function': 'tanh'}

    for alpha in alphas:
        for max_iter in max_iters:
            for solver in solvers:
                result.append({
                    "test_size": 0.1,
                    "alpha": alpha,
                    "max_iter": max_iter,
                    "act_function":  "tanh",
                    "solver": solver
                })
    return result

def fit_and_score(classifier, train, test, classification_threshold, report_path, report_name):
    # drops frameTime and name columns
    x_train = train[[col for col in train.columns if col not in ['name', 'label', 'frameTime']]]
    y_train = train['label']


    # drops frameTime 
    test = test[[col for col in test.columns if col not in ['frameTime']]]
    
    classifier.fit(x_train, y_train)

    classifier.score(test, classification_threshold, report_path, report_name)

def language_dependent(dataset, classification_threshold, report_name, mlp_params):
    csv_file = '{}/{}/{}_mfcc_plp.csv'.format(FEATURES, dataset, dataset)

    report_path = get_report_path("baseline", dataset, mlp_params["test_size"], mlp_params["alpha"], mlp_params["max_iter"], mlp_params["act_function"], mlp_params["solver"]) 

    csv = pd.read_csv(csv_file, sep=";")

    # hidden layer sizes is features + 1
    # csv.columns is features + 1 (features + label)
    mlp = MLP(hidden_layer_sizes=(len(csv.columns)), activation=mlp_params["act_function"], alpha=mlp_params["alpha"], max_iter=mlp_params["max_iter"], solver=mlp_params["solver"], dataset=dataset)
    
    x = csv.loc[:, csv.columns != 'label']
    y = csv['label']

    train, test = mlp.split_train_test(csv, test_size=mlp_params["test_size"])

    fit_and_score(mlp, train, test, classification_threshold, report_path, report_name)

def language_independent(train_dataset, test_datasets, classification_threshold, report_name, mlp_params):
    train = '{}/{}/{}_normalized.csv'.format(FEATURES, train_dataset, train_dataset)

    report_path = get_report_path("independent", train_dataset, mlp_params["test_size"], mlp_params["alpha"], mlp_params["max_iter"], mlp_params["act_function"], mlp_params["solver"]) 

    train_csv = pd.read_csv(train, sep=";")
    test_csv  = pd.concat([pd.read_csv('{}/{}/{}_normalized.csv'.format(FEATURES, dataset, dataset), sep=";") for dataset in test_datasets ], ignore_index=True)

    # hidden layer sizes is features + 1
    # csv.columns is features + 1 (features + label)
    mlp = MLP(hidden_layer_sizes=(len(train_csv.columns)), activation=mlp_params["act_function"], alpha=mlp_params["alpha"], max_iter=mlp_params["max_iter"], solver=mlp_params["solver"])
    
    x_train = train_csv.loc[:, train_csv.columns != 'label']
    y_train = train_csv['label']

    x_test = test_csv.loc[:, test_csv.columns != 'label']
    y_test = test_csv['label']

    fit_and_score(mlp, x_train, y_train, x_test, y_test, classification_threshold, report_path, report_name)

def perform_baseline(mlp_params_list):
    
    print("Language dependent experiments...\n")
    
    for mlp_params in mlp_params_list:
        language_dependent('fralusopark', 0.5, "Fralusopark Baseline", mlp_params)
        language_dependent('gita',        0.5, "Gita baseline"       , mlp_params)
        language_dependent('mdvr_kcl',    0.5, "MDVR KCL baseline"   , mlp_params)

def perform_semi_independent(mlp_params_list):
    print("Semi independent experiments...\n")

def perform_language_independent(mlp_params_list):
    
    print("Language independent experiments...\n")
    
    for mlp_params in mlp_params_list:
        language_independent('fralusopark', ['gita', 'mdvr_kcl'], 0.5, "Trained with fralusopark", mlp_params)
        language_independent('gita', ['fralusopark', 'mdvr_kcl'], 0.5, "Trained with Gita"       , mlp_params)
        language_independent('mdvr_kcl', ['fralusopark', 'gita'], 0.5, "Trained with MDVR KCL"   , mlp_params)

def main():
    #for dataset in DATASETS:
    #    normalized = '{}/{}/{}_normalized.csv'.format(FEATURES, dataset, dataset)
    #    if not os.path.isfile(normalized):
    #        input_csv  = '{}/{}/{}_mfcc_plp.csv'.format(FEATURES, dataset, dataset)
    #        pre_process(input_csv, normalized)

    mlp_params_list = generate_mlp_params_list()
    
    perform_baseline(mlp_params_list)

    #p_baseline = Process(target=perform_baseline, args=(mlp_params_list,))
    #p_baseline.start()
    #p_baseline.join()

    #p_independent
    #perform_semi_independent(mlp_params_list)
    #perform_language_independent(mlp_params_list)

if __name__ == '__main__':
    main()
