import sys
import os

import pandas as pd
from multiprocessing import Process

from mlp import MLP
from data_processing import DataProcessing

FEATURES     = '/afs/inesc-id.pt/home/aof/thesis/features'
REPORTS_PATH = "../reports"
DATASETS = ['fralusopark', 'gita', 'mdvr_kcl']

def get_report_path(experiment, dataset, test_size, alpha, max_iter, act_funcion, solver):
    return "{}/{}/{}/{}_{}_{}_{}_{}.txt".format(REPORTS_PATH, experiment, dataset, solver, test_size, alpha, max_iter, act_funcion)
    
def generate_mlp_params_list(test=False):
    result = []
    alphas = [0.0001, 0.001, 0.01]
    max_iters = [2000, 5000]
    solvers = ['lbfgs', 'adam']

    if test:
        return [{'test_size': 0.1, 'alpha': 0.001, 'max_iter': 1, 'solver': 'adam', 'activation': 'tanh'}]

    for alpha in alphas:
        for max_iter in max_iters:
            for solver in solvers:
                result.append({
                    "test_size": 0.1,
                    "alpha": alpha,
                    "max_iter": max_iter,
                    "activation":  "tanh",
                    "solver": solver
                })
    return result

def fit_and_score(classifier, train, test, classification_threshold, report_path, report_name):
    x_train = train[[col for col in train.columns if col not in ['name', 'label', 'frameTime']]]
    y_train = train['label']

    test = test[[col for col in test.columns if col not in ['frameTime']]]
    
    classifier.fit(x_train, y_train)

    classifier.score(test, classification_threshold, report_path, report_name)

def report_already_exists(report_path):
    if os.path.isfile(report_path):
        print("Past experiment results found, ignoring...")
        return True
    
    return False

def language_dependent(dataset, classification_threshold, report_name, mlp_params, hidden_layer_sizes=None):
    print("Starting language dependent for {}".format(mlp_params))

    csv_file = '{}/{}/{}_complete.csv'.format(FEATURES, dataset, dataset)
    experiment = "baseline" if hidden_layer_sizes == None else "baseline_200"

    report_path = get_report_path(experiment, dataset, mlp_params["test_size"], mlp_params["alpha"], mlp_params["max_iter"], mlp_params["activation"], mlp_params["solver"]) 

    if report_already_exists(report_path):
        return

    csv = pd.read_csv(csv_file, sep=";")
    
    # hidden layer sizes is features + 1
    # csv.columns is features + 1 (features + label)
    if hidden_layer_sizes == None:
        hidden_layer_sizes = (len(csv.columns))

    mlp = MLP(hidden_layer_sizes, mlp_params=mlp_params, dataset=dataset, experiment=experiment)
    
    train, test = MLP.split_train_test(csv, test_size=mlp_params["test_size"])

    fit_and_score(mlp, train, test, classification_threshold, report_path, report_name)


def semi_independent(full_dataset, semi_dataset, classification_threshold, report_name, mlp_params, hidden_layer_sizes=None):
    print("Starting semi language dependent for {}".format(mlp_params))

    dataset_name = "{}_{}".format(full_dataset, semi_dataset)

    full_csv_file = '{}/{}/{}_complete.csv'.format(FEATURES, full_dataset, full_dataset)
    semi_csv_file = '{}/{}/{}_complete.csv'.format(FEATURES, semi_dataset, semi_dataset)
    experiment    = "semi" if hidden_layer_sizes == None else "semi_200"
    
    report_path = get_report_path(experiment, dataset_name, mlp_params["test_size"], mlp_params["alpha"], mlp_params["max_iter"], mlp_params["activation"], mlp_params["solver"]) 

    if report_already_exists(report_path):
        return

    full_csv  = pd.read_csv(full_csv_file, sep=";")
    semi_csv  = pd.read_csv(semi_csv_file, sep=";")

    train_part, test_csv = MLP.split_train_test(semi_csv, test_size=mlp_params["test_size"])
    
    train_csv = pd.concat([full_csv, train_part], ignore_index=True)

    # hidden layer sizes is features + 1
    # csv.columns is features + 1 (features + label)
    if hidden_layer_sizes == None:
        hidden_layer_sizes = (len(train_csv.columns))

    mlp = MLP(hidden_layer_sizes, mlp_params, dataset=dataset_name, experiment=experiment)
    
    fit_and_score(mlp, train_csv, test_csv, classification_threshold, report_path, report_name)

def language_independent(train_datasets, test_dataset, classification_threshold, report_name, mlp_params, hidden_layer_sizes=None):
    print("Starting language dependent for {}".format(mlp_params))

    test_csv_file = '{}/{}/{}_complete.csv'.format(FEATURES, test_dataset, test_dataset)
    experiment = "independent" if hidden_layer_sizes == None else "independent_200"

    report_path = get_report_path(experiment, test_dataset, mlp_params["test_size"], mlp_params["alpha"], mlp_params["max_iter"], mlp_params["activation"], mlp_params["solver"]) 

    if report_already_exists(report_path):
        return

    train_csv = pd.concat([pd.read_csv('{}/{}/{}_complete.csv'.format(FEATURES, dataset, dataset), sep=";") for dataset in train_datasets ], ignore_index=True)
    test_csv  = pd.read_csv(test_csv_file, sep=";")
    
    # hidden layer sizes is features + 1
    # csv.columns is features + 1 (features + label)
    if hidden_layer_sizes == None:
        hidden_layer_sizes = (len(train_csv.columns))

    mlp = MLP(hidden_layer_sizes, mlp_params, dataset=test_dataset, experiment=experiment)
    
    fit_and_score(mlp, train_csv, test_csv, classification_threshold, report_path, report_name)

def perform_baseline(mlp_params_list):
    
    print("Language dependent experiments...\n")
    
    processes = []

    for mlp_params in mlp_params_list:
        #processes.append(Process(target=language_dependent, args=('fralusopark', 0.5, "Fralusopark Baseline", mlp_params, )))
        #processes.append(Process(target=language_dependent, args=('gita'       , 0.5, "Gita Baseline"       , mlp_params, )))
        #processes.append(Process(target=language_dependent, args=('mdvr_kcl'   , 0.5, "MDVR KCL Baseline"   , mlp_params, )))

        #processes.append(Process(target=language_dependent, args=('fralusopark', 0.5, "Fralusopark Baseline", mlp_params, (200,200), )))
        #processes.append(Process(target=language_dependent, args=('gita'       , 0.5, "Gita Baseline"       , mlp_params, (200,200), )))
        #processes.append(Process(target=language_dependent, args=('mdvr_kcl'   , 0.5, "MDVR KCL Baseline"   , mlp_params, (200,200), )))

    #for process in processes:
    #    process.start()

    #for process in processes:
    #    process.join()

        language_dependent('fralusopark', 0.5, "Fralusopark Baseline", mlp_params, (200,200))
        language_dependent('gita'       , 0.5, "Gita Baseline"       , mlp_params, (200,200))
        language_dependent('mdvr_kcl'   , 0.5, "MDVR KCL Baseline"   , mlp_params, (200,200))

    print("Language dependent experiments: DONE!\n")

def perform_semi_independent(mlp_params_list):
    print("Semi independent experiments...\n")

    #processes = []
    for mlp_params in mlp_params_list:
        #processes.append(Process(target=semi_independent, args=('fralusopark', 'gita'    , 0.5, "Trained with fralusopark and part Gita"    , mlp_params, )))
        #processes.append(Process(target=semi_independent, args=('fralusopark', 'mdvr_kcl', 0.5, "Trained with fralusopark and part MDVR_KCL", mlp_params, )))
        #processes.append(Process(target=semi_independent, args=('gita', 'fralusopark'    , 0.5, "Trained with gita and part fralusopark"    , mlp_params, )))
        #processes.append(Process(target=semi_independent, args=('gita', 'mdvr_kcl'       , 0.5, "Trained with gita and part MDVR_KCL"       , mlp_params, )))
        #processes.append(Process(target=semi_independent, args=('mdvr_kcl', 'fralusopark', 0.5, "Trained with MDVR_KCL and part fralusopark", mlp_params, )))
        #processes.append(Process(target=semi_independent, args=('mdvr_kcl', 'gita'       , 0.5, "Trained with MDVR_KCL and part gita"       , mlp_params, )))

        semi_independent('fralusopark', 'gita'    , 0.5, "Trained with fralusopark and part Gita"    , mlp_params)
        semi_independent('fralusopark', 'mdvr_kcl', 0.5, "Trained with fralusopark and part MDVR_KCL", mlp_params)
        semi_independent('gita', 'fralusopark'    , 0.5, "Trained with gita and part fralusopark"    , mlp_params)
        semi_independent('gita', 'mdvr_kcl'       , 0.5, "Trained with gita and part MDVR_KCL"       , mlp_params)
        semi_independent('mdvr_kcl', 'fralusopark', 0.5, "Trained with MDVR_KCL and part fralusopark", mlp_params)
        semi_independent('mdvr_kcl', 'gita'       , 0.5, "Trained with MDVR_KCL and part gita"       , mlp_params)

        #processes.append(Process(target=semi_independent, args=('fralusopark', 'gita'    , 0.5, "Trained with fralusopark and part Gita"    , mlp_params, (200,200), )))
        #processes.append(Process(target=semi_independent, args=('fralusopark', 'mdvr_kcl', 0.5, "Trained with fralusopark and part MDVR_KCL", mlp_params, (200,200), )))
        #processes.append(Process(target=semi_independent, args=('gita', 'fralusopark'    , 0.5, "Trained with gita and part fralusopark"    , mlp_params, (200,200), )))
        #processes.append(Process(target=semi_independent, args=('gita', 'mdvr_kcl'       , 0.5, "Trained with gita and part MDVR_KCL"       , mlp_params, (200,200), )))
        #processes.append(Process(target=semi_independent, args=('mdvr_kcl', 'fralusopark', 0.5, "Trained with MDVR_KCL and part fralusopark", mlp_params, (200,200), )))
        #processes.append(Process(target=semi_independent, args=('mdvr_kcl', 'gita'       , 0.5, "Trained with MDVR_KCL and part gita"       , mlp_params, (200,200), )))
        
        semi_independent('fralusopark', 'gita'    , 0.5, "Trained with fralusopark and part Gita"    , mlp_params, (200,200))
        semi_independent('fralusopark', 'mdvr_kcl', 0.5, "Trained with fralusopark and part MDVR_KCL", mlp_params, (200,200))
        semi_independent('gita', 'fralusopark'    , 0.5, "Trained with gita and part fralusopark"    , mlp_params, (200,200))
        semi_independent('gita', 'mdvr_kcl'       , 0.5, "Trained with gita and part MDVR_KCL"       , mlp_params, (200,200))
        semi_independent('mdvr_kcl', 'fralusopark', 0.5, "Trained with MDVR_KCL and part fralusopark", mlp_params, (200,200))
        semi_independent('mdvr_kcl', 'gita'       , 0.5, "Trained with MDVR_KCL and part gita"       , mlp_params, (200,200))

    #for process in processes:
    #    process.start()

    #for process in processes:
    #    process.join()

    print("Language semi dependent experiments: DONE!\n")

def perform_language_independent(mlp_params_list):
    
    print("Language independent experiments...\n")
    
    #processes = []
    for mlp_params in mlp_params_list:
        #processes.append(Process(target=language_independent, args=(['gita', 'mdvr_kcl'], 'fralusopark', 0.5, "Tested with fralusopark", mlp_params, )))
        #processes.append(Process(target=language_independent, args=(['fralusopark', 'mdvr_kcl'], 'gita', 0.5, "Tested with Gita"       , mlp_params, )))
        #processes.append(Process(target=language_independent, args=(['fralusopark', 'gita'], 'mdvr_kcl', 0.5, "Tested with MDVR KCL"   , mlp_params, )))

        language_independent(['gita', 'mdvr_kcl'], 'fralusopark', 0.5, "Tested with fralusopark", mlp_params)
        language_independent(['fralusopark', 'mdvr_kcl'], 'gita', 0.5, "Tested with Gita"       , mlp_params)
        language_independent(['fralusopark', 'gita'], 'mdvr_kcl', 0.5, "Tested with MDVR KCL"   , mlp_params)
        
        #processes.append(Process(target=language_independent, args=(['gita', 'mdvr_kcl'], 'fralusopark', 0.5, "Tested with fralusopark", mlp_params, (200,200), )))
        #processes.append(Process(target=language_independent, args=(['fralusopark', 'mdvr_kcl'], 'gita', 0.5, "Tested with Gita"       , mlp_params, (200,200), )))
        #processes.append(Process(target=language_independent, args=(['fralusopark', 'gita'], 'mdvr_kcl', 0.5, "Tested with MDVR KCL"   , mlp_params, (200,200), )))
        language_independent(['gita', 'mdvr_kcl'], 'fralusopark', 0.5, "Tested with fralusopark", mlp_params, (200,200))
        language_independent(['fralusopark', 'mdvr_kcl'], 'gita', 0.5, "Tested with Gita"       , mlp_params, (200,200))
        language_independent(['fralusopark', 'gita'], 'mdvr_kcl', 0.5, "Tested with MDVR KCL"   , mlp_params, (200,200))

    #for process in processes:
    #    process.start()

    #for process in processes:
    #    process.join()

    print("Language independent experiments: DONE!\n")

def main():
    mlp_params_list = generate_mlp_params_list()
    
    print(mlp_params_list)
    targets = [perform_baseline, perform_semi_independent, perform_language_independent]
    experiments = []

    for target in targets:
        experiments.append(Process(target=target, args=(mlp_params_list,)))

    for experiment in experiments:
        experiment.start()

    for experiment in experiments:
        experiment.join()

if __name__ == '__main__':
    main()
