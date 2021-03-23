import pandas as pd

from mlp import MLP

FRALUSOPARK = '/afs/inesc-id.pt/home/aof/thesis/features/fralusopark/fralusopark.csv' 
GITA        = '/afs/inesc-id.pt/home/aof/thesis/features/gita/gita.csv' 
MDVR_KCL    = '/afs/inesc-id.pt/home/aof/thesis/features/mdvr_kcl/mdvr_kcl.csv' 

REPORTS_PATH = "../reports"

def language_dependent(dataset, classification_threshold, report_path, report_name):
    csv = pd.read_csv(dataset, sep=";")

    # hidden layer sizes is features + 1
    # csv.columns is features + 1 (features + label)
    mlp = MLP(hidden_layer_sizes=(len(csv.columns)), activation='relu')
    
    x = csv.loc[:, csv.columns != 'label']
    y = csv['label']

    x_train, x_test, y_train, y_test = mlp.split_train_test(x, y)
    
    mlp.fit(x_train, y_train)

    mlp.score(x_test, y_test, classification_threshold, report_path, report_name)

def main():
    
    print("Language dependent experiments...\n")

    language_dependent(FRALUSOPARK, 0.5, "{}/fralusopark_baseline.txt".format(REPORTS_PATH), "Fralusopark Baseline")
    language_dependent(GITA,        0.5, "{}/gita_baseline.txt".format(REPORTS_PATH), "Gita baseline")
    language_dependent(MDVR_KCL,    0.5, "{}/mdvr_kcl_baseline.txt".format(REPORTS_PATH), "MDVR KCL baseline")

if __name__ == '__main__':
    main()
