import math
import os
import pandas as pd

experiments = ['baseline', 'baseline_200', 'semi', 'semi_200', 'independent', 'independent_200']

def extract_values(lines, dataset, filename):
    result = {}

    result["dataset"] = dataset
    
    params = filename[:-4].split("_")
    params_headers = ['solver', 'test_size', 'alpha', 'iterations', 'activation']

    for i in range(len(params)):
        result[params_headers[i]] = params[i]

    del result['activation']
    del result['test_size']

    for line in lines:
        if ":" in line:
            splitted = line.split(":")
            if splitted[0] == "Separationthreshold":
                continue
            
            if splitted[0].lower() == 'f1-score':
                key = 'fscore'
            else:
                key = splitted[0].lower()
            
            result[key] = round(float(splitted[1]), 3)
    
    return result

# Allow to see al rows
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

total_number_experiments = 0
for experiment in experiments:
    print(experiment)

    filenames = [os.path.join(dp, f) for dp, dn, filenames in os.walk("../reports/{}".format(experiment)) for f in filenames if os.path.splitext(f)[1] == '.txt']
    result_dicts = []
    
    for filename in filenames:
        filename_splitted = filename.split("/")
        dataset = filename_splitted[3]

        report_lines = open(filename).read().split("\n")
        report_lines = [line.replace("\t", "").replace("\n", "").replace(" ", "") for line in report_lines]

        result = extract_values(report_lines, dataset, filename_splitted[-1])
        result_dicts.append(result)

    csv = pd.DataFrame(result_dicts)

    # Removes files with 100% accuracy (test set too small, not relevant)
    csv = csv[csv.accuracy != float(1)]

    # Removes files without values (corrupted)
    csv = csv.dropna(subset=['accuracy'])

    #Order by accuracy
    csv = csv.sort_values('accuracy', ascending=False)

    total_number_experiments += csv[csv.columns[0]].count()

    csv.to_csv("../reports/csv/{}.csv".format(experiment), sep=",", index=False)

print("Found a total of {} valid experiments".format(total_number_experiments))
