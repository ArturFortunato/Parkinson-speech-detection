import pandas as pd
import json
import os

NUM_FEATURES = 5

def get_csvs_paths():
    result = []
    experiments = ['baseline', 'baseline_200', 'semi', 'semi_200', 'independent', 'independent_200']
    for experiment in experiments:
        result += [os.path.join(dp, f) for dp, dn, filenames in os.walk('./csv/{}'.format(experiment)) for f in filenames if os.path.splitext(f)[1] == '.csv']

    return result

# Converts dictionary to csv
# And writes it to a file
def dic_to_csv(dic, num_patients, base_name, sorters):
    csv = pd.DataFrame()

    for feature in dic:
        percentage_patients = dic[feature]['count'] / num_patients * 100
        line = {'feature': feature, 'percentage': percentage_patients, 'weight': dic[feature]['weight'], 'count': dic[feature]['count']}
        csv = csv.append(line, ignore_index=True)
    
    for sorter in sorters:
        result = csv.sort_values(by =sorters[sorter], ascending=False, key=abs)
        result['percentage'] = result['percentage'].apply(lambda x: "{:.3f}".format(x))
        result['weight'] = result['weight'].apply(lambda x: "{:.4f}".format(x))
        result = result.drop(['count'], axis=1)
        result.to_csv('{}_by_{}.csv'.format(base_name, sorter), index=False)

# Generate stats
def generate_stats(files):
    result = {}
    total_count = 0
    # For each csv result
    for f in files:
        csv = pd.read_csv(f, index_col=None)

        # For each test patient
        for _, row in csv.iterrows():
            total_count += 1
            features = json.loads(row['features'])
            feature_list = list(features.keys())[:NUM_FEATURES]
            top_features = dict((k, features[k]) for k in feature_list if k in features)

            # For each one of the top features
            for feature in top_features:
                if feature in result:
                    result[feature]['count']  += 1
                    result[feature]['weight'] += abs(top_features[feature])

                else:
                    result[feature] = {'count': 1, 'weight': top_features[feature]}
        
    for feature in result:
        result[feature]['weight'] /= result[feature]['count']

    return result, total_count
    
def main():
    files = get_csvs_paths()
    stats, total_count = generate_stats(files)

    dic_to_csv(stats, total_count, './teste', {'weight': ['weight', 'count'], 'percentage': 'percentage'})

if __name__ == '__main__':
    main()
