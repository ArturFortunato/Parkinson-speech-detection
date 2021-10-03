import pandas as pd

csvs = {
        'fralusopark': '../features/fralusopark/fralusopark_complete.csv',
        'gita': '../features/gita/gita_complete.csv',
        'mdvr_kcl': '../features/mdvr_kcl/mdvr_kcl_complete.csv'
}

for dataset in csvs:
    csv = pd.read_csv(csvs[dataset], index_col=None)

    total_duration = csv.shape[0] / 100

    print('{}: {} minutes and {} seconds'.format(dataset, total_duration // 60, total_duration % 60))
