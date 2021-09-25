import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = '../dissertation2/csvs'
EXPERIMENTS = ['baseline', 'baseline_200', 'semi', 'semi_200', 'independent', 'independent_200']
RESULT_COLUMNS = ['fscore', 'specificity', 'accuracy', 'recall', 'precision']
COLORS = ['C1', 'C2', 'C3', 'C4', 'C5']
YTICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

def generate_radar_chart(experiment):

    df = pd.read_csv('{}/{}_top.csv'.format(BASE_PATH, experiment))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")

    # theta has 5 different angles, and the first one repeated
    theta = np.arange(len(df) + 1) / float(len(df)) * 2 * np.pi

    for row in range(len(df)):
        # values has the 5 values from 'Col B', with the first element repeated
        values = df.iloc[row][RESULT_COLUMNS].values
        values = np.append(values, values[0])

        # draw the polygon and the mark the points for each angle/value combination
        ax.plot(theta, values, color=COLORS[row], marker="o", label="Name of {}".format(row))
        plt.xticks(theta[:-1], [col if col != 'fscore' else 'f1-score' for col in RESULT_COLUMNS], color=COLORS[row], size=12)
        ax.tick_params(pad=15) # to increase the distance of the labels to the plot

        ax.set_yticks(YTICKS)
        ax.set_yticklabels(map(str, YTICKS))

    # plt.title('{} experiment results'.format(experiment.replace('_', ' ')))
    plt.savefig('charts/{}_radar.jpg'.format(experiment))


for experiment in EXPERIMENTS:
    generate_radar_chart(experiment)