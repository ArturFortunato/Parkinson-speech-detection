import matplotlib.pyplot as plt
import pandas as pd

def generate(csv_file, image_file, x, y):
    csv = pd.read_csv(csv_file, index_col=None)

    features = list(csv[x])
    for i in range(len(features)):
        features[i] = features[i].replace("$\Delta$", "Δ") 

    print(features)

    return

    weight = list(csv[y])
    feature_spaces = [4 * i for i in range(len(features))]

    plt.figure(figsize=(100, 70))
    plt.bar(feature_spaces, weight, align='edge')
    plt.xticks(feature_spaces, features, rotation=90)

    for t in plt.xticks()[1]:
        t.set_fontsize(35)

    for t in plt.yticks()[1]:
        t.set_fontsize(35)

    plt.savefig(image_file)

generate('/home/alex/Documents/IST/Thesis/Parkinson-speech-detection/explainability/teste_by_weight.csv', './feature_by_weight.png', 'feature', 'weight')
generate('/home/alex/Documents/IST/Thesis/Parkinson-speech-detection/explainability/teste_by_percentage.csv', './feature_by_percentage.png', 'feature', 'percentage')
