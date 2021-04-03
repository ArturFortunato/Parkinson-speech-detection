class Metrics:
    def __init__(self, predicted, labels):
        self.__matches(predicted, labels)

    # Should be called if the instance is needed for multiple classification experiements
    def set_results(self, predicted, labels):
        self.__matches(predicted, labels)
        
    def __matches(self, predicted, labels):
        if len(predicted) != len(labels):
            raise Exception("Predicted(len {}) and labels(len {}) should match".format(len(predicted), len(labels)))
        
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.total = len(predicted)

        for i in range(self.total):
            # Model classified correctly a patient
            if predicted[i] == labels[i] and predicted[i] == 1:
                self.tp += 1

            # Model classified correctly a control
            elif predicted[i] == labels[i] and predicted[i] == 0:
                self.tn += 1

            # Model wrongly classified   a patient
            elif predicted[i] == 0:
                self.fn += 1
            
            # Model wrongly classified   a control
            elif predicted[i] == 1:
                self.fp += 1

    def accuracy(self):
        return self.tp / self.total

    def precision(self):
        if self.tp + self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    def recall(self):
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def specificity(self):
        if self.tn + self.fp == 0:
            return 0
        return self.tn / (self.tn + self.fp)

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r)

    def generate_report(self, name, threshold):
        report_format = "{} report \n\n\tSeparation threshold: {}\n\n\tAccuracy:             {}\n\n\tPrecision:            {}\n\n\tRecall:               {}\n\n\tSpecificity:          {}\n\n\tF1-Score:             {}"
        return report_format.format(name, threshold, self.accuracy(), self.precision(), self.recall(), self.specificity(), self.f1_score())
