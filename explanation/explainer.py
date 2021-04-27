import os

import lime
import pickle

class Explainer:
    def __init__(self, pickles_path):
        self.pickles_path = pickles_path

    '''
        Returns the list of the pickles files' path
    '''
    def __get_classification_models_path(self):
        return os.listdir(self.pickles_path)
    #with open('./pickles/classifier_{}.pkl'.format(self.__generate_sufix()), 'wb') as filename:

