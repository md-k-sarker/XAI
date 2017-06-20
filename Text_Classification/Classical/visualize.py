'''
Created on Jun 20, 2017

@author: sarker
'''

import matplotlib.pyplot as plt
import pickle


figure_file = '../../data/20news-18828/preloaded/fig.dt'

[f1, f2, f3, f4] = pickle.load(open(figure_file, 'rb'))
plt.show()
