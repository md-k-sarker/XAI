'''
Created on Jun 20, 2017

@author: sarker
'''

import matplotlib.pyplot as plt
import pickle


figure_file = '../../data/20news-18828/2_class/model/fig.pickle'

'''after loading from pickle zooming is not working'''
[fig1,fig2,fig3,fig4] = pickle.load(open(figure_file, 'rb'))
plt.show()
