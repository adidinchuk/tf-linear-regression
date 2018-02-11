'''
By adidinchuk park. adidinchuk@gmail.com.
https://github.com/adidinchuk/tf-linear-regression
'''

import data as d
import numpy as np
from network import Network
import hyperparams as hp

data = d.load_data('breast-cancer-wisconsin.data.txt')

# strip out only columns 2, 3 and 4
# 0,1 - inputs
# 2 - labels
input_data = [[int(row[1]), int(row[2])] for row in data]
output_data = [[int(row[3])] for row in data]

nwk = Network(loss_function='l2')
nwk.train(np.array(input_data), np.array(output_data), plot=True, lr=hp.learning_rate,
          batch_size=hp.batch_size, epochs=hp.epochs, validation_size=hp.validation_size)
