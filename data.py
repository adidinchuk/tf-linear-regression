# import numpy as np
import hyperparams as hp


# breast-cancer-wisconsin.data
def load_data(name):

    file_name = hp.data_dir + '\\' + name

    with open(file_name) as fp:
        lines = fp.read().split("\n")

    data = [line.split(',') for line in lines]
    return data
