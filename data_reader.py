import pandas as pd
import numpy as np


def read_train_dataset(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        line = line.rstrip(',\n').strip() + '\n'
        elements = line.strip().split(',')
        
        last_element = elements[-1]
        other_elements = elements[:-1]
        
        while len(other_elements) < 20:
            other_elements.append(np.nan)
        
        row = other_elements + [last_element]
        data.append(row)

    columns = [f'col_{i}' for i in range(20)] + ['label']
    return pd.DataFrame(data, columns=columns)


def read_test_dataset(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        line = line.rstrip(',\n').strip() + '\n'
        elements = line.strip().split(',')
        
        while len(elements) < 20:
            elements.append(np.nan)
        
        data.append(elements)

    columns = [f'col_{i}' for i in range(20)]
    return pd.DataFrame(data, columns=columns)
