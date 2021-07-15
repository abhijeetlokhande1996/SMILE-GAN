import numpy as np
import pandas as pd


def read_csv(file_path, nrows, fields):
    return pd.read_csv(file_path, nrows=nrows, usecols=fields)


def pad_smile_string_right(smile_string, n_factor=0):
    padding_string = " " * n_factor
    return padding_string + smile_string


def smile_to_one_hot_3D(smiles_arr):
    SMILE_LENGTH = len(smiles_arr)
    MAX_LENGTH = max([len(s) for s in smiles_arr])
    for idx in range(len(smiles_arr)):
        smile_string = smiles_arr[idx]
        if len(smile_string) == MAX_LENGTH:
            continue
        else:
            smiles_arr[idx] = pad_smile_string_right(
                smile_string=smile_string, n_factor=MAX_LENGTH - len(smile_string))

    s = set()
    for item in smiles_arr:
        s = s.union(item)

    NCHARS = len(s)

    char_dict = {}
    idx_to_char_dict = {}
    for idx, c in enumerate(s):
        char_dict[c] = idx
        idx_to_char_dict[idx] = c
    # print("SMILE LENGTH: ", SMILE_LENGTH)
    # print("MAX LENGTH: ", MAX_LENGTH)
    # print("NCHARS: ", NCHARS)
    X = np.zeros((SMILE_LENGTH, MAX_LENGTH))
    for i, smile in enumerate(smiles_arr):
        for j, char in enumerate(smile):
            X[i, j] = char_dict[char]
    # print("X.shape: ", X.shape)
    return X, MAX_LENGTH, NCHARS, idx_to_char_dict
