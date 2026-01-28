# 1. WAP to take 1D array and convert to numpy and pandas 

import numpy as np
import pandas as pd

n = int(input())
arr = []

for i in range(n):
    arr.append(int(input()))

np_arr = np.array(arr)
pd_arr = pd.Series(arr)

print(np_arr)
print(pd_arr)
