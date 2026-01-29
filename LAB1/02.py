# 2. WAP to take 2D array and convert to 1D array using numpy and pandas

import numpy as np
import pandas as pd

r = int(input())
c = int(input())

arr = []
for i in range(r):
    row = []
    for j in range(c):
        row.append(int(input()))
    arr.append(row)

np_arr = np.array(arr).flatten()
pd_arr = pd.DataFrame(arr).values.flatten()

print(np_arr)
print(pd_arr)
