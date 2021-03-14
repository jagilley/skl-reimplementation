from code import k_nearest_neighbor, collaborative_filtering, load_movielens
import pandas as pd
import random
import math

data = load_movielens.load_movielens_data("data/ml-100k")
#print(data)

pd_data = pd.DataFrame(data)
"""
for itc, row in enumerate(data):
    this_row = pd_data.iloc[itc]
    this_row_no_zeroes = this_row[this_row != 0]
    my_median = this_row_no_zeroes.median()
    pd_data.iloc[itc][pd_data.iloc[itc] == 0] = my_median
"""
pd_data = pd_data.replace(0, 2.5)
#print(pd_data)

def getError(N, K, D, A):
    this_data = pd_data
    modified = []
    orig_vals = []
    dimensions = this_data.shape

    remaining = N
    while True:
        changed_x = random.randint(0, (dimensions[0]-1))
        changed_y = random.randint(0, (dimensions[1]-1))
        if this_data.at[changed_x, changed_y] != 2.5:
            orig_vals.append(this_data.at[changed_x, changed_y])
            this_data.at[changed_x, changed_y] = 0
            modified.append((changed_x, changed_y))
            remaining -= 1
        if remaining == 0:
            break
    
    #print("modified is", modified)
    #print("orig_vals is", orig_vals)

    ndarr = this_data.to_numpy()
    for x, y in modified:
        if ndarr[x][y] != 0:
            badval = ndarr[x][y]
            raise AssertionError(f"Badval is {badval}")
    
    cf = collaborative_filtering(ndarr, K, distance_measure=D, aggregator=A)

    errors = []

    for itc, (x_dim, y_dim) in enumerate(modified):
        orig = orig_vals[itc]
        changed = cf[x_dim][y_dim]
        #print("orig is", orig, "and changed is", changed)
        error = (orig-changed)**2
        errors.append(error)
    
    mean = sum(errors)/len(errors)
    mse = math.sqrt(mean)
    return mse
"""
# q1
for n_val in [5, 10, 20, 40]:
    print("final error for n =", n_val, "is", getError(n_val, 3, "euclidean", "mean"))

# q2
for dist in ["euclidean", "manhattan", "cosine"]:
    print("error for", dist, "distance measure is", getError(7, 3, dist, "mean"))

# q3
for k_val in [1, 3, 7, 11, 15, 31]:
    print("error for k_val", k_val, "is", getError(1, k_val, "euclidean", "mean"))
"""
# q4
for agg in ["mean", "median", "mode"]:
    print("error for aggregator", agg, "is", getError(7, 3, "euclidean", agg))