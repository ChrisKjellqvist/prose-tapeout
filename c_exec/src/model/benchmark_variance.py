import numpy as np
import random as rand
import time

# generate n_times sets of 768 random numbers
n_times = 10000
rand_numbers = np.random.rand(768)
print(rand_numbers.shape)

# count how many microseconds it takes to compute the variance
start_time = time.time()
for i in range(n_times):
    np.var(rand_numbers)
end_time = time.time()
print(str(end_time - start_time))
print(f"Time taken: {(end_time - start_time)/n_times*1000*1000}µs")