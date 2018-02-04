# test for compute_displacement
import numpy as np
from hw7_release import detection
test_array = np.array([[0,1],[1,2],[2,3],[3,4]])
test_shape = (6,6)
mu, std = detection.compute_displacement(test_array, test_shape)
assert(np.all(mu == [1,0]))
assert(np.sum(std-[ 1.11803399,  1.11803399])<1e-5)
print("Your implementation is correct!")
