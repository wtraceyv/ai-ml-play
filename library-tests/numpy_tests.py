# https://numpy.org/devdocs/user/quickstart.html
#######################################################3

import numpy as np


grid = np.zeros((5, 5), dtype=np.int16)
print("Bunch of zeros (could be 1s, or random):")
print(grid)

squares = [x**2 for x in range(0, 20)]
cubes = [x**3 for x in range(0, 20)]
print("List comp squares, cubes:")
print(squares)
print(cubes)

my_tensor = np.array(squares).reshape(4,5)
print("Reshaped:")
print(my_tensor)

# If the shape lines up, -1 makes it automatically
# figure out the correct dimension size (otherwise error)
my_tensor = my_tensor.reshape(2,-1)
print("Reshaped, auto calc the 2nd dimension:")
print(my_tensor)

print("How about transposed:")
print(my_tensor.T)

print("Horizontal stack squares, cubes:")
print(np.hstack((squares, cubes)))

print("Vertical stack squares, cubes:")
print(np.vstack((squares, cubes)))

print("Convolve these babies (flip second, then multiply add sliding second across first):")
c1 = np.array([1,2,3])
c2 = np.array([4,5,6])
print(c1)
print(c2)
print(np.convolve(c1, c2))

# Making copies:
# my_tensor.view() is shallow copy (pointer to original)
# my_tensor.copy() is deep copy (independent copy)

# Try to use built in functions for anything common like
# Cumulative sum, complex num manip, cross dot outer products,
# statistics like cov mean std var, ops like sort max min, etc.
