import numpy as np

test = np.array([[2, 1],[1, 5],[3,2]])
test2 = np.random.randn(3, 2) / np.sqrt(2)
test3 = np.random.randn(3).astype(np.float).ravel()

print (test2)
print (test2.shape)
horizon = [[5], [7], [0]]

ans = np.dot(test3, test2)

print(ans)