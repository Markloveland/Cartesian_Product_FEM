import numpy as np

a = np.array([0.5,1.0,-0.5,11.0])
b = np.array([2.0,-5.0,-1.0,3.0])

print('array a')
print(a)
print('array b')
print(b)
print('a>0')
print(a[a>0])
print('b>0')
print(b[b>0])
print('a where a>0 and b>0')
print(a[np.logical_and(a>0,b>0)])
