import numpy as np
c=np.linspace(1,100,100)
c_prime=c.reshape((10,10))
print(type(c_prime))
print(type(c))
print(c.shape)
print(c_prime.shape)
print(c_prime.flatten().shape)