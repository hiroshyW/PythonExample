import matplotlib.pyplot as plt
from numpy.random import rand
from sklearn.cluster import KMeans

x = rand(50, 2) * 100

plt.scatter(x[:, 0], x[:, 1])
plt.show()


pred = KMeans(n_clusters=5).fit_predict(x)

plt.scatter(x[:, 0], x[:, 1], c=pred)
plt.show()
