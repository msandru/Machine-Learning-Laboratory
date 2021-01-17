from matplotlib.colors import ListedColormap
from scipy.stats import norm

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pandas as pd

d = pd.DataFrame({
    'X': [10, 11, 12, 19, 20, 22]
})

c = ListedColormap(['red', 'green'])

mean_2 = 15
mean_1 = mean_2 - 8

means = [mean_1, mean_2]
variances = [1.5, 1.5]


def plot_pdfs(means, variances, ax, colours):
    """ Plot the pdfs for given means and variances """
    for i in range(0, 2):
        mean, var = means[i], variances[i]
        x = np.linspace(norm.ppf(0.01, mean, np.sqrt(var)), norm.ppf(0.99, mean, var), 100)
        pdf = norm.pdf(x, mean, var)
        ax.plot(x, pdf, color=colours(i), label=f"$\mu_{i}$")
        ax.legend()

p_1 = norm.pdf(d, means[0], np.sqrt(variances[0]))
p_2 = norm.pdf(d, means[1], np.sqrt(variances[1]))

E0 = p_1 / (p_1 + p_2)
E1 = p_2 / (p_1 + p_2)
print("E for j=0:\n", E0)
print("E for j=1:\n", E1)

from matplotlib.colors import LinearSegmentedColormap

c_grad = LinearSegmentedColormap.from_list('mygrad', [c(0), c(1)])

fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(d, np.zeros_like(d), c=E1, cmap=c_grad)
plot_pdfs(means, variances, ax, c)
plt.title('Expectation 1')
plt.show()

x = d.to_numpy()
means[0] = (E0 * x).sum() / E0.sum()
means[1] = (E1 * x).sum() / E1.sum()
print("Updated means are: ", means)

fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(d, np.zeros_like(d))
plot_pdfs(means, variances, ax, c)
plt.title("Maximisation 1")
plt.show()

p_1 = norm.pdf(d, means[0], np.sqrt(variances[0]))
p_2 = norm.pdf(d, means[1], np.sqrt(variances[1]))

E0 = p_1 / (p_1 + p_2)  # E for j=0
E1 = p_2 / (p_1 + p_2)  # E for j=1

fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(d, np.zeros_like(d), c=E1, cmap=c_grad)
plot_pdfs(means, variances, ax, c)
plt.title('Expectation 2')
plt.show()

means[0] = (E0 * x).sum() / E0.sum()
means[1] = (E1 * x).sum() / E1.sum()

fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(d, np.zeros_like(d))
plot_pdfs(means, variances, ax, c)
plt.title("Maximisation 2")
plt.show()

p_1 = norm.pdf(d, means[0], np.sqrt(variances[0]))
p_2 = norm.pdf(d, means[1], np.sqrt(variances[1]))

E0 = p_1 / (p_1 + p_2)  # E for j=0
E1 = p_2 / (p_1 + p_2)  # E for j=1

fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(d, np.zeros_like(d), c=E1, cmap=c_grad)
plot_pdfs(means, variances, ax, c)
plt.title('Expectation 3')
plt.show()

means[0] = (E0 * x).sum() / E0.sum()
means[1] = (E1 * x).sum() / E1.sum()

fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(d, np.zeros_like(d))
plot_pdfs(means, variances, ax, c)
plt.title("Maximisation 3")
plt.show()
