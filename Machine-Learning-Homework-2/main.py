from scipy.stats import bernoulli
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from sklearn import tree

size = 10000


def correlated_df(corr):
    w1 = bernoulli.rvs(0.5, size=size, random_state=1)
    d = pd.DataFrame({'w1': w1})

    mask = bernoulli.rvs(corr, size=size, random_state=2)
    random = bernoulli.rvs(0.5, size=size, random_state=3)
    d['w2'] = d['w1'] & mask | random & ~mask
    d['mask'] = mask
    d['random'] = random
    d['y'] = d['w1'] & ~ d['w2']
    # print(d)
    return d

correlations = []
training_errors = []

for i in np.arange(0.0, 1.0, 0.01):
    correlations.append(i)

    d = correlated_df(i)
    print("Correlation: ", pearsonr(d['w1'], d['w2']))
    print("Correlation corr: ", i)

    features = ['w1', 'w2']

    X = d[features]
    y = d['y']

    cl = BernoulliNB(alpha=1e-10).fit(X, y)
    print("Accuracy:", cl.score(X, y))
    print("Training error:", 1 - cl.score(X, y))
    print("\n\n\n")
    training_errors.append(1 - cl.score(X, y))

plt.plot(correlations, training_errors)
plt.show()

correlations_tree = []
training_errors_tree = []

for i in np.arange(0.0, 1.0, 0.01):
    correlations_tree.append(i)

    d = correlated_df(i)
    print("Correlation: ", pearsonr(d['w1'], d['w2']))
    print("Correlation corr: ", i)

    features = ['w1', 'w2']

    X = d[features]
    y = d['y']

    classifier = tree.DecisionTreeClassifier(criterion='entropy')
    dt = classifier.fit(X, y)

    print("Accuracy:", dt.score(X, y))
    print("Training error:", 1 - dt.score(X, y))
    print("\n\n\n")

    training_errors_tree.append(1 - dt.score(X, y))


plt.plot(correlations_tree, training_errors_tree)
plt.show()
