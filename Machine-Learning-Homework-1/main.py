from scipy.stats import entropy
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

features = ['X1', 'X2']

X = pd.DataFrame({'X1': [1, 1, 1, 1, 0, 0],
                  'X2': [1, 1, 1, 0, 0, 0]})
Y = pd.Series([1, 1, 2, 3, 2, 3])

X['Y'] = Y

X1 = pd.DataFrame({'X1': [1, 1, 1, 1, 0, 0],
                   'X2': [1, 1, 1, 0, 0, 0]})


def probability_mass_function(X, x):
    return X[x].value_counts(normalize=True)


def H(X, x):
    return entropy(list(probability_mass_function(X, x)), base=2)


def sp_conditional_entropy(X, x, y, value):
    return H(X.query(f'{y}=="{value}"'), x)


def conditional_entropy(X, x, y):
    sum = 0

    for key, value in probability_mass_function(X, y).iteritems():
        sum += sp_conditional_entropy(X, x, y, key) * value

    return sum


def information_gain(X, x, y):
    return H(X, x) - conditional_entropy(X, x, y)


print('information_gain(X1|Y))', information_gain(X, 'X1', 'Y'))
print('information_gain(X2|Y))', information_gain(X, 'X2', 'Y'))

print(
    '\nThe attribute that will be used for the first node of the ID3 tree\nis the one with the highest information gain, '
    'which is X2\n')

decision_tree_classifier = tree.DecisionTreeClassifier(criterion='entropy').fit(X1, Y)
drawing, ax = plt.subplots(figsize=(7, 8))
figure = tree.plot_tree(decision_tree_classifier, ax=ax, fontsize=10, feature_names=features)
plt.show()

new_instances = pd.DataFrame([
    (0, 1)
],
    index=['1'],
    columns=['X1', 'X2'])

new_with_prediction = new_instances.copy()
new_with_prediction['predicted'] = decision_tree_classifier.predict(new_instances)
print('\nClassifying the instance: {X1: 0, X2: 1}:\n\n', new_with_prediction)
