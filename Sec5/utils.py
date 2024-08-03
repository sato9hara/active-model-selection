import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# data
def load_data(data, train_size=5000, test_size=500, random_state=0):
    x, y = datasets.load_svmlight_file('./data/%s' % (data,))
    x = np.array(x.todense())
    y = LabelEncoder().fit_transform(y.astype(int))
    x, _, y, _ = train_test_split(x, y, train_size=train_size+test_size, random_state=random_state)
    x, xte, y, yte = train_test_split(x, y, train_size=train_size, random_state=random_state+1)
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    xte = scaler.transform(xte)
    return x, y, xte, yte

# fit functions
def fit_logreg(x, y, option=0, random_state=0):
    clf = LogisticRegressionCV(multi_class='multinomial', max_iter=int(1e+6), random_state=random_state)
    clf.fit(x, y)
    return clf

def fit_random_forest(x, y, option=10, random_state=0):
    n_estimators = 100
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=option, random_state=random_state).fit(x, y)
    clf.fit(x, y)
    return clf

def fit_mlp(x, y, option=100, random_state=0):
    clf = MLPClassifier(hidden_layer_sizes=(option, option//2), max_iter=int(1e+3), random_state=random_state)
    clf.fit(x, y)
    return clf