import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



# embedding = np.load('embeddings/netmf_cora.out.id.npy')

embedding = pd.read_csv('embeddings/line/cora.emb',delim_whitespace=True)

# sort embedding after node id
embedding[embedding[:, 0].sort()]
node_ids = embedding[:, 0]


X = pd.DataFrame(data=embedding[:, 1:])


# sort labels after node_ids
data = pd.read_csv('labels/cora.content',
                   sep="\t", header=None, index_col=0)
data = data[1434]
y = data.sort_index()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1, train_size=0.8)


# classifiers = [('qda', QuadraticDiscriminantAnalysis()), ('lda', LinearDiscriminantAnalysis(
# )), ('rf', RandomForestClassifier(criterion='entropy', n_estimators=250, max_features='sqrt'))]
# model = StackingClassifier(classifiers, cv=10)

model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('F1-Micro', f1_score(y_test, y_pred, average='micro'))
print('F1-Macro', f1_score(y_test, y_pred, average='macro'))

# print(X)
# print(y)
