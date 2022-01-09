import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import sys

algorithm = sys.argv[1]
dataset = sys.argv[2]

columns = ["node_id"]

for i in range(128):
    columns.append("dim"+str(i+1))

emb = pd.read_csv(f'embeddings/{algorithm}/{dataset}.emb', delim_whitespace=True, names=columns).sort_values(by=['node_id'], ascending=True)
emb = emb.sample(12000)

dimensions = []
for i in range(128):
    dimensions.append("dim"+str(i+1))

nodes = []
for index, row in emb.iterrows():
    nodes.append(row["node_id"])

pairs = []
for node1 in nodes:
    for node2 in nodes:
        pairs.append([node1, node2])

pairs = pd.DataFrame.from_records(pairs)
pairs.columns=["node1", "node2"]

node_embeddings = {}
for index, row in emb.iterrows():
    node_embeddings[row["node_id"]] = [row[dim] for dim in dimensions]

edges = pd.read_csv(f'edgelists/{dataset}.edgelist', delim_whitespace=True, names=["node1", "node2"])
edges = edges.reset_index(drop=True)
edges = edges.sample(350000)

negative = pd.concat([edges, pairs]).drop_duplicates(keep=False)
negative = negative.sample(len(edges))
negative = negative.reset_index(drop=True)

ones = [[1]] * len(edges)
ones = pd.DataFrame.from_records(ones)
ones.columns=["edge"]
positive_edges = pd.concat([edges,ones], axis=1)

zeros = [[0]] * len(negative)
zeros = pd.DataFrame.from_records(zeros)
zeros.columns=["edge"]
negative_edges = pd.concat([negative,zeros], axis=1)

labeled_dataset = pd.concat([positive_edges, negative_edges])

labeled_dataset = labeled_dataset.sample(frac=1)

labeled_dataset['node1'] = labeled_dataset['node1'].map(node_embeddings)
labeled_dataset['node2'] = labeled_dataset['node2'].map(node_embeddings)
labeled_dataset = labeled_dataset.dropna()

training_set, validation_set = train_test_split(labeled_dataset, test_size = 0.5, random_state = 21)

X_train = training_set[["node1", "node2"]]
Y_train = training_set[["edge"]]
X_val = validation_set[["node1", "node2"]]
Y_val = validation_set[["edge"]]

print("Splitting...")
X_train_split = pd.concat([pd.DataFrame(X_train['node1'].to_list()),pd.DataFrame(X_train['node2'].to_list())], axis=1)
# with open('airport-X_train_split.pkl', 'wb') as f:
#     pickle.dump(X_train_split,f)

logisticRegr = LogisticRegression(max_iter=10000)
logisticRegr.fit(X_train_split, Y_train.values.ravel())

X_val_split = pd.concat([pd.DataFrame(X_val['node1'].to_list()),pd.DataFrame(X_val['node2'].to_list())], axis=1)
# with open('airport-X_val_split.pkl', 'wb') as f:
#     pickle.dump(X_val_split,f)

Y_pred = logisticRegr.predict(X_val_split)


def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

cm = confusion_matrix(Y_pred, Y_val)

with open(f"scores/{dataset}_{algorithm}_scores.txt", "w") as f:
    # Writing data to a file
    f.write("Macro F1: ")
    f.write(str(f1_score(Y_val, Y_pred, average='macro')))
    f.write("\n")
    f.write("Micro F1: ")
    f.write(str(f1_score(Y_val, Y_pred, average='micro')))  
    f.write("\n")
    f.write(f"Accuracy of Link Prediction :{accuracy(cm)}")
