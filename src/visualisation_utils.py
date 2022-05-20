import numpy as np
import scipy.sparse as sp
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

def concat_chr_arrays(results:np.ndarray, chr:np.ndarray):
    if len(chr) != len(results):
        raise ValueError("The chromosome and the corresponding output have not the same size")
    outputs = results[0][2]
    labels = sp.csr_matrix(chr[0][0][:,1]) # label is a sparse array
    for tr_index in range(1, len(chr)):
        assert results[tr_index][0] == chr[tr_index][1], "les transcrit ne sont pas dans le même ordre dans les deux fichiers"
        outputs = np.concatenate((outputs, results[tr_index][2]))
        labels=sp.hstack((labels, sp.coo_matrix(chr[tr_index][0][:,1])), format="csr")
    return outputs, labels

def get_inference(array:np.ndarray, n:int)->np.ndarray:
    """https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array"""
    predictions = np.zeros_like(array)
    vector = array.flatten()
    indices = np.argpartition(vector, -n)[-n:]
    indices = indices[np.argsort(-vector[indices])]
    prediction_indices = np.unravel_index(indices, array.shape)
    predictions[prediction_indices] = 1
    return predictions

def plot_distribution_graph(outputs:np.ndarray, labels:np.ndarray):
    df = pd.DataFrame({"predictions":outputs, "labels":labels})
    ax = sns.histplot(data=df, x="predictions", hue="labels", multiple="layer", binwidth=0.01)
    ax.axvline(x=0.5, linestyle="--", color="black")
    ax.set_ylim(0,450)

def roc_pr_auc(outputs:np.ndarray, labels:np.ndarray):
    precision, recall, _ = precision_recall_curve(labels, outputs)
    return roc_auc_score(labels, outputs), auc(recall, precision)

def plot_bar_roc_pr_by_chromosomes(roc_pr_df:pd.DataFrame):
    roc_pr_df["Model"] = [s[:-4] for s in roc_pr_df["Model"]]
    transformed = roc_pr_df.melt(id_vars='Model').rename(columns=str.title)

    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=transformed, x="Model", y='Value',hue='Variable')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
