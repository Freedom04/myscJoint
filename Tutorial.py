import scJoint.process_db as process_db
import h5py
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(1)


# ------------------------ Prepare input for scJoint -------------------------------
# rna_h5_files = ["data/citeseq_control_rna.h5"] 
# rna_label_files = ["data/citeseq_control_cellTypes.csv"] # csv file

# atac_h5_files = ["data/asapseq_control_atac.h5"]
# atac_label_files = ["data/asapseq_control_cellTypes.csv"]

# rna_protein_files = ["data/citeseq_control_adt.h5"] 
# atac_protein_files = ["data/asapseq_control_adt.h5"] 


# h5 = h5py.File(rna_h5_files[0], "r")
# h5_data = h5['matrix/data']
# h5_barcode = h5['matrix/barcodes']
# h5_features = h5['matrix/features']

# # convert to a compressed sparse row matrix
# process_db.data_parsing(rna_h5_files, atac_h5_files)
# process_db.data_parsing(rna_protein_files, atac_protein_files)


# rna_label = pd.read_csv(rna_label_files[0], index_col = 0)
# atac_label = pd.read_csv(atac_label_files[0], index_col = 0)

# # map rna labels to index 
# process_db.label_parsing(rna_label_files, atac_label_files)


# ------------------------------ Visualisation -------------------------------------
rna_embeddings = np.loadtxt('./output/citeseq_control_rna_embeddings.txt')
atac_embeddings = np.loadtxt('./output/asapseq_control_atac_embeddings.txt')
embeddings =  np.concatenate((rna_embeddings, atac_embeddings))
tsne_results = TSNE(perplexity=30, n_iter = 1000).fit_transform(embeddings)
tsne_results.shape
df = pd.DataFrame()
df['tSNE1'] = tsne_results[:,0]
df['tSNE2'] = tsne_results[:,1]


rna_labels = np.loadtxt('./data/citeseq_control_cellTypes.txt')
atac_labels = np.loadtxt('./data/asapseq_control_cellTypes.txt')
atac_predictions = np.loadtxt('./output/asapseq_control_atac_knn_predictions.txt')
labels =  np.concatenate((rna_labels, atac_predictions))
label_to_idx = pd.read_csv('./data/label_to_idx.txt', sep = '\t', header = None)
label_to_idx.shape
label_dic = []
for i in range(label_to_idx.shape[0]):
    label_dic = np.append(label_dic, label_to_idx[0][i][:-2])

common_label_cnt = 0
correct = 0
for i in range(len(atac_labels)):
    if atac_labels[i] >= 0:
        common_label_cnt += 1
        if atac_labels[i] == atac_predictions[i]:
            correct += 1

print(correct/common_label_cnt)


data_label = np.array(["CITE-seq", "ASAP-seq"])
temp = np.repeat(data_label, [rna_embeddings.shape[0], atac_embeddings.shape[0]], axis=0)
temp2 = label_dic[labels.astype(int)]
df['data'] = np.repeat(data_label, [rna_embeddings.shape[0], atac_embeddings.shape[0]], axis=0)
df['predicted'] = label_dic[labels.astype(int)]

df.to_csv("results.csv", encoding="utf-8", index=False)

plt.figure(figsize=(10,10))
sns.scatterplot(
    x = "tSNE1", y = "tSNE2",
    hue = "data",
    palette = sns.color_palette("tab10", 2),
    data = df,
    legend = "full",
    alpha = 0.3
)

plt.figure(figsize=(10,10))
sns.scatterplot(
    x = "tSNE1", y = "tSNE2",
    hue = "predicted",
    palette = sns.color_palette("Set2", 7),
    data = df,
    legend = "full",
    alpha = 0.3
)

plt.show()