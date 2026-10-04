"""Run once to export PCA + scaler + gene names for the demo app."""
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X    = np.load("combat_corrected_matrix.npy")
pids = np.load("combat_patient_ids.npy", allow_pickle=True).tolist()

gene_df       = pd.read_csv("genomic_matrix.csv")
all_gene_cols = [c for c in gene_df.columns if c != "patient_id"]
matched       = gene_df[gene_df["patient_id"].isin(set(pids))]
stds          = matched[all_gene_cols].values.std(axis=0)
gene_names    = np.array(all_gene_cols)[stds > 0]

assert X.shape[1] == len(gene_names), f"Mismatch: {X.shape[1]} vs {len(gene_names)}"
np.save("inference_gene_names.npy", gene_names)
print(f"[1/3] Saved {len(gene_names)} gene names → inference_gene_names.npy")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open("inference_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("[2/3] Saved scaler → inference_scaler.pkl")

pca = PCA(n_components=480, random_state=42)
pca.fit(X_scaled)
with open("inference_pca.pkl", "wb") as f:
    pickle.dump(pca, f)
print(f"[3/3] Saved PCA (480 components, {np.sum(pca.explained_variance_ratio_)*100:.1f}% variance) → inference_pca.pkl")
print("Done. All artifacts ready for app.py.")
