from random import random

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow.keras import layers, Model
import umap
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from tensorflow.keras.callbacks import EarlyStopping


SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
umap_model = umap.UMAP(random_state=SEED)

HYPERPARAMS = {
    'latent_dim': 32,
    'tcr_ngram_features': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'patience': 5
}

LATENT_DIM = HYPERPARAMS['latent_dim']
TCR_NGRAM_FEATURES = HYPERPARAMS['tcr_ngram_features']

os.makedirs("output", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# --- Step 1: Load filtered cluster 8 barcodes ---
# barcodes = pd.read_csv("~/DeepLearningProject/output/cluster8_barcodes.csv")["x"].str.replace("-1$", "", regex=True)
barcodes = pd.read_csv("~/DeepLearningProject/output/cluster8_barcodes.csv", index_col=0)
barcodes = barcodes.index.to_series()  # Turn index into a Series
barcodes = barcodes.str.replace("-1_1$", "", regex=True)  # Remove -1_1 suffix



# --- Step 2: Load full RNA matrix and metadata ---
rna_all = pd.read_csv("~/DeepLearningProject/output/integrated_expression_matrix.csv", index_col=0)
meta_all = pd.read_csv("~/DeepLearningProject/output/integrated_metadata.csv")
rna_all.index = rna_all.index.str.replace("-1$", "", regex=True)

rna_all.index = rna_all.index.str.replace("-1_1$", "", regex=True)

# Subset RNA by cluster 8 barcodes
rna_cluster8 = rna_all.loc[rna_all.index.intersection(barcodes)]

print("Number of cluster8 barcodes:", len(barcodes))
print("Number of RNA cells (after filtering):", rna_cluster8.shape[0])



# --- Step 3: Load TCR data and match barcodes ---
tcr_df = pd.read_csv("/Users/pradeepmaripala/Downloads/Deep_Learning_Projec/vdj_analysis/filtered_contig_annotations.csv")
tcr_df = tcr_df[['barcode', 'chain', 'cdr3', 'v_gene', 'j_gene']].drop_duplicates('barcode').fillna('')

tcr_df['barcode'] = tcr_df['barcode'].str.replace("-1$", "", regex=True)

tcr_df.set_index("barcode", inplace=True)

# Intersect barcodes for matching RNA and TCR
common_barcodes = rna_cluster8.index.intersection(tcr_df.index)

print("Number of TCR barcodes:", tcr_df.shape[0])
print("Number of matched barcodes:", len(common_barcodes))
print("Example barcodes from RNA matrix:")
print(rna_all.index[:10])

print("Example barcodes from cluster8 file:")
print(barcodes[:10])


rna_matrix = rna_cluster8.loc[common_barcodes].values
tcr_strings = tcr_df.loc[common_barcodes][['cdr3', 'v_gene', 'j_gene']].astype(str).apply(lambda x: '_'.join(x), axis=1)
vec = CountVectorizer(analyzer='char', ngram_range=(2, 3), max_features=TCR_NGRAM_FEATURES)
tcr_matrix = vec.fit_transform(tcr_strings).toarray()

# --- Step 4: Define MVAE model ---

def build_encoder(input_dim, latent_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    return Model(inputs, [z_mean, z_log_var])

def build_decoder(latent_dim, output_dim):
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(output_dim)(x)
    return Model(inputs, outputs)

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class MVAE(Model):
    def __init__(self, enc_rna, enc_tcr, dec_rna, dec_tcr, latent_dim):
        super().__init__()
        self.encoder_rna = enc_rna
        self.encoder_tcr = enc_tcr
        self.decoder_rna = dec_rna
        self.decoder_tcr = dec_tcr
        self.sampling = Sampling()
        self.latent_dim = latent_dim

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def train_step(self, data):
        x_rna, x_tcr = data[0]
        x_tcr = tf.cast(x_tcr, dtype=tf.float32)
        with tf.GradientTape() as tape:
            z_mean_rna, z_log_var_rna = self.encoder_rna(x_rna)
            z_mean_tcr, z_log_var_tcr = self.encoder_tcr(x_tcr)
            z_mean = (z_mean_rna + z_mean_tcr) / 2
            z_log_var = (z_log_var_rna + z_log_var_tcr) / 2
            z = self.sampling((z_mean, z_log_var))
            x_rna_recon = self.decoder_rna(z)
            x_tcr_recon = self.decoder_tcr(z)
            recon_loss_rna = tf.reduce_mean(tf.square(x_rna - x_rna_recon))
            recon_loss_tcr = tf.reduce_mean(tf.square(x_tcr - x_tcr_recon))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = recon_loss_rna + recon_loss_tcr + kl_loss
            tf.print("total loss:", total_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

# --- Step 5: Build and Train MVAE ---
enc_rna = build_encoder(rna_matrix.shape[1], LATENT_DIM)
enc_tcr = build_encoder(tcr_matrix.shape[1], LATENT_DIM)
dec_rna = build_decoder(LATENT_DIM, rna_matrix.shape[1])
dec_tcr = build_decoder(LATENT_DIM, tcr_matrix.shape[1])

model = MVAE(enc_rna, enc_tcr, dec_rna, dec_tcr, LATENT_DIM)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=HYPERPARAMS['learning_rate']))
model.fit([rna_matrix, tcr_matrix], epochs=50, batch_size=32)

# --- Step 6: Latent UMAP ---
z_mean_rna, _ = model.encoder_rna(rna_matrix)
z_mean_tcr, _ = model.encoder_tcr(tcr_matrix)
latent = (z_mean_rna + z_mean_tcr) / 2
latent_umap = umap.UMAP(random_state=42).fit_transform(latent.numpy())


plt.figure(figsize=(6, 5))
plt.scatter(latent_umap[:, 0], latent_umap[:, 1], s=10, c='blue')
plt.title("Latent UMAP - Cluster 8 Cells")
plt.tight_layout()
plt.savefig("output/cluster8_latent_umap.png")
plt.show()

# --- Step 7: Cross-Modal Prediction ---
z_rna, log_rna = model.encoder_rna(rna_matrix)
z_sampled = model.sampling((z_rna, log_rna))
tcr_pred = model.decoder_tcr(z_sampled)

z_tcr, log_tcr = model.encoder_tcr(tcr_matrix)
z_sampled2 = model.sampling((z_tcr, log_tcr))
rna_pred = model.decoder_rna(z_sampled2)

np.save("output/cluster8_tcr_predicted.npy", tcr_pred.numpy())
np.save("output/cluster8_rna_predicted.npy", rna_pred.numpy())

#testing

barcodes_all = common_barcodes.to_numpy()

rna_train, rna_test, tcr_train, tcr_test, barcodes_train, barcodes_test = train_test_split(
    rna_matrix,
    tcr_matrix,
    barcodes_all,
    test_size=0.2,
    random_state=42
)

# Encode the TRAIN data only
z_mean_rna_train, _ = model.encoder_rna(rna_train)
z_mean_tcr_train, _ = model.encoder_tcr(tcr_train)
latent_train = (z_mean_rna_train + z_mean_tcr_train) / 2

# UMAP and KMeans on train latent space
latent_umap = umap.UMAP(random_state=42).fit_transform(latent_train.numpy())


# 2. Train

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='loss',
    patience=HYPERPARAMS['patience'],
    restore_best_weights=True
)


history = model.fit(
    [rna_train, tcr_train],
    epochs=HYPERPARAMS['epochs'],
    batch_size=HYPERPARAMS['batch_size'],
    verbose=1,
    callbacks=[early_stopping]
)

# 3. Evaluating on Test
z_mean_rna, z_log_var_rna = model.encoder_rna(rna_test)
z_mean_tcr, z_log_var_tcr = model.encoder_tcr(tcr_test)

z_mean = (z_mean_rna + z_mean_tcr) / 2
z_log_var = (z_log_var_rna + z_log_var_tcr) / 2

z = model.sampling((z_mean, z_log_var))

rna_recon = model.decoder_rna(z)
tcr_recon = model.decoder_tcr(z)

tcr_test = tf.cast(tcr_test, dtype=tf.float32)

# Calculating test reconstruction losses
rna_recon_loss = tf.reduce_mean(tf.square(rna_test - rna_recon)).numpy()
tcr_recon_loss = tf.reduce_mean(tf.square(tcr_test - tcr_recon)).numpy()

print(f"Test RNA reconstruction loss: {rna_recon_loss:.4f}")
print(f"Test TCR reconstruction loss: {tcr_recon_loss:.4f}")

plt.figure(figsize=(6, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.tight_layout()
plt.savefig("output/training_loss_curve.png")
plt.show()

#k-means

def find_best_k(latent_data, k_range=(2, 6)):
    best_score = -1
    best_k = None
    best_labels = None

    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(latent_data)
        score = silhouette_score(latent_data, labels)
        print(f"Silhouette score for k={k}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    print(f"\n✅ Best number of clusters: {best_k} (silhouette = {best_score:.4f})")
    return best_k, best_labels

best_k, cluster_labels = find_best_k(latent_train.numpy(), k_range=(2, 6))


kmeans = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = kmeans.fit_predict(latent_train.numpy())

print("Cluster labels assigned:", np.unique(cluster_labels))

plt.figure(figsize=(6, 5))
plt.scatter(latent_umap[:, 0], latent_umap[:, 1], c=cluster_labels, cmap='coolwarm', s=20)
plt.title("MVAE Latent UMAP Colored by KMeans Clusters")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.tight_layout()

plt.savefig("output/cluster8_latent_umap_kmeans.png")
plt.show()

latent_df = pd.DataFrame({
    'UMAP1': latent_umap[:, 0],
    'UMAP2': latent_umap[:, 1],
    'Cluster_Label': cluster_labels
})

latent_df.to_csv("output/cluster8_latent_clusters.csv", index=False)

print("Cluster labels saved to output/cluster8_latent_clusters.csv")

print("len(barcodes_train):", len(barcodes_train))
print("len(cluster_labels):", len(cluster_labels))
print("latent_umap shape:", latent_umap.shape)

linked_df = pd.DataFrame({
    'Barcode': barcodes_train,  # Barcodes match the order of latent points
    'Cluster_Label': cluster_labels,
    'UMAP1': latent_umap[:, 0],
    'UMAP2': latent_umap[:, 1]
})

linked_df.to_csv("output/cluster8_cells_with_clusters.csv", index=False)

print("Linked cluster labels saved to output/cluster8_cells_with_clusters.csv")

linked_df = pd.read_csv("output/cluster8_cells_with_clusters.csv")

# Find barcodes for each cluster
cluster0_barcodes = linked_df.loc[linked_df['Cluster_Label'] == 0, 'Barcode']
cluster1_barcodes = linked_df.loc[linked_df['Cluster_Label'] == 1, 'Barcode']

print(f"Cluster 0 cells: {len(cluster0_barcodes)}")
print(f"Cluster 1 cells: {len(cluster1_barcodes)}")

# rna_cluster8 has the full RNA expression matrix
# Subset for each cluster
rna_cluster0 = rna_cluster8.loc[cluster0_barcodes]
rna_cluster1 = rna_cluster8.loc[cluster1_barcodes]

print(f"Cluster 0 RNA shape: {rna_cluster0.shape}")
print(f"Cluster 1 RNA shape: {rna_cluster1.shape}")


# Calculating mean gene expression per cluster
mean_expr_cluster0 = rna_cluster0.mean(axis=0)
mean_expr_cluster1 = rna_cluster1.mean(axis=0)

# Calculating log2 fold change
log2fc = np.log2((mean_expr_cluster1 + 1e-6) / (mean_expr_cluster0 + 1e-6))  # Adding small value to avoid division by zero

gene_fc_df = pd.DataFrame({
    'Gene': mean_expr_cluster0.index,
    'Mean_Cluster0': mean_expr_cluster0.values,
    'Mean_Cluster1': mean_expr_cluster1.values,
    'Log2FC_Cluster1_vs_Cluster0': log2fc
})

# Sorting by absolute log2 fold change
gene_fc_df['abs_Log2FC'] = np.abs(gene_fc_df['Log2FC_Cluster1_vs_Cluster0'])
gene_fc_df_sorted = gene_fc_df.sort_values(by='abs_Log2FC', ascending=False)

gene_fc_df_sorted.to_csv("output/cluster8_gene_log2fc.csv", index=False)

print("Top genes differentiating clusters:")
print(gene_fc_df_sorted.head(10))

# Picking top 10 marker genes
top10_genes = gene_fc_df_sorted.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top10_genes['Gene'], top10_genes['Log2FC_Cluster1_vs_Cluster0'], color='skyblue')
plt.axvline(0, color='black')
plt.xlabel('Log2 Fold Change (Cluster1 vs Cluster0)')
plt.title('Top 10 Marker Genes')
plt.tight_layout()
plt.savefig("output/top10_marker_genes.png")
plt.show()

class RNA_VAE(Model):
    def __init__(self, encoder, decoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()
        self.latent_dim = latent_dim

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def train_step(self, x):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(x)
            z = self.sampling((z_mean, z_log_var))
            x_recon = self.decoder(z)
            recon_loss = tf.reduce_mean(tf.square(x - x_recon))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = recon_loss + kl_loss
            tf.print("total loss:", total_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

enc_rna_only = build_encoder(rna_matrix.shape[1], LATENT_DIM)
dec_rna_only = build_decoder(LATENT_DIM, rna_matrix.shape[1])
vae_rna = RNA_VAE(enc_rna_only, dec_rna_only, LATENT_DIM)
vae_rna.compile(optimizer=tf.keras.optimizers.Adam())
vae_rna.fit(rna_matrix, epochs=50, batch_size=32)

z_mean_rna_only, _ = vae_rna.encoder(rna_matrix)
latent_umap_rna = umap.UMAP(random_state=42).fit_transform(z_mean_rna_only.numpy())

plt.figure(figsize=(6, 5))
plt.scatter(latent_umap_rna[:, 0], latent_umap_rna[:, 1], s=10, c='blue')
plt.title("VAE (RNA only) - Latent UMAP")
plt.tight_layout()
plt.savefig("output/vae_rna_only_umap.png")
plt.show()

rna_cluster8_df = pd.DataFrame(rna_matrix, index=common_barcodes, columns=rna_cluster8.columns)


detailed_linked_df = linked_df.merge(
    rna_cluster8_df, left_on="Barcode", right_index=True
)


detailed_linked_df.to_csv("output/cluster8_cells_with_genes.csv", index=False)

print("Saved full cluster-gene expression data to 'output/cluster8_cells_with_genes.csv'")

rna_with_clusters = rna_cluster8.copy()
rna_with_clusters['Cluster_Label'] = rna_with_clusters.index.map(
    dict(zip(linked_df['Barcode'], linked_df['Cluster_Label']))
)
rna_with_clusters = rna_with_clusters.dropna(subset=["Cluster_Label"])

rna_with_clusters = rna_cluster8.copy()
rna_with_clusters['Cluster_Label'] = rna_with_clusters.index.map(
    dict(zip(linked_df['Barcode'], linked_df['Cluster_Label']))
)

top_genes = {}
for c in sorted(rna_with_clusters['Cluster_Label'].unique()):
    cluster_data = rna_with_clusters[rna_with_clusters['Cluster_Label'] == c].drop(columns='Cluster_Label')
    mean_expr = cluster_data.mean().sort_values(ascending=False).head(10)
    top_genes[c] = mean_expr
    print(f"\n🔹 Top genes in Cluster {c}:\n{mean_expr}")

def run_de_analysis(cluster1_data, cluster2_data, cluster1_name, cluster2_name):
    """Run differential expression analysis between two clusters"""
    p_vals = []
    log2fc = []
    genes = []
    mean1 = []
    mean2 = []

    for gene in cluster1_data.columns:
        # Skip non-gene columns
        if gene == 'Cluster_Label':
            continue

        # Calculate statistics
        mean_expr1 = cluster1_data[gene].mean()
        mean_expr2 = cluster2_data[gene].mean()

        # Only test genes with some expression
        if mean_expr1 > 0 or mean_expr2 > 0:
            # Adding pseudocount to handle zeros
            fc = (mean_expr2 + 1e-6) / (mean_expr1 + 1e-6)
            log2_fc = np.log2(fc)

            # Statistical test
            t_stat, p = ttest_ind(
                cluster2_data[gene].values,
                cluster1_data[gene].values,
                equal_var=False
            )

            # Store results
            genes.append(gene)
            p_vals.append(p)
            log2fc.append(log2_fc)
            mean1.append(mean_expr1)
            mean2.append(mean_expr2)

    fdr = multipletests(p_vals, method='fdr_bh')[1]

    result = pd.DataFrame({
        'Gene': genes,
        f'Mean_{cluster1_name}': mean1,
        f'Mean_{cluster2_name}': mean2,
        f'Log2FC_{cluster2_name}_vs_{cluster1_name}': log2fc,
        'p_value': p_vals,
        'FDR': fdr
    })

    return result


result = run_de_analysis(
    rna_cluster0.drop(columns=['Cluster_Label'], errors='ignore'),
    rna_cluster1.drop(columns=['Cluster_Label'], errors='ignore'),
    'Cluster0', 'Cluster1'
)

# Filter significant DEGs with better handling of missing values
sig = result[(result['FDR'] < 0.05) & (np.abs(result[f'Log2FC_Cluster1_vs_Cluster0']) > 0.5)]
sig_sorted = sig.sort_values(by='FDR')

def analyze_tcr_characteristics(tcr_df, cluster_barcodes, cluster_name):
    """Analyze TCR characteristics for a given cluster"""

    # Filter TCR data for the cluster
    cluster_tcr = tcr_df[tcr_df.index.isin(cluster_barcodes)]

    # V gene usage
    v_gene_counts = cluster_tcr['v_gene'].value_counts()
    v_gene_freq = v_gene_counts / len(cluster_tcr) * 100

    # J gene usage
    j_gene_counts = cluster_tcr['j_gene'].value_counts()
    j_gene_freq = j_gene_counts / len(cluster_tcr) * 100

    # CDR3 length distribution
    cdr3_lengths = cluster_tcr['cdr3'].str.len()
    cdr3_length_stats = {
        'mean': cdr3_lengths.mean(),
        'median': cdr3_lengths.median(),
        'min': cdr3_lengths.min(),
        'max': cdr3_lengths.max()
    }

    return {
        'v_gene_freq': v_gene_freq,
        'j_gene_freq': j_gene_freq,
        'cdr3_length_stats': cdr3_length_stats,
        'sample_size': len(cluster_tcr)
    }


# Analyze TCR characteristics for each cluster
tcr_cluster0 = analyze_tcr_characteristics(tcr_df, cluster0_barcodes, 'Cluster0')
tcr_cluster1 = analyze_tcr_characteristics(tcr_df, cluster1_barcodes, 'Cluster1')

# Visualize V gene usage differences
top_v_genes = set(tcr_cluster0['v_gene_freq'].nlargest(5).index).union(
    set(tcr_cluster1['v_gene_freq'].nlargest(5).index)
)

v_gene_comparison = pd.DataFrame({
    'V_Gene': list(top_v_genes),
    'Cluster0_Freq': [tcr_cluster0['v_gene_freq'].get(gene, 0) for gene in top_v_genes],
    'Cluster1_Freq': [tcr_cluster1['v_gene_freq'].get(gene, 0) for gene in top_v_genes]
})

# Plot V gene usage comparison
plt.figure(figsize=(10, 6))
bar_width = 0.35
x = np.arange(len(v_gene_comparison))
plt.bar(x - bar_width / 2, v_gene_comparison['Cluster0_Freq'], bar_width, label='Cluster 0')
plt.bar(x + bar_width / 2, v_gene_comparison['Cluster1_Freq'], bar_width, label='Cluster 1')
plt.xlabel('V Gene')
plt.ylabel('Frequency (%)')
plt.title('V Gene Usage Between Clusters')
plt.xticks(x, v_gene_comparison['V_Gene'], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("output/v_gene_usage_comparison.png")
plt.show()

def analyze_latent_dimensions(latent_data, cluster_labels):
    """Analyze which latent dimensions contribute most to cluster separation"""

    # Create DataFrame with latent dimensions and cluster labels
    latent_df = pd.DataFrame(
        latent_data,
        columns=[f'Dim_{i}' for i in range(latent_data.shape[1])]
    )
    latent_df['Cluster'] = cluster_labels

    # Calculate significance of each dimension
    p_values = []
    effect_sizes = []

    for dim in range(latent_data.shape[1]):
        dim_name = f'Dim_{dim}'
        dim_values_0 = latent_df[latent_df['Cluster'] == 0][dim_name]
        dim_values_1 = latent_df[latent_df['Cluster'] == 1][dim_name]

        # Calculate t-test
        t_stat, p_val = ttest_ind(dim_values_0, dim_values_1, equal_var=False)

        # Calculate effect size (Cohen's d)
        mean_diff = dim_values_1.mean() - dim_values_0.mean()
        pooled_std = np.sqrt(
            ((len(dim_values_0) - 1) * dim_values_0.std() ** 2 +
             (len(dim_values_1) - 1) * dim_values_1.std() ** 2) /
            (len(dim_values_0) + len(dim_values_1) - 2)
        )
        effect_size = mean_diff / pooled_std

        p_values.append(p_val)
        effect_sizes.append(effect_size)

    dim_importance = pd.DataFrame({
        'Dimension': [f'Dim_{i}' for i in range(latent_data.shape[1])],
        'P_Value': p_values,
        'Effect_Size': effect_sizes
    })

    dim_importance = dim_importance.sort_values(
        by='Effect_Size',
        key=abs,
        ascending=False
    )

    return dim_importance

latent_dimensions = analyze_latent_dimensions(latent_train.numpy(), cluster_labels)
print("Top latent dimensions:")
print(latent_dimensions.head(10))

latent_dimensions.to_csv("output/latent_dimension_importance.csv", index=False)

top_dims = latent_dimensions.head(2)['Dimension'].values
dim_indices = [int(dim.split('_')[1]) for dim in top_dims]

plt.figure(figsize=(8, 6))
plt.scatter(
    latent_train.numpy()[:, dim_indices[0]],
    latent_train.numpy()[:, dim_indices[1]],
    c=cluster_labels,
    cmap='coolwarm',
    s=20,
    alpha=0.7
)
plt.title(f"Top Discriminative Latent Dimensions: {top_dims[0]} vs {top_dims[1]}")
plt.xlabel(top_dims[0])
plt.ylabel(top_dims[1])
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.savefig("output/top_latent_dimensions.png")
plt.show()

def evaluate_cross_modal_prediction(model, rna_test, tcr_test):
    """Evaluate how well one modality can predict the other"""

    # RNA to TCR prediction
    z_mean_rna, z_log_var_rna = model.encoder_rna(rna_test)
    z_rna = model.sampling((z_mean_rna, z_log_var_rna))
    tcr_pred_from_rna = model.decoder_tcr(z_rna)

    # TCR to RNA prediction
    z_mean_tcr, z_log_var_tcr = model.encoder_tcr(tcr_test)
    z_tcr = model.sampling((z_mean_tcr, z_log_var_tcr))
    rna_pred_from_tcr = model.decoder_rna(z_tcr)

    tcr_test_binary = tf.cast(tcr_test > 0, tf.float32)
    tcr_pred_binary = tf.cast(tcr_pred_from_rna > 0.5, tf.float32)

    tcr_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tcr_test_binary, tcr_pred_binary), tf.float32)
    ).numpy()

    rna_mse = tf.reduce_mean(tf.square(rna_test - rna_pred_from_tcr)).numpy()

    return {
        'tcr_from_rna_accuracy': tcr_accuracy,
        'rna_from_tcr_mse': rna_mse,
        'tcr_predictions': tcr_pred_from_rna.numpy(),
        'rna_predictions': rna_pred_from_tcr.numpy()
    }


# Evaluate cross-modal prediction
cross_modal_results = evaluate_cross_modal_prediction(model, rna_test, tcr_test)
print(f"TCR prediction accuracy from RNA: {cross_modal_results['tcr_from_rna_accuracy']:.4f}")
print(f"RNA prediction MSE from TCR: {cross_modal_results['rna_from_tcr_mse']:.4f}")



def analyze_cdr3_length_distribution(tcr_df, cluster_barcodes_dict, output_dir="output"):
    """
    Analyze CDR3 length distributions between clusters
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from scipy import stats

    all_lengths = []

    for cluster_name, barcodes in cluster_barcodes_dict.items():
        # Filter TCR data for this cluster
        cluster_tcr = tcr_df[tcr_df.index.isin(barcodes)]

        # Calculate CDR3 lengths (skip empty strings)
        cdr3_lengths = cluster_tcr['cdr3'].str.len()
        cdr3_lengths = cdr3_lengths[cdr3_lengths > 0]

        cluster_lengths = pd.DataFrame({
            'Cluster': cluster_name,
            'CDR3_Length': cdr3_lengths.values
        })
        all_lengths.append(cluster_lengths)

    # Combine all data
    length_df = pd.concat(all_lengths, ignore_index=True)

    # Calculating statistics
    stats_df = length_df.groupby('Cluster')['CDR3_Length'].agg([
        'count', 'mean', 'std', 'min', 'max',
        lambda x: np.percentile(x, 25),
        lambda x: np.percentile(x, 50),
        lambda x: np.percentile(x, 75)
    ]).rename(columns={
        '<lambda_0>': 'percentile_25',
        '<lambda_1>': 'percentile_50',
        '<lambda_2>': 'percentile_75'
    }).reset_index()

    stats_df.to_csv(f"{output_dir}/cdr3_length_statistics.csv", index=False)
    print("CDR3 Length Statistics:")
    print(stats_df)

    # Statistical test for difference in length distributions
    clusters = list(cluster_barcodes_dict.keys())
    if len(clusters) >= 2:
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                lengths_1 = length_df[length_df['Cluster'] == clusters[i]]['CDR3_Length']
                lengths_2 = length_df[length_df['Cluster'] == clusters[j]]['CDR3_Length']

                # Mann-Whitney U test (non-parametric)
                u_stat, p_value = stats.mannwhitneyu(lengths_1, lengths_2)
                print(f"\nMann-Whitney U test for {clusters[i]} vs {clusters[j]}:")
                print(f"U statistic: {u_stat}, p-value: {p_value}")

                if p_value < 0.05:
                    print(f"CDR3 length distributions are significantly different (p={p_value:.4f})")
                else:
                    print(f"No significant difference in CDR3 length distributions (p={p_value:.4f})")

    # Create visualizations

    # 1. Histogram of CDR3 lengths
    plt.figure(figsize=(10, 6))
    sns.histplot(data=length_df, x='CDR3_Length', hue='Cluster',
                 element="step", stat="density", common_norm=False)
    plt.title('CDR3 Length Distribution by Cluster')
    plt.xlabel('CDR3 Length (amino acids)')
    plt.ylabel('Density')
    plt.savefig(f"{output_dir}/cdr3_length_histogram.png", dpi=300)
    plt.close()

    # 2. Box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=length_df, x='Cluster', y='CDR3_Length')
    plt.title('CDR3 Length Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('CDR3 Length (amino acids)')
    plt.savefig(f"{output_dir}/cdr3_length_boxplot.png", dpi=300)
    plt.close()

    # 3. Violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=length_df, x='Cluster', y='CDR3_Length')
    plt.title('CDR3 Length Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('CDR3 Length (amino acids)')
    plt.savefig(f"{output_dir}/cdr3_length_violinplot.png", dpi=300)
    plt.close()

    # 4. KDE plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=length_df, x='CDR3_Length', hue='Cluster', fill=True, alpha=0.5)
    plt.title('CDR3 Length Density by Cluster')
    plt.xlabel('CDR3 Length (amino acids)')
    plt.ylabel('Density')
    plt.savefig(f"{output_dir}/cdr3_length_kde.png", dpi=300)
    plt.close()

    return {
        'length_data': length_df,
        'statistics': stats_df
    }

cluster_barcodes_dict = {
    'Cluster0': cluster0_barcodes,
    'Cluster1': cluster1_barcodes
}

cdr3_length_results = analyze_cdr3_length_distribution(tcr_df, cluster_barcodes_dict)