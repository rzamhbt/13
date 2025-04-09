import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from model import MvDC_HCL, contrastive_loss
from dataset_loader import load_bbc_data

# ------------------- Load Data -------------------
x1, x2, labels = load_bbc_data("data/bbc", max_features=100)
batch_size = 32
dataset = TensorDataset(x1, x2, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------------- Model Init -------------------
input_dim = x1.shape[1]
model = MvDC_HCL(input_dim=input_dim, hidden_dim=64, fusion_dim=32, num_clusters=5)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------- Training -------------------
for epoch in range(100):
    model.train()
    total_loss = 0
    for x1_batch, x2_batch, label_batch in dataloader:
        optimizer.zero_grad()
        clusters, projected, z1, z2 = model(x1_batch, x2_batch)
        loss_c = contrastive_loss(z1, z2)
        loss_r = F.mse_loss(clusters, F.one_hot(label_batch, num_classes=5).float())
        loss = loss_c + loss_r
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} - Loss: {total_loss / len(dataloader):.4f}")

# ------------------- Evaluation -------------------
model.eval()
with torch.no_grad():
    _, features, _, _ = model(x1, x2)
    pred_labels = KMeans(n_clusters=5).fit_predict(features.numpy())
    nmi = normalized_mutual_info_score(labels.numpy(), pred_labels)
    ari = adjusted_rand_score(labels.numpy(), pred_labels)
    print("\nClustering Results")
    print("NMI:", nmi)
    print("ARI:", ari)
