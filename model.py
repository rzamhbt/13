import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Semantic Learner ----------------
class SemanticLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SemanticLearner, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# ---------------- Gated Fusion ----------------
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, z1, z2):
        gate = self.gate(torch.cat([z1, z2], dim=1))
        return gate * z1 + (1 - gate) * z2

# ---------------- Clustering Learner ----------------
class ClusteringLearner(nn.Module):
    def __init__(self, input_dim, num_clusters):
        super(ClusteringLearner, self).__init__()
        self.to_cluster = nn.Linear(input_dim, num_clusters)

    def forward(self, x):
        return self.to_cluster(x)

# ---------------- Full Model ----------------
class MvDC_HCL(nn.Module):
    def __init__(self, input_dim, hidden_dim, fusion_dim, num_clusters):
        super(MvDC_HCL, self).__init__()
        self.semantic1 = SemanticLearner(input_dim, hidden_dim)
        self.semantic2 = SemanticLearner(input_dim, hidden_dim)
        self.fusion = GatedFusion(hidden_dim)
        self.projector = nn.Linear(hidden_dim, fusion_dim)
        self.cluster = ClusteringLearner(fusion_dim, num_clusters)

    def forward(self, x1, x2):
        z1 = self.semantic1(x1)
        z2 = self.semantic2(x2)
        fused = self.fusion(z1, z2)
        projected = self.projector(fused)
        clusters = self.cluster(projected)
        return clusters, projected, z1, z2

# ---------------- Contrastive Loss ----------------
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(logits, labels)
