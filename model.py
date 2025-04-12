import torch
import torch.nn as nn
import torch.nn.functional as F

# Semantic Learner Network for each view
class SemanticLearner(nn.Module):
    def init(self, input_dim, hidden_dim):
        super(SemanticLearner, self).init()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# Gated fusion module to combine two views
class GatedFusion(nn.Module):
    def init(self, dim):
        super(GatedFusion, self).init()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, z1, z2):
        gate = self.gate(torch.cat([z1, z2], dim=1))
        return gate * z1 + (1 - gate) * z2

# Clustering module (maps fused feature to cluster logits)
class ClusteringLearner(nn.Module):
    def init(self, input_dim, num_clusters):
        super(ClusteringLearner, self).init()
        self.to_cluster = nn.Linear(input_dim, num_clusters)

    def forward(self, x):
        return self.to_cluster(x)

# Full MvDC-HCL model combining all modules
class MvDC_HCL(nn.Module):
    def init(self, input_dim, hidden_dim, fusion_dim, num_clusters):
        super(MvDC_HCL, self).init()
        self.semantic1 = SemanticLearner(input_dim, hidden_dim)
        self.semantic2 = SemanticLearner(input_dim, hidden_dim)
        self.fusion = GatedFusion(hidden_dim)
        self.projector = nn.Linear(hidden_dim, fusion_dim)
        self.cluster = ClusteringLearner(fusion_dim, num_clusters)

    def forward(self, x1, x2):
        z1 = self.semantic1(x1)  # View 1 representation
        z2 = self.semantic2(x2)  # View 2 representation
        fused = self.fusion(z1, z2)  # Fuse both views
        projected = self.projector(fused)  # Project to low-dim space
        clusters = self.cluster(projected)  # Predict cluster logits
        return clusters, projected, z1, z2

# Contrastive loss between two view representations
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(logits, labels)
