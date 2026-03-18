import torch
import torch.nn as nn
import torch.nn.functional as F
 
class TransE(nn.Module):
    def __init__(self, n_entities, n_relations, dim=50, margin=1.0):
        super().__init__()
        self.ent_emb = nn.Embedding(n_entities, dim)
        self.rel_emb = nn.Embedding(n_relations, dim)
        self.margin = margin
        nn.init.uniform_(self.ent_emb.weight, -6/dim**0.5, 6/dim**0.5)
        nn.init.uniform_(self.rel_emb.weight, -6/dim**0.5, 6/dim**0.5)
 
    def score(self, h, r, t):
        h_e = F.normalize(self.ent_emb(h), p=2, dim=-1)
        r_e = self.rel_emb(r)
        t_e = F.normalize(self.ent_emb(t), p=2, dim=-1)
        return -torch.norm(h_e + r_e - t_e, p=2, dim=-1)
 
    def margin_loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos = self.score(pos_h, pos_r, pos_t)
        neg = self.score(neg_h, neg_r, neg_t)
        return F.relu(self.margin - pos + neg).mean()
      def link_predict(self, h, r, top_k=5, n_entities=None):
        n = n_entities or self.ent_emb.num_embeddings
        h_e = F.normalize(self.ent_emb(torch.tensor([h])), p=2, dim=-1)
        r_e = self.rel_emb(torch.tensor([r]))
        all_t = F.normalize(self.ent_emb.weight, p=2, dim=-1)
        scores = -torch.norm(h_e + r_e - all_t, p=2, dim=-1)
        return scores.topk(top_k).indices.tolist()
 
# FB15k-style mini dataset
n_ent, n_rel = 100, 20
triples = [(torch.randint(0,n_ent,(32,)), torch.randint(0,n_rel,(32,)), torch.randint(0,n_ent,(32,))) for _ in range(5)]
model = TransE(n_ent, n_rel, dim=50)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
for epoch in range(50):
    total_loss = 0
    for pos_h, pos_r, pos_t in triples:
        neg_t = torch.randint(0, n_ent, pos_t.shape)
        loss = model.margin_loss(pos_h, pos_r, pos_t, pos_h, pos_r, neg_t)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0: print(f"Epoch {epoch}: Loss={total_loss/len(triples):.4f}")
 
preds = model.link_predict(0, 1, top_k=5, n_entities=n_ent)
print(f"Link prediction for (e0, r1, ?): {preds}")
