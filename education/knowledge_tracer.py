"""Deep Knowledge Tracing for adaptive learning."""
import torch
import torch.nn as nn
from typing import List, Tuple

class DKTModel(nn.Module):
    """LSTM-based Deep Knowledge Tracing."""
    def __init__(self, n_concepts: int, embed_dim: int = 100, hidden_dim: int = 200):
        super().__init__()
        self.n_concepts = n_concepts
        # Input: concept_id * 2 (correct/incorrect separate embeddings)
        self.embedding = nn.Embedding(n_concepts * 2, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_concepts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, interactions: torch.Tensor) -> torch.Tensor:
        # interactions: [batch, seq_len, 2] (concept_id, correct)
        concept_ids = interactions[:, :, 0].long()
        correct = interactions[:, :, 1].long()
        # Encode: wrong=concept_id, correct=concept_id+n_concepts
        input_ids = concept_ids + correct * self.n_concepts
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        logits = self.output_layer(lstm_out)
        return self.sigmoid(logits)  # [batch, seq_len, n_concepts]: P(correct) per concept

class KnowledgeTracer:
    def __init__(self, n_concepts: int):
        self.model = DKTModel(n_concepts)
        self.n_concepts = n_concepts

    def predict_mastery(self, interaction_history: List[Tuple[int, bool]]) -> dict:
        """Given interaction history, predict mastery per concept."""
        if not interaction_history: return {i: 0.5 for i in range(self.n_concepts)}
        interactions = torch.tensor([[c, int(r)] for c, r in interaction_history],
                                    dtype=torch.float).unsqueeze(0)
        with torch.no_grad(): probs = self.model(interactions)[0, -1].tolist()
        return {i: round(p, 3) for i, p in enumerate(probs)}

    def recommend_next(self, mastery: dict, target_concept: int) -> List[int]:
        """Recommend prerequisite concepts to study before target."""
        low_mastery = [c for c, m in mastery.items() if m < 0.7]
        return low_mastery[:3]  # Top 3 weakest prerequisites
