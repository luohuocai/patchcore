"""FastRef-style prototype refinement for PatchCore.

The official FastRef repository is currently empty, so this module implements
the PatchCore+ refinement described in the paper: query features refine the
normal memory prototypes through alternating Sinkhorn transport and a closed
form reconstruction update.
"""

import numpy as np
import torch


class FastRefiner:
    def __init__(
        self,
        device,
        balance=1.0,
        iterations=2,
        sinkhorn_iterations=10,
        epsilon=0.05,
        ridge=1e-5,
        chunk_size=1024,
    ):
        self.device = device
        self.balance = float(balance)
        self.iterations = int(iterations)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.epsilon = float(epsilon)
        self.ridge = float(ridge)
        self.chunk_size = int(chunk_size)
        self.prototype_features = None
        self._prototype_tensor = None
        self._projection_tensor = None

        if self.balance < 0:
            raise ValueError("FastRef balance must be >= 0.")
        if self.iterations < 0:
            raise ValueError("FastRef iterations must be >= 0.")
        if self.sinkhorn_iterations <= 0:
            raise ValueError("FastRef sinkhorn_iterations must be > 0.")
        if self.epsilon <= 0:
            raise ValueError("FastRef epsilon must be > 0.")
        if self.ridge < 0:
            raise ValueError("FastRef ridge must be >= 0.")
        if self.chunk_size <= 0:
            raise ValueError("FastRef chunk_size must be > 0.")

    def fit(self, prototype_features):
        prototype_features = np.asarray(prototype_features, dtype=np.float32)
        if prototype_features.ndim != 2:
            raise ValueError("FastRef prototypes must be a 2D feature matrix.")
        if prototype_features.shape[0] == 0:
            raise ValueError("FastRef requires at least one prototype feature.")

        self.prototype_features = prototype_features
        self._prototype_tensor = None
        self._projection_tensor = None

    def _prototypes(self):
        if self.prototype_features is None:
            raise RuntimeError("FastRef has not been fitted with prototypes.")
        if self._prototype_tensor is None:
            self._prototype_tensor = torch.as_tensor(
                self.prototype_features, dtype=torch.float32, device=self.device
            )
        return self._prototype_tensor

    def _projection(self, prototypes):
        if self._projection_tensor is not None:
            return self._projection_tensor

        num_prototypes, feature_dim = prototypes.shape
        if num_prototypes >= feature_dim:
            gram = prototypes.T @ prototypes
            eye = torch.eye(feature_dim, device=prototypes.device, dtype=prototypes.dtype)
            self._projection_tensor = torch.linalg.solve(
                gram + self.ridge * eye,
                gram,
            )
        else:
            gram = prototypes @ prototypes.T
            eye = torch.eye(
                num_prototypes, device=prototypes.device, dtype=prototypes.dtype
            )
            solved = torch.linalg.solve(gram + self.ridge * eye, prototypes)
            self._projection_tensor = prototypes.T @ solved

        return self._projection_tensor

    @staticmethod
    def _pairwise_squared_distances(left, right):
        left_norm = torch.sum(left * left, dim=1, keepdim=True)
        right_norm = torch.sum(right * right, dim=1).reshape(1, -1)
        distances = left_norm + right_norm - 2 * left @ right.T
        return torch.clamp(distances, min=0.0)

    def _sinkhorn_transport(self, cost):
        cost_scale = torch.mean(cost.detach()).clamp_min(1e-12)
        log_kernel = -(cost / cost_scale) / self.epsilon

        num_rows, num_cols = cost.shape
        log_row_mass = -torch.log(
            torch.tensor(float(num_rows), device=cost.device, dtype=cost.dtype)
        )
        log_col_mass = -torch.log(
            torch.tensor(float(num_cols), device=cost.device, dtype=cost.dtype)
        )

        log_u = torch.zeros(num_rows, device=cost.device, dtype=cost.dtype)
        log_v = torch.zeros(num_cols, device=cost.device, dtype=cost.dtype)
        for _ in range(self.sinkhorn_iterations):
            log_u = log_row_mass - torch.logsumexp(
                log_kernel + log_v.reshape(1, -1), dim=1
            )
            log_v = log_col_mass - torch.logsumexp(
                log_kernel + log_u.reshape(-1, 1), dim=0
            )

        return torch.exp(log_kernel + log_u.reshape(-1, 1) + log_v.reshape(1, -1))

    def _minimum_distances(self, query_features, refined_prototypes):
        min_scores = []
        min_indices = []
        for start in range(0, len(query_features), self.chunk_size):
            query_chunk = query_features[start : start + self.chunk_size]
            distances = self._pairwise_squared_distances(
                query_chunk, refined_prototypes
            )
            chunk_scores, chunk_indices = torch.min(distances, dim=1)
            min_scores.append(chunk_scores)
            min_indices.append(chunk_indices)

        return torch.cat(min_scores), torch.cat(min_indices)

    def predict(self, query_features):
        query_features = torch.as_tensor(
            np.asarray(query_features, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        if query_features.ndim != 2:
            raise ValueError("FastRef query features must be a 2D feature matrix.")

        prototypes = self._prototypes()
        projection = self._projection(prototypes)

        refined = query_features @ projection
        for _ in range(self.iterations):
            cost = self._pairwise_squared_distances(refined, prototypes)
            transport = self._sinkhorn_transport(cost)
            row_mass = torch.sum(transport, dim=1, keepdim=True)
            transported_prototypes = transport @ prototypes
            target = (query_features + self.balance * transported_prototypes) / (
                1.0 + self.balance * row_mass
            )
            refined = target @ projection

        scores, indices = self._minimum_distances(query_features, refined)
        return scores.detach().cpu().numpy(), indices.detach().cpu().numpy()
