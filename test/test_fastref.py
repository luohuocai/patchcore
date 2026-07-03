import numpy as np
import torch

from patchcore.fastref import FastRefiner


def test_fastref_predict_returns_one_score_per_query_feature():
    prototypes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    query_features = np.array(
        [
            [0.05, 0.0],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )

    refiner = FastRefiner(
        device=torch.device("cpu"),
        iterations=1,
        sinkhorn_iterations=3,
        chunk_size=1,
    )
    refiner.fit(prototypes)

    scores, indices = refiner.predict(query_features)

    assert scores.shape == (len(query_features),)
    assert indices.shape == (len(query_features),)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)
