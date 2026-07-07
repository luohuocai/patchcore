"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.fastref
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device
        self.last_inference_stats = []

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        score_gamma=1.0,
        fastref_enabled=False,
        fastref_lambda=1.0,
        fastref_iterations=2,
        fastref_sinkhorn_iterations=10,
        fastref_epsilon=0.05,
        fastref_ridge=1e-5,
        fastref_chunk_size=1024,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler
        self.score_gamma = float(score_gamma)
        if self.score_gamma <= 0:
            raise ValueError("score_gamma must be > 0.")
        self.fastref_enabled = bool(fastref_enabled)
        self.fastref_params = {
            "fastref_lambda": float(fastref_lambda),
            "fastref_iterations": int(fastref_iterations),
            "fastref_sinkhorn_iterations": int(fastref_sinkhorn_iterations),
            "fastref_epsilon": float(fastref_epsilon),
            "fastref_ridge": float(fastref_ridge),
            "fastref_chunk_size": int(fastref_chunk_size),
        }
        self.fastrefiner = (
            patchcore.fastref.FastRefiner(
                device=self.device,
                balance=self.fastref_params["fastref_lambda"],
                iterations=self.fastref_params["fastref_iterations"],
                sinkhorn_iterations=self.fastref_params[
                    "fastref_sinkhorn_iterations"
                ],
                epsilon=self.fastref_params["fastref_epsilon"],
                ridge=self.fastref_params["fastref_ridge"],
                chunk_size=self.fastref_params["fastref_chunk_size"],
            )
            if self.fastref_enabled
            else None
        )

    def _apply_score_gamma(self, patch_scores):
        if self.score_gamma == 1.0:
            return patch_scores
        patch_scores = np.asarray(patch_scores, dtype=np.float32)
        patch_scores = np.clip(patch_scores, a_min=0.0, a_max=None)
        max_scores = patch_scores.reshape(patch_scores.shape[0], -1).max(axis=1)
        max_scores = np.maximum(max_scores, 1e-8).reshape(-1, 1, 1)
        normalized_scores = np.clip(patch_scores / max_scores, 0.0, 1.0)
        return np.power(normalized_scores, self.score_gamma) * max_scores

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])
        if self.fastrefiner is not None:
            self.fastrefiner.fit(self.anomaly_scorer.detection_features)

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _track_cuda_memory(self):
        return self.device.type == "cuda" and torch.cuda.is_available()

    @staticmethod
    def _format_gpu_memory(num_bytes):
        if num_bytes is None:
            return "N/A"
        return "{:.2f} MB".format(num_bytes / (1024**2))

    @staticmethod
    def _as_batch_list(value, batchsize):
        if value is None:
            return [None] * batchsize
        if isinstance(value, torch.Tensor):
            return value.cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value] * batchsize

    def _predict_single_image_with_stats(self, image):
        track_cuda_memory = self._track_cuda_memory()
        if track_cuda_memory:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()
        scores, masks = self._predict(image)

        if track_cuda_memory:
            torch.cuda.synchronize(self.device)

        elapsed_time = time.perf_counter() - start_time
        peak_gpu_memory = (
            torch.cuda.max_memory_allocated(self.device) if track_cuda_memory else None
        )
        return scores, masks, elapsed_time, peak_gpu_memory

    @staticmethod
    def _print_inference_stats(inference_stats):
        if not inference_stats:
            return

        tqdm.tqdm.write("Per-image inference stats:")
        for stat in inference_stats:
            tqdm.tqdm.write(
                "image={image} | inference_time={time:.6f}s | "
                "peak_gpu_memory={memory}".format(
                    image=stat["image"],
                    time=stat["inference_time"],
                    memory=PatchCore._format_gpu_memory(stat["peak_gpu_memory"]),
                )
            )

        average_inference_time_ms = (
            sum(stat["inference_time"] for stat in inference_stats)
            / len(inference_stats)
            * 1000
        )
        peak_gpu_memories = [
            stat["peak_gpu_memory"]
            for stat in inference_stats
            if stat["peak_gpu_memory"] is not None
        ]
        inference_peak_gpu_memory = (
            max(peak_gpu_memories) if peak_gpu_memories else None
        )
        tqdm.tqdm.write(
            "Inference summary: average_inference_time={time:.3f} ms/per | "
            "inference_peak_gpu_memory={memory}".format(
                time=average_inference_time_ms,
                memory=PatchCore._format_gpu_memory(inference_peak_gpu_memory),
            )
        )

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        self.last_inference_stats = []
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        inference_stats = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for batch in data_iterator:
                image_paths = None
                if isinstance(batch, dict):
                    labels_gt.extend(batch["is_anomaly"].numpy().tolist())
                    masks_gt.extend(batch["mask"].numpy().tolist())
                    image_paths = batch.get("image_path")
                    batch = batch["image"]

                batchsize = batch.shape[0]
                image_paths = self._as_batch_list(image_paths, batchsize)
                for image_index in range(batchsize):
                    image = batch[image_index : image_index + 1]
                    _scores, _masks, inference_time, peak_gpu_memory = (
                        self._predict_single_image_with_stats(image)
                    )
                    image_name = image_paths[image_index]
                    if image_name is None:
                        image_name = "image_{}".format(len(inference_stats))
                    inference_stats.append(
                        {
                            "image": image_name,
                            "inference_time": inference_time,
                            "peak_gpu_memory": peak_gpu_memory,
                        }
                    )
                    score = _scores[0]
                    mask = _masks[0]
                    scores.append(score)
                    masks.append(mask)
        self.last_inference_stats = inference_stats
        self._print_inference_stats(inference_stats)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            if self.fastrefiner is not None:
                patch_count = patch_shapes[0][0] * patch_shapes[0][1]
                features_per_image = features.reshape(batchsize, patch_count, -1)
                fastref_patch_scores = []
                for image_features in features_per_image:
                    image_patch_scores, _ = self.fastrefiner.predict(image_features)
                    fastref_patch_scores.append(image_patch_scores)
                patch_scores = image_scores = np.concatenate(
                    fastref_patch_scores, axis=0
                )
            else:
                patch_scores = image_scores = self.anomaly_scorer.predict([features])[
                    0
                ]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            patch_scores = self._apply_score_gamma(patch_scores)

            masks = self.anomaly_segmentor.convert_to_segmentation(
                patch_scores, target_size=images.shape[-2:]
            )

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path,
            save_features_separately=self.fastref_enabled,
            prepend=prepend,
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
            "score_gamma": self.score_gamma,
            "fastref_enabled": self.fastref_enabled,
            **self.fastref_params,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
        score_gamma: float = None,
        fastref_enabled: bool = None,
        fastref_lambda: float = None,
        fastref_iterations: int = None,
        fastref_sinkhorn_iterations: int = None,
        fastref_epsilon: float = None,
        fastref_ridge: float = None,
        fastref_chunk_size: int = None,
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        if score_gamma is not None:
            patchcore_params["score_gamma"] = score_gamma
        fastref_overrides = {
            "fastref_enabled": fastref_enabled,
            "fastref_lambda": fastref_lambda,
            "fastref_iterations": fastref_iterations,
            "fastref_sinkhorn_iterations": fastref_sinkhorn_iterations,
            "fastref_epsilon": fastref_epsilon,
            "fastref_ridge": fastref_ridge,
            "fastref_chunk_size": fastref_chunk_size,
        }
        for key, value in fastref_overrides.items():
            if value is not None:
                patchcore_params[key] = value
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)
        if self.fastrefiner is not None:
            if not hasattr(self.anomaly_scorer, "detection_features"):
                raise RuntimeError(
                    "FastRef requires saved nnscorer_features.pkl, but this "
                    "PatchCore model was saved without memory-bank features."
                )
            self.fastrefiner.fit(self.anomaly_scorer.detection_features)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
