# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.simsiam import simsiam_loss_func
from solo.methods.base import BaseMethod
from solo.utils.metrics import corrcoef, pearsonr_cor
from solo.utils.metrics import accuracy_at_k, weighted_mean


class SimSiam(BaseMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        **kwargs,
    ):
        """Implements SimSiam (https://arxiv.org/abs/2011.10566).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(**kwargs)

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim, affine=False),
        )
        self.projector[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimSiam, SimSiam).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simsiam")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params: List[dict] = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters(), "static_lr": True},
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector and the predictor.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected and predicted features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        out.update({"z": z, "p": p})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimSiam reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimSiam loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]
        p1, p2 = out["p"]

        # ------- contrastive loss -------
        neg_cos_sim = simsiam_loss_func(p1, z2) / 2 + simsiam_loss_func(p2, z1) / 2

        # calculate std of features
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        ## new metric Note: SimSiam has BN in projector output, hence corr is not correct, may design again if serious
        feats1, feats2 = out["feats"]

        with torch.no_grad():
            corr_z = (torch.abs(corrcoef(z1, z2).triu(1)) + torch.abs(corrcoef(z1, z2).tril(-1))).mean()
            pear_z = pearsonr_cor(z1, z2).mean()
            corr_feats = (torch.abs(corrcoef(feats1, feats2).triu(1)) + torch.abs(corrcoef(feats1, feats2).tril(-1)) ).mean()
            pear_feats = pearsonr_cor(feats1, feats2).mean()

        ### new metrics
        metrics = {
            "Logits/avg_sum_logits_P": (torch.stack((p1,p2))).sum(-1).mean(),
            "Logits/avg_sum_logits_P_normalized": F.normalize(torch.stack((p1,p2)), dim=-1).sum(-1).mean(),
            "Logits/avg_sum_logits_Z": (torch.stack((z1,z2))).sum(-1).mean(),
            "Logits/avg_sum_logits_Z_normalized": F.normalize(torch.stack((z1,z2)), dim=-1).sum(-1).mean(),
            
            "Logits/logits_P_max": (torch.stack((p1,p2))).max(),
            "Logits/logits_P_min": (torch.stack((p1,p2))).min(),
            "Logits/logits_Z_max": (torch.stack((z1,z2))).max(),
            "Logits/logits_Z_min": (torch.stack((z1,z2))).min(),

            "Logits/logits_P_normalized_max": F.normalize(torch.stack((p1,p2)), dim=-1).max(),
            "Logits/logits_P_normalized_min": F.normalize(torch.stack((p1,p2)), dim=-1).min(),
            "Logits/logits_Z_normalized_max": F.normalize(torch.stack((z1,z2)), dim=-1).max(),
            "Logits/logits_Z_normalized_min": F.normalize(torch.stack((z1,z2)), dim=-1).min(),

            "MeanVector/mean_vector_P_max": (torch.stack((p1,p2))).mean(1).max(),
            "MeanVector/mean_vector_P_min": (torch.stack((p1,p2))).mean(1).min(),
            "MeanVector/mean_vector_P_normalized_max": F.normalize(torch.stack((p1,p2)), dim=-1).mean(1).max(),
            "MeanVector/mean_vector_P_normalized_min": F.normalize(torch.stack((p1,p2)), dim=-1).mean(1).min(),

            "MeanVector/mean_vector_Z_max": (torch.stack((z1,z2))).mean(1).max(),
            "MeanVector/mean_vector_Z_min": (torch.stack((z1,z2))).mean(1).min(),
            "MeanVector/mean_vector_Z_normalized_max": F.normalize(torch.stack((z1,z2)), dim=-1).mean(1).max(),
            "MeanVector/mean_vector_Z_normalized_min": F.normalize(torch.stack((z1,z2)), dim=-1).mean(1).min(),

            "MeanVector/norm_vector_P": (torch.stack((p1,p2))).mean(1).mean(0).norm(),
            "MeanVector/norm_vector_P_normalized": F.normalize(torch.stack((p1,p2)), dim=-1).mean(1).mean(0).norm(),
            "MeanVector/norm_vector_Z": (torch.stack((z1,z2))).mean(1).mean(0).norm(),
            "MeanVector/norm_vector_Z_normalized": F.normalize(torch.stack((z1,z2)), dim=-1).mean(1).mean(0).norm(),

            "Logits/var_P": (torch.stack((p1,p2))).var(-1).mean(),
            "Logits/var_Z": (torch.stack((z1,z2))).var(-1).mean(),

            "Backbone/var": (torch.stack((feats1, feats2))).var(-1).mean(),
            "Backbone/max": (torch.stack((feats1, feats2))).max(),

            "Corr/corr_z": corr_z,
            "Corr/pear_z": pear_z,
            "Corr/corr_feats": corr_feats,
            "Corr/pear_feats": pear_feats,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        ### new metrics

        return neg_cos_sim + class_loss
