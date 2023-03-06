from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from networkx import from_numpy_array, pagerank
from torch import Tensor
from torch.nn import KLDivLoss, Softmax
from torch_geometric.data import Batch, Data
from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm.base import ExplainerAlgorithm
from torch_geometric.explain.config import (ExplainerConfig, MaskType,
                                            ModelConfig, ModelMode,
                                            ModelTaskLevel)
from torch_geometric.loader import DataLoader
from torch_geometric.utils import index_to_mask, k_hop_subgraph, subgraph

from eixgnn.shapley import mc_shapley


class EiXGNN(ExplainerAlgorithm):
    r"""
    The official EiX-GNN model from the `"EiX-GNN: Concept-level eigencentrality explainer for graph neural
    networks"<https://arxiv.org/abs/2206.03491>`_ paper for identifying useful pattern from a GNN model adapted to user background.

    The following configurations are currently supported:

    - :class:`torch_geometric.explain.config.ModelConfig`
        - :attr:`task_level`: :obj:`"graph"`

    - :class:`torch_geometric.explain.config.ExplainerConfig`

        - :attr:`node_mask_type`: :obj:`"object"`, :obj:`"common_attributes"` or :obj:`"attributes"`
        - :attr:`edge_mask_type`: :obj:`"object"` or :obj:`None`


    Args:
        L (int): The number of concept, it needs to be a positive integer.
            (default: :obj:`60`)
        p (float): The parameter in [0,1] representing the concept assimibility constraint.
            (default: :obj:`0.1`)
        **kwargs (optional): Additional features such as the version of the algorithm or other [TODO: BETTER DESCRIPTION]
    """

    def __init__(
        self,
        L: int = 30,
        p: float = 0.2,
        importance_sampling_strategy: str = "node",
        domain_similarity: str = "relative_edge_density",
        signal_similarity: str = "KL",
        shap_val_approx: int = 100,
        **kwargs,
    ):
        super().__init__()
        self.L = L
        self.p = p
        self.domain_similarity = domain_similarity
        self.signal_similarity = signal_similarity
        self.shap_val_approx = shap_val_approx
        self.importance_sampling_strategy = importance_sampling_strategy
        self.name = "EIXGNN"

    def _domain_similarity(self, graph: Data) -> float:
        if self.domain_similarity == "relative_edge_density":
            if graph.num_edges != 0:
                return graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1))
            else:
                return 1 / (graph.num_nodes * (graph.num_nodes - 1))
        else:
            raise ValueError(f"{self.domain_metric} is not supported yet")

    def _signal_similarity(self, graph1: Tensor, graph2: Tensor) -> float:
        if self.signal_similarity == "KL":
            kldiv = KLDivLoss(reduction="batchmean")
            graph_1 = F.log_softmax(graph1, dim=1)
            graph_2 = F.softmax(graph2, dim=1)
            return kldiv(graph_1, graph_2).item()
        elif self.signal_similarity == "KL_sym":
            kldiv = KLDivLoss(reduction="batchmean")
            graph_11 = F.log_softmax(graph1, dim=1)
            graph_12 = F.log_softmax(graph2, dim=1)
            graph_21 = F.softmax(graph1, dim=1)
            graph_22 = F.softmax(graph2, dim=1)
            return (kldiv(graph_11, graph_22) + kldiv(graph_12, graph_21)).item()

        else:
            raise ValueError(f"{self.domain_metric} is not supported yet")

    def supports(self) -> bool:
        task_level = self.model_config.task_level
        if task_level not in [ModelTaskLevel.graph]:
            logging.error(f"Task level '{task_level.value}' not supported")
            return False

        edge_mask_type = self.explainer_config.edge_mask_type
        if edge_mask_type not in [MaskType.object, None]:
            logging.error(f"Edge mask type '{edge_mask_type.value}' not " f"supported")
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type not in [
            MaskType.common_attributes,
            MaskType.object,
            MaskType.attributes,
        ]:
            logging.error(f"Node mask type '{node_mask_type.value}' not " f"supported.")
            return False

        if not self.importance_sampling_strategy in [
            "node",
            "neighborhood",
            "no_prior",
        ]:
            logging.error(
                f"This node ablation strategy : {node_ablation['strategy']} is not supported yet. No explanation provided."
            )
            return False
        if not self.domain_similarity in ["relative_edge_density"]:
            logging.error(
                f"This domain signal similarity metric : {domain_similarity['metric']} is not supported yet. No explanation provided."
            )
            return False
        if not self.signal_similarity in ["KL", "KL_sym"]:
            logging.error(
                f"This signal similarity metric : {signal_similarity['metric']} is not supported yet. No explanation provided."
            )
            return False
        # TODO ADD OTHER CASE
        return True

    def get_mc_shapley(self, subset_node: list, data: Data) -> Tensor:
        shap = mc_shapley(
            coalition=subset_node,
            data=data,
            value_func=self.model,
            sample_num=self.shap_val_approx,
        )
        return shap

    def get_mc_shapley_concept(self, concept: Data) -> Tensor:
        shap_val = []
        for ind in range(concept.num_nodes):
            coalition = torch.LongTensor([ind]).to(concept.x.device)
            shap_val.append(self.get_mc_shapley(subset_node=coalition, data=concept))
        return torch.FloatTensor(shap_val)

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        target,
        **kwargs,
    ) -> Explanation:
        if int(x.shape[0] * self.p) <= 1:
            raise ValueError(
                f"Provided graph with {x.shape[0]} and parameter p={self.p} produce concept of size {int(x.shape[0]*self.p)}, which is not suitable. Aborting"
            )

        self.model = model
        input_graph = Data(x=x, edge_index=edge_index).to(x.device)

        node_prior_distribution = self._compute_node_ablation_prior(x, edge_index)
        node_prior_distribution = F.softmax(node_prior_distribution, dim=0)
        node_prior_distribution = node_prior_distribution.detach().cpu().numpy()

        concept_nodes_index = np.random.choice(
            np.arange(x.shape[0]),
            size=(self.L, int(self.p * x.shape[0])),
            p=node_prior_distribution,
        )
        indexes = [
            torch.LongTensor(concept_nodes).to(x.device)
            for concept_nodes in concept_nodes_index
        ]
        concepts = [input_graph.subgraph(ind) for ind in indexes]

        A = self._global_concept_similarity_matrix(concepts)
        pr = self._adjacency_pr(A)
        shap_val = [self.get_mc_shapley_concept(concept) for concept in concepts]
        shap_val_ext = self.extend(
            shap_val, indexes=concept_nodes_index, size=(self.L, x.shape[0])
        )

        explanation_map = torch.sum(
            torch.FloatTensor(np.diag(pr) @ shap_val_ext), dim=0
        ).to(x.device)

        edge_mask = None
        node_feat_mask = None
        edge_feat_mask = None

        exp = Explanation(
            x=x,
            edge_index=edge_index,
            y=target,
            node_mask=explanation_map,
            edge_mask=edge_mask,
            node_feat_mask=node_feat_mask,
            edge_feat_mask=edge_feat_mask,
            shap=torch.FloatTensor(shap_val_ext).to(x.device),
            indexes=torch.LongTensor(concept_nodes_index).to(x.device),
            pr=torch.FloatTensor(pr).to(x.device),
        )

        return exp

    def extend(self, shap_vals: list, indexes: list, size: tuple):
        extended_map = np.zeros(size)
        for i in range(indexes.shape[0]):
            for j in range(indexes.shape[1]):
                extended_map[i, indexes[i][j]] += abs(shap_vals[i][j])
        return extended_map

    def _global_concept_similarity_matrix(self, concepts):
        A = np.zeros((len(concepts), len(concepts)))
        concepts_pred = self.model(Batch().from_data_list(concepts))
        concepts_prob = F.softmax(concepts_pred, dim=1)
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                if j >= i:
                    continue
                else:
                    dm1 = self._domain_similarity(c1)
                    dm2 = self._domain_similarity(c2)
                    ss = self._signal_similarity(
                        concepts_prob[i].unsqueeze(0), concepts_prob[j].unsqueeze(0)
                    )
                    A[i, j] = (dm1 / dm2) * ss
                    A[j, i] = (dm2 / dm1) * ss
        return A

    def _adjacency_pr(self, A):
        G = from_numpy_array(A)
        pr = list(pagerank(G).values())
        return pr

    def _compute_node_ablation_prior(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        if self.importance_sampling_strategy == "no_prior":
            node_importance = torch.ones(x.shape[0]) / x.shape[0]
            return node_importance.to(x.device)

        pred = self.model(x=x, edge_index=edge_index)
        node_importance = torch.zeros(x.shape[0])
        for node_index in range(x.shape[0]):
            if self.importance_sampling_strategy == "node":
                mask = index_to_mask(torch.LongTensor([node_index]), size=x.shape[0])
            if self.importance_sampling_strategy == "neighborhood":
                neighborhood_index, _, _, _ = k_hop_subgraph(
                    node_idx=node_index, num_hops=1, edge_index=edge_index
                )
                mask = index_to_mask(neighborhood_index, size=x.shape[0])
            mask = mask <= 0
            node_mask = torch.arange(x.shape[0]).to(x.device)
            node_mask = node_mask[mask]
            edge_index_sub, _ = subgraph(node_mask, edge_index)
            sub_pred = self.model(x=x, edge_index=edge_index_sub)
            node_importance[node_index] = torch.norm(pred - sub_pred, p=1)
        return node_importance.to(x.device)
