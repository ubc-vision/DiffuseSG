import torch
import torch.nn as nn

from utils.graph_utils import mask_adjs, mask_nodes

class NodeAdjRainbowLoss(nn.Module):
    def __init__(self, edge_loss_weight, node_loss_weight, objective, flag_reweight=False,):
        """
        Rainbow loss with multiple ingredients.
        - Reweight regularization
        - Graph matching loss (debug)
        """

        super(NodeAdjRainbowLoss, self).__init__()

        self.edge_loss_weight = edge_loss_weight
        self.node_loss_weight = node_loss_weight
        self.flag_reweight = flag_reweight

        self.objective = objective

        assert objective in ['score', 'diffusion', 'edm'], "Loss mode {:s} is not supported!".format(objective)

    def forward(self, net_pred_a, net_pred_x, net_target_a, net_target_x,  net_cond,
                adjs_perturbed, adjs_gt, x_perturbed, x_gt, node_flags,
                loss_weight=None, cond_val=None, flag_matching=False,
                reduction='mean'):
        if flag_matching:
            raise ValueError("Graph matching is not supported for node-adj loss!")
        reweight_coef = None
        regression_loss = self.get_regression_loss(net_pred_a, net_pred_x, net_target_a, net_target_x, net_cond,
                                                   node_flags, reweight_coef, loss_weight, cond_val, reduction)

        return regression_loss

    def get_regression_loss(self, pred_adj, pred_node, target_adj, target_node, net_cond,
                            node_flags, reweight_coef, loss_weight,
                            condition_true_values, reduction):
        """
        Compute regression loss for score estimation or epsilon-noise prediction.
        @param pred_adj:                [B, N, N] or [B, C, N, N]
        @param pred_node:               [B, N] or [B, C, N]
        @param target_adj:              [B, N, N] or [B, C, N, N]
        @param target_node:             [B, N] or [B, C, N]
        @param net_cond:                [B]
        @param node_flags:              [B, N] or [B, N, N]
        @param reweight_coef:           [B, N, N]
        @param loss_weight:             [B]
        @param condition_true_values:   [B]
        @param reduction:               str
        @return score_loss:             scalar or [B], loss per entry
        """
        loss_weight = torch.ones_like(net_cond).float() if loss_weight is None else loss_weight  # [B]
        _loss_weight = loss_weight.view(-1)
        batch_size = len(_loss_weight)
        # loss_weight = loss_weight[:, None, None]  # [B, N, N]
        if self.objective == "score":
            raise NotImplementedError
        elif self.objective in ["diffusion", 'edm']:
            square_loss_adj = (pred_adj - target_adj) ** 2  # [B, N, N] or [B, C, N, N]
            square_loss_node = (pred_node - target_node) ** 2  # [B, N] or [B, N, C]
            reweight_coef = 1.0 if reweight_coef is None else reweight_coef

            # [B, N, N] or [B, C, N, N]
            _loss_weight_shape = [batch_size] + [1] * (len(square_loss_adj.shape) - 1)
            square_loss_adj = square_loss_adj * reweight_coef * loss_weight.view(_loss_weight_shape)

            # [B, N] or [B, N, C]
            _loss_weight_shape = [batch_size] + [1] * (len(square_loss_node.shape) - 1)
            square_loss_node = square_loss_node * reweight_coef * loss_weight.view(_loss_weight_shape)

            square_loss_adj = mask_adjs(square_loss_adj, node_flags)  # [B, N, N] or [B, C, N, N]
            square_loss_node = mask_nodes(square_loss_node, node_flags)  # [B, N] or [B, N, C]

            # tensor shape reduction
            if len(node_flags.shape) == 2:
                num_adj_entries = node_flags.sum(dim=-1) ** 2       # [B]
                num_node_entries = node_flags.sum(dim=-1)           # [B]
            else:
                num_adj_entries = node_flags.sum(dim=[-1, -2])      # [B]
                num_node_entries = node_flags.sum(dim=[-1, -2])     # [B]

            if reduction == 'mean':
                square_loss_adj = square_loss_adj.sum() / num_adj_entries * self.edge_loss_weight       # scalar
                square_loss_node = square_loss_node.sum() / num_node_entries * self.edge_loss_weight    # scalar
            elif reduction is None or reduction == 'none':
                # keep the output in the shape of [B]
                if len(square_loss_adj.shape) == 3:
                    square_loss_adj = square_loss_adj.sum(dim=[-1, -2]) / num_adj_entries
                elif len(square_loss_adj.shape) == 4:
                    square_loss_adj = square_loss_adj.sum(dim=[-1, -2, -3]) / num_adj_entries / square_loss_adj.size(1)
                square_loss_adj = square_loss_adj * self.edge_loss_weight  # [B]

                if len(square_loss_node.shape) == 2:
                    square_loss_node = square_loss_node.sum(dim=-1) / num_node_entries
                elif len(square_loss_node.shape) == 3:
                    square_loss_node = square_loss_node.sum(dim=[-1, -2]) / num_node_entries / square_loss_node.size(-1)
                square_loss_node = square_loss_node * self.node_loss_weight  # [B]
            return square_loss_adj, square_loss_node
        else:
            raise NotImplementedError
