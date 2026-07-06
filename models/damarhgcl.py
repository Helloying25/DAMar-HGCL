import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.mg_encoder import Mg_encoder
from models.feat_encoder import FeatEncoder
from models.lt_builder import build_lt_context


class DAMarHGCL(nn.Module):
    def __init__(self, topo_dim, hidden_dim, feats_dim_list, feat_drop, att_drop, mp_num):
        super(DAMarHGCL, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.nd = FeatEncoder(topo_dim, hidden_dim, feat_drop)
        self.mg = Mg_encoder(hidden_dim, mp_num, feat_drop, attn_drop=att_drop)

        self.lt_context = None
        self.gamma_weight = None
        self.divergence_weight = None
        self.tree_to_leaves = None
        self.tree_to_all_nodes = None
        self.tree_to_root = None
        self.tree_to_candidate_leaves = None
        self.diff_tree_cache = None
        self.same_tree_cache = None

    def forward(self, features, ADJ, MetaGraph, L_norm):
        h_all = []
        for i in range(len(features)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](features[i]))))
        h, _ = self.nd(h_all[0], ADJ)
        z_mg, _ = self.mg(h_all[0], MetaGraph, L_norm)
        return z_mg, h

    def embed(self, features, ADJ, MetaGraph, L_norm):
        h_all = []
        for i in range(len(features)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](features[i]))))
        h, _ = self.nd(h_all[0], ADJ)
        z_mg, _ = self.mg(h_all[0], MetaGraph, L_norm)
        z = torch.cat((h, z_mg), 1)
        return z

    def build_lt_context(self, node_features, lt_num, dc=0.1, print_info=True,
                         metric='l2', adjs=None, labels=None, min_tree_size=5):
        self.lt_context = build_lt_context(
            node_features, lt_num=lt_num, dc=dc, print_info=print_info,
            metric=metric, adjs=adjs, labels=labels, min_tree_size=min_tree_size,
        )
        self.gamma_weight = np.asarray(self.lt_context['gamma'], dtype=np.float32)
        self.divergence_weight = np.asarray(self.lt_context['divergence'], dtype=np.float32)
        self._prebuild_tree_indices()

        if print_info:
            gamma = self.gamma_weight
            divergence = self.divergence_weight
            corr = np.corrcoef(gamma, divergence)[0, 1]
            print(f"[LT] gamma: mean={gamma.mean():.3f}, med={np.median(gamma):.3f}, std={gamma.std():.3f} | "
                  f"div: mean={divergence.mean():.3f}, med={np.median(divergence):.3f}, std={divergence.std():.3f} | corr={corr:.3f}")

        return self.lt_context

    def _prebuild_tree_indices(self):
        tree_id = self.lt_context['tree_id']
        roots = self.lt_context['roots']
        leaf_mask = self.lt_context['leaf_mask']
        divergence = self.lt_context.get('divergence', np.ones(len(tree_id), dtype=np.float32))

        num_trees = len(roots)

        self.tree_to_leaves = {}
        self.tree_to_all_nodes = {}
        self.tree_to_root = {}
        self.tree_to_candidate_leaves = {}

        for tree_idx in range(num_trees):
            root_node = roots[tree_idx]
            self.tree_to_root[tree_idx] = root_node

            nodes_in_tree = np.where(tree_id == tree_idx)[0]
            self.tree_to_all_nodes[tree_idx] = nodes_in_tree

            leaves_in_tree = nodes_in_tree[leaf_mask[nodes_in_tree]]
            self.tree_to_leaves[tree_idx] = leaves_in_tree

            non_root_nodes = nodes_in_tree[nodes_in_tree != root_node]
            if len(non_root_nodes) > 0:
                node_divergences = divergence[non_root_nodes]
                sorted_indices = np.argsort(-node_divergences)
                self.tree_to_candidate_leaves[tree_idx] = non_root_nodes[sorted_indices]
            else:
                self.tree_to_candidate_leaves[tree_idx] = np.array([], dtype=np.int64)

        num_nodes = len(tree_id)
        self.diff_tree_cache = {}
        self.same_tree_cache = {}
        for node_idx in range(num_nodes):
            anchor_tree = tree_id[node_idx]
            diff_trees = [t for t in range(num_trees) if t != anchor_tree]
            self.diff_tree_cache[node_idx] = np.array(diff_trees, dtype=np.int64)

            same_tree_mask = tree_id == anchor_tree
            same_tree_nodes = np.where(same_tree_mask)[0]
            same_tree_nodes = same_tree_nodes[same_tree_nodes != node_idx]
            self.same_tree_cache[node_idx] = same_tree_nodes

    def sample_structured_pairs(self, num_nodes, num_neg_trees, num_leaves_per_tree,
                                num_random_per_tree, num_lt_pos=1, seed=None):
        if self.lt_context is None:
            raise ValueError("LT context not built. Call build_lt_context first.")

        neg_tree_info = []

        for tree_sample_id in range(num_neg_trees):
            pair_seed = seed + tree_sample_id if seed is not None else None
            rng_pair = np.random.default_rng(pair_seed)
            neg_root_idx = np.zeros((num_nodes,), dtype=np.int64)
            neg_leaf_idx_for_tree = np.zeros((num_leaves_per_tree, num_nodes), dtype=np.int64)
            neg_rand_idx_for_tree = np.zeros((num_random_per_tree, num_nodes), dtype=np.int64)

            for idx in range(num_nodes):
                diff_trees = self.diff_tree_cache[idx]
                if len(diff_trees) == 0:
                    neg_root_idx[idx] = (idx + 1) % num_nodes
                    for m in range(num_leaves_per_tree):
                        neg_leaf_idx_for_tree[m, idx] = (idx + 2 + m) % num_nodes
                    for r in range(num_random_per_tree):
                        neg_rand_idx_for_tree[r, idx] = (idx + 3 + r) % num_nodes
                    continue

                selected_tree_idx = rng_pair.choice(diff_trees)
                selected_root = self.tree_to_root[selected_tree_idx]
                neg_root_idx[idx] = selected_root

                all_nodes_in_tree = self.tree_to_all_nodes[selected_tree_idx]
                selected_leaves = self.tree_to_leaves[selected_tree_idx]
                if len(selected_leaves) == 0:
                    selected_leaves = self.tree_to_candidate_leaves[selected_tree_idx]
                if len(selected_leaves) == 0:
                    selected_leaves = all_nodes_in_tree[all_nodes_in_tree != selected_root]
                if len(selected_leaves) == 0:
                    selected_leaves = np.array([selected_root])

                num_leaves_actual = len(selected_leaves)
                for m in range(num_leaves_per_tree):
                    neg_leaf_idx_for_tree[m, idx] = selected_leaves[m % num_leaves_actual]

                if len(all_nodes_in_tree) >= num_random_per_tree:
                    selected_rand = rng_pair.choice(all_nodes_in_tree, size=num_random_per_tree, replace=False)
                elif len(all_nodes_in_tree) > 0:
                    selected_rand = list(all_nodes_in_tree)
                    remaining_rand = num_random_per_tree - len(all_nodes_in_tree)
                    if remaining_rand > 0:
                        repeat_rand = rng_pair.choice(all_nodes_in_tree, size=remaining_rand, replace=True)
                        selected_rand.extend(list(repeat_rand))
                    selected_rand = np.array(selected_rand)
                else:
                    selected_rand = np.array([selected_root] * num_random_per_tree)
                for r in range(num_random_per_tree):
                    neg_rand_idx_for_tree[r, idx] = selected_rand[r]

            neg_tree_info.append((neg_root_idx, neg_leaf_idx_for_tree, neg_rand_idx_for_tree))

        if num_lt_pos <= 0:
            pos_info = None
        else:
            pos_info = np.zeros((num_lt_pos, num_nodes), dtype=np.int64)
            for k in range(num_lt_pos):
                pos_seed = seed + 9999 + k if seed is not None else None
                pos_rng = np.random.default_rng(pos_seed)
                for idx in range(num_nodes):
                    same_tree_nodes = self.same_tree_cache[idx]
                    if len(same_tree_nodes) > 0:
                        pos_info[k, idx] = pos_rng.choice(same_tree_nodes)
                    else:
                        pos_info[k, idx] = idx

        self.neg_tree_info = neg_tree_info
        self.pos_info = pos_info
        return self.neg_tree_info, self.pos_info

    def compute_structured_manifold_loss(self, z_mg, z_feat, margin_root, margin_leaf,
                                         margin_random_base, lambda_main, lambda_root,
                                         lambda_leaf, use_coupled_loss, device):
        if self.neg_tree_info is None:
            raise ValueError("Negative pairs not sampled. Call sample_structured_pairs first.")

        num_neg_trees = len(self.neg_tree_info)
        if num_neg_trees == 0:
            return torch.tensor(0.0, device=device), 0.0, 0.0, 0.0

        max_leaves = self.neg_tree_info[0][1].shape[0]
        num_random_per_tree = self.neg_tree_info[0][2].shape[0]

        s_p_self = F.pairwise_distance(z_mg, z_feat)

        if self.pos_info is not None:
            pos_idx = torch.as_tensor(self.pos_info, dtype=torch.long, device=device)
            z_feat_pos = z_feat[pos_idx.reshape(-1)]
            z_mg_exp_pos = z_mg.unsqueeze(0).expand(pos_idx.shape[0], -1, -1).reshape(-1, z_mg.shape[1])
            s_p_lt_all = F.pairwise_distance(z_mg_exp_pos, z_feat_pos).reshape(pos_idx.shape[0], z_mg.shape[0])
            s_p_lt = s_p_lt_all.mean(dim=0)
            s_p = (s_p_self + s_p_lt) / 2.0
        else:
            s_p = s_p_self

        gamma_t = torch.as_tensor(self.gamma_weight, dtype=torch.float32, device=device)
        divergence_t = torch.as_tensor(self.divergence_weight, dtype=np.float32, device=device)

        num_nodes = z_mg.shape[0]

        all_root_indices = []
        all_leaf_indices = []
        all_rand_indices = []

        for (root_idx, leaf_idx, rand_idx) in self.neg_tree_info:
            all_root_indices.append(torch.as_tensor(root_idx, dtype=torch.long, device=device))
            all_leaf_indices.append(torch.as_tensor(leaf_idx, dtype=torch.long, device=device))
            all_rand_indices.append(torch.as_tensor(rand_idx, dtype=torch.long, device=device))

        all_root_indices = torch.stack(all_root_indices, dim=0)
        all_leaf_indices = torch.stack(all_leaf_indices, dim=0)
        all_rand_indices = torch.stack(all_rand_indices, dim=0)

        neg_root_z = z_mg[all_root_indices.reshape(-1)]
        z_mg_exp_root = z_mg.unsqueeze(0).expand(num_neg_trees, -1, -1).reshape(-1, z_mg.shape[1])
        d_root_all = F.pairwise_distance(z_mg_exp_root, neg_root_z).reshape(num_neg_trees, num_nodes)

        gamma_neg_all = gamma_t[all_root_indices]
        dyn_margin_root_all = margin_root * (0.5 + gamma_neg_all)
        s_p_exp_root = s_p.unsqueeze(0).expand_as(d_root_all)
        root_term_all = torch.relu(s_p_exp_root - d_root_all + dyn_margin_root_all)

        neg_leaf_z = z_mg[all_leaf_indices.reshape(-1)]
        z_mg_exp_leaf = z_mg.unsqueeze(0).unsqueeze(0).expand(num_neg_trees, max_leaves, -1, -1).reshape(-1, z_mg.shape[1])
        d_leaf_all = F.pairwise_distance(z_mg_exp_leaf, neg_leaf_z).reshape(num_neg_trees, max_leaves, num_nodes)

        psi_neg_all = divergence_t[all_leaf_indices]
        dyn_margin_leaf_all = margin_leaf * (0.5 + psi_neg_all)
        s_p_exp_leaf = s_p.unsqueeze(0).unsqueeze(0).expand_as(d_leaf_all)
        leaf_term_all = torch.relu(s_p_exp_leaf - d_leaf_all + dyn_margin_leaf_all)
        leaf_per_tree_avg = leaf_term_all.mean(dim=1)

        neg_rand_z = z_mg[all_rand_indices.reshape(-1)]
        z_mg_exp_rand = z_mg.unsqueeze(0).unsqueeze(0).expand(num_neg_trees, num_random_per_tree, -1, -1).reshape(-1, z_mg.shape[1])
        d_main_all = F.pairwise_distance(z_mg_exp_rand, neg_rand_z).reshape(num_neg_trees, num_random_per_tree, num_nodes)

        gamma_main_all = gamma_t[all_rand_indices]
        psi_main_all = divergence_t[all_rand_indices]
        status_main_all = torch.max(gamma_main_all, psi_main_all)
        dyn_margin_main_all = margin_random_base * (0.5 + status_main_all)
        s_p_exp_main = s_p.unsqueeze(0).unsqueeze(0).expand_as(d_main_all)
        main_term_all = torch.relu(s_p_exp_main - d_main_all + dyn_margin_main_all)
        main_per_tree_avg = main_term_all.mean(dim=1)

        loss_root_avg = root_term_all.mean()
        loss_leaf_avg = leaf_term_all.mean()
        loss_main_avg = main_term_all.mean()

        if use_coupled_loss:
            per_tree_loss = (lambda_main * main_per_tree_avg
                             + lambda_root * root_term_all
                             + lambda_leaf * leaf_per_tree_avg)
            loss_total = per_tree_loss.mean()
        else:
            loss_total = (lambda_main * main_term_all.mean()
                          + lambda_root * root_term_all.mean()
                          + lambda_leaf * leaf_term_all.mean())

        return loss_total, loss_root_avg, loss_leaf_avg, loss_main_avg

    def prepare_lt_and_sample(self, args, features, labels, nb_nodes, device):
        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

        lt_context = self.build_lt_context(
            features[0],
            lt_num=args['lt_num'],
            dc=args['lt_dc'],
            print_info=args.get('lt_print_tree', True),
            metric=args['lt_metric'],
            labels=labels_np,
            min_tree_size=args['num_random_per_tree'],
        )
        lt_report = str(lt_context.get('lt_report', '')).strip()
        return lt_report
