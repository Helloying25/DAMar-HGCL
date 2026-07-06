import importlib.util
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

EPS = 1e-12


def _format_lt_summary(context: Dict[str, np.ndarray], source: str, lt_dc: float) -> str:
    parents = context['parents']
    roots = context['roots']
    layer = context['layer']
    tree_id = context['tree_id']
    leaf_mask = context['leaf_mask']
    children = context['children']

    num_nodes = int(parents.shape[0])
    num_roots = int(roots.shape[0])
    unique_trees = int(np.unique(tree_id).shape[0])
    max_depth = int(layer.max()) if layer.size > 0 else 0
    leaf_count = int(leaf_mask.sum()) if leaf_mask.size > 0 else 0
    leaf_ratio = (100.0 * leaf_count / max(num_nodes, 1))

    tree_sizes = np.bincount(np.clip(tree_id, 0, None), minlength=max(unique_trees, 1))
    tree_sizes_sorted = np.sort(tree_sizes)[::-1]
    max_tree = int(tree_sizes_sorted[0]) if tree_sizes_sorted.size > 0 else 0
    min_tree = int(tree_sizes_sorted[-1]) if tree_sizes_sorted.size > 0 else 0
    mean_tree = float(tree_sizes.mean()) if tree_sizes.size > 0 else 0.0
    median_tree = int(np.median(tree_sizes)) if tree_sizes.size > 0 else 0

    top1_ratio = 100.0 * (tree_sizes_sorted[0] / max(num_nodes, 1)) if tree_sizes_sorted.size > 0 else 0.0
    top5_ratio = 100.0 * (tree_sizes_sorted[:5].sum() / max(num_nodes, 1)) if tree_sizes_sorted.size > 0 else 0.0

    singleton_count = int((tree_sizes == 1).sum()) if tree_sizes.size > 0 else 0
    small_tree_count = int((tree_sizes <= 5).sum()) if tree_sizes.size > 0 else 0

    root_preview = ', '.join(str(int(x)) for x in roots[:10].tolist())
    if num_roots > 10:
        root_preview += ', ...'

    max_branching = max((len(ch) for ch in children), default=0)

    lines = [
        f"(n_roots={num_roots}, lt_dc={lt_dc}) ---",
        '',
        '  [Leading Tree Diagnostic Report]',
        f"  - Source: {source}",
        f"  - Total nodes: {num_nodes} | Forest Size: {unique_trees} trees",
        f"  - Tree Size Stats: Max={max_tree}, Min={min_tree}, Mean={mean_tree:.2f}, Median={median_tree}",
        f"  - Dominance: Top-1 tree holds {top1_ratio:.2f}% nodes | Top-5 trees hold {top5_ratio:.2f}%",
        f"  - Fragmentation: Singletons (Size 1): {singleton_count} | Small trees (Size<=5): {small_tree_count}",
        f"  - Topology: Max Depth={max_depth} | Total Leaves={leaf_count} ({leaf_ratio:.2f}% of nodes) | Max Branching={max_branching}",
        f"  - Root IDs (Top10): [{root_preview}]",
        f"  [Tree Stats] Forest Size: {unique_trees} trees, Max Depth: {max_depth}",
    ]
    return '\n'.join(lines)


def _load_leading_tree_class():
    try:
        from .leading_tree import LeadingTree
        return LeadingTree
    except ImportError:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lt_file = os.path.join(current_dir, 'leading_tree.py')
        spec = importlib.util.spec_from_file_location('leading_tree_module', lt_file)
        module = importlib.util.module_from_spec(spec)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load LeadingTree from {lt_file}')
        spec.loader.exec_module(module)
        return module.LeadingTree


def _pairwise_l2(features: np.ndarray) -> np.ndarray:
    sq = np.sum(features * features, axis=1, keepdims=True)
    dist2 = np.maximum(sq + sq.T - 2.0 * np.matmul(features, features.T), 0.0)
    return np.sqrt(dist2, dtype=np.float32)


def _to_dense_tensor(mat) -> torch.Tensor:
    if isinstance(mat, torch.Tensor):
        out = mat
    else:
        out = torch.as_tensor(mat)
    if out.is_sparse:
        out = out.to_dense()
    return out.float()


def _cosine_similarity(mat: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    norm = torch.norm(mat, dim=-1, keepdim=True)
    norm = torch.where(norm > eps, norm, torch.full_like(norm, eps))
    sim = torch.mm(mat, mat.t()) / torch.mm(norm, norm.t())
    return torch.clamp(sim, min=-1.0, max=1.0)


def _row_random_walk(adj: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    n = adj.shape[0]
    eye = torch.eye(n, dtype=adj.dtype, device=adj.device)
    adj = adj + eye
    denom = adj.sum(dim=1, keepdim=True)
    denom = torch.where(denom > eps, denom, torch.full_like(denom, eps))
    return adj / denom


def _distance_from_similarity(sim: torch.Tensor) -> np.ndarray:
    sim = 0.5 * (sim + sim.t())
    dist = 1.0 - sim
    dist = torch.clamp(dist, min=0.0)
    dist.fill_diagonal_(0.0)
    return dist.detach().cpu().numpy().astype(np.float32)


def _compute_distance_matrix(
    node_features: torch.Tensor,
    metric: str,
    adjs: Optional[List[torch.Tensor]] = None,
) -> np.ndarray:
    metric = str(metric).lower()
    x = _to_dense_tensor(node_features)
    x_np = x.detach().cpu().numpy().astype(np.float32)

    if metric == 'l2':
        return _pairwise_l2(x_np)

    fea_sim = _cosine_similarity(x)
    if metric == 'fea_cos':
        return _distance_from_similarity(fea_sim)

    if adjs is None or len(adjs) == 0:
        adj_sim = fea_sim
        adjs_rw = None
    else:
        rw_list = []
        for adj in adjs:
            adj_dense = _to_dense_tensor(adj).to(x.device)
            rw_list.append(_row_random_walk(adj_dense))
        adjs_rw = torch.stack(rw_list, dim=0).mean(dim=0)
        adj_sim = _cosine_similarity(adjs_rw)

    if metric == 'ax':
        if adjs_rw is None:
            ax_feat = x
        else:
            ax_feat = torch.mm(adjs_rw, x)
        ax_sim = _cosine_similarity(ax_feat)
        return _distance_from_similarity(ax_sim)

    if metric == 'adj_only':
        dist = _distance_from_similarity(adj_sim)
        return dist

    sim_l = adj_sim * fea_sim
    sim_l = sim_l - torch.diag_embed(torch.diag(sim_l))

    sim_h_raw = (1.0 - adj_sim) * (1.0 - fea_sim)
    sim_h = 1.0 - sim_h_raw
    sim_h = sim_h - torch.diag_embed(torch.diag(sim_h))

    if metric == 'sim_l':
        return _distance_from_similarity(sim_l)
    if metric == 'sim_h':
        return _distance_from_similarity(sim_h)

    raise ValueError(f'Unsupported lt metric: {metric}. use l2|fea_cos|ax|adj_only|sim_l|sim_h')


def _build_children(parents: np.ndarray) -> List[List[int]]:
    num_nodes = len(parents)
    children = [[] for _ in range(num_nodes)]
    for node, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(node)
    return children


def _assign_tree_ids(parents: np.ndarray, roots: np.ndarray) -> np.ndarray:
    num_nodes = len(parents)
    tree_id = -np.ones((num_nodes,), dtype=np.int64)
    root_to_tid = {int(root): idx for idx, root in enumerate(roots.tolist())}

    for root in roots:
        tree_id[root] = root_to_tid[int(root)]

    for node in range(num_nodes):
        if tree_id[node] != -1:
            continue

        path = []
        cur = node
        while cur >= 0 and tree_id[cur] == -1:
            path.append(cur)
            cur = int(parents[cur])

        tid = tree_id[cur] if cur >= 0 else 0

        for n in path:
            tree_id[n] = tid

    return tree_id


def _log_min_max_norm(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    log_x = np.log(np.maximum(x, eps))
    min_val = float(np.min(log_x))
    max_val = float(np.max(log_x))
    if max_val - min_val <= eps:
        return np.zeros_like(log_x, dtype=np.float32)
    return ((log_x - min_val) / (max_val - min_val)).astype(np.float32)


def build_lt_context(
    node_features: torch.Tensor,
    lt_num: int,
    dc: float = 0.1,
    print_info: bool = True,
    metric: str = 'l2',
    adjs: Optional[List[torch.Tensor]] = None,
    labels: Optional[np.ndarray] = None,
    min_tree_size: int = 5,
) -> Dict[str, np.ndarray]:
    LeadingTree = _load_leading_tree_class()
    x = _to_dense_tensor(node_features).detach().cpu().numpy().astype(np.float32)
    distance_matrix = _compute_distance_matrix(node_features=node_features, metric=metric, adjs=adjs)
    lt_num = max(1, min(int(lt_num), x.shape[0]))

    lt = LeadingTree(X_train=x, dc=dc, lt_num=lt_num, D=distance_matrix)
    lt.fit()

    parents = lt.Pa.astype(np.int64)
    roots = lt.gamma_D[:lt_num].astype(np.int64)
    rho = lt.density.astype(np.float32)
    gamma_raw = lt.gamma.astype(np.float32)
    layer = lt.layer.astype(np.int64)
    children = _build_children(parents)
    tree_id = _assign_tree_ids(parents, roots)
    leaf_mask = np.array([len(children[idx]) == 0 for idx in range(len(children))], dtype=np.bool_)

    if min_tree_size > 1:
        unique_tree_ids, tree_sizes = np.unique(tree_id, return_counts=True)
        small_trees = unique_tree_ids[tree_sizes < min_tree_size]

        if len(small_trees) > 0:
            for small_tree_id in small_trees:
                small_tree_nodes = np.where(tree_id == small_tree_id)[0]

                for node in small_tree_nodes:
                    min_dist = float('inf')
                    nearest_tree = -1

                    for other_tree_id, other_tree_size in zip(unique_tree_ids, tree_sizes):
                        if other_tree_size >= min_tree_size and other_tree_id != small_tree_id:
                            other_tree_nodes = np.where(tree_id == other_tree_id)[0]
                            if len(other_tree_nodes) > 0:
                                dists = distance_matrix[node, other_tree_nodes]
                                min_dist_to_tree = np.min(dists)
                                if min_dist_to_tree < min_dist:
                                    min_dist = min_dist_to_tree
                                    nearest_tree = other_tree_id

                    if nearest_tree != -1:
                        tree_id[node] = nearest_tree
                        parents[node] = roots[nearest_tree] if nearest_tree < len(roots) else -1

            unique_tree_ids, tree_sizes = np.unique(tree_id, return_counts=True)
            valid_trees = unique_tree_ids[tree_sizes >= min_tree_size]

            new_roots = []
            for tid in valid_trees:
                tree_nodes = np.where(tree_id == tid)[0]
                for node in tree_nodes:
                    if parents[node] == -1 or parents[node] not in tree_nodes:
                        new_roots.append(node)
                        parents[node] = -1
                        break

            if len(new_roots) > 0:
                roots = np.array(new_roots, dtype=np.int64)

            layer = np.zeros_like(layer)
            for i, node in enumerate(np.argsort(-rho)):
                parent = parents[node]
                if parent >= 0:
                    layer[node] = layer[parent] + 1

            if print_info:
                print(f"[Min Tree Size Filter] Applied min_tree_size={min_tree_size}")
                print(f"[Min Tree Size Filter] Valid trees after filtering: {len(roots)}")
                print(f"[Min Tree Size Filter] Removed {len(small_trees)} small trees")

    tree_purity = np.ones_like(gamma_raw, dtype=np.float32)
    if labels is not None:
        unique_trees = np.unique(tree_id)
        for tid in unique_trees:
            tree_nodes = np.where(tree_id == tid)[0]
            if len(tree_nodes) > 0:
                tree_labels = labels[tree_nodes]
                counts = np.bincount(tree_labels)
                majority = counts.max()
                purity = majority / len(tree_labels)
                tree_purity[tree_nodes] = purity

    beta = np.maximum(layer.astype(np.float32) - 1.0, 0.0)
    psi_raw = beta / np.maximum(rho, EPS)

    gamma = _log_min_max_norm(gamma_raw)
    divergence = _log_min_max_norm(psi_raw)

    context = {
        'parents': parents,
        'roots': roots,
        'gamma': gamma,
        'gamma_raw': gamma_raw,
        'rho': rho,
        'layer': layer,
        'tree_id': tree_id,
        'children': children,
        'leaf_mask': leaf_mask,
        'divergence': divergence,
        'psi_raw': psi_raw,
        'tree_purity': tree_purity,
        'distance_matrix': distance_matrix,
    }

    report = _format_lt_summary(context, source=f'build[{metric}]', lt_dc=float(dc))
    if print_info:
        print(report)
    context['lt_report'] = report

    return context
