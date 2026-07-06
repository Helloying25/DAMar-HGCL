import os
import random
import argparse
import time
import numpy as np
import torch
from datetime import datetime

from utils.data import load_ACM_data, load_DBLP_data, load_IMDB_data, load_YELP_data, load_Aminer_data, load_FreeBase_data
from utils.config import get_dataset_params, list_datasets
from utils.eval_tools import evaluate_results_nc_detail
from models.damarhgcl import DAMarHGCL


def set_random_seed(seed=164):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def trainandeval(args, ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask):
    nb_nodes = features[0].shape[0]
    num_classes_val = int(labels.max()) + 1
    device = args['device']

    ckpt_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_path = f"checkpoint/{args['dataset']}_{args['seed']}_{ckpt_tag}.pt"

    features = [f.to(device) if isinstance(f, torch.Tensor) else torch.FloatTensor(f).to(device) for f in features]
    ADJ = ADJ.to(device) if isinstance(ADJ, torch.Tensor) else torch.FloatTensor(ADJ).to(device)
    MG = [m.to(device) if isinstance(m, torch.Tensor) else torch.FloatTensor(m).to(device) for m in MG]
    labels_t = labels.to(device) if isinstance(labels, torch.Tensor) else torch.LongTensor(labels).to(device)

    model = DAMarHGCL(
        topo_dim=ADJ.shape[1],
        hidden_dim=args['hidden_dim'],
        feats_dim_list=[f.shape[1] for f in features],
        feat_drop=args['feat_drop'],
        att_drop=args['att_drop'],
        mp_num=len(MG),
    ).to(device)

    lt_build_start = time.time()
    lt_report = model.prepare_lt_and_sample(args, features, labels, nb_nodes, device)
    lt_build_time = time.time() - lt_build_start

    set_random_seed(args['seed'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    best_loss = 9999.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args['num_epochs']):
        optimizer.zero_grad()
        model.train()

        neg_tree_info, pos_info = model.sample_structured_pairs(
            num_nodes=nb_nodes,
            num_neg_trees=args['num_neg_trees'],
            num_leaves_per_tree=args['num_leaves_per_tree'],
            num_random_per_tree=args['num_random_per_tree'],
            num_lt_pos=args['num_lt_pos'],
            seed=args['seed'] + epoch,
        )

        z_mg, h = model(features, ADJ, MG, args['L_norm'])

        loss, loss_root, loss_leaf, loss_main = model.compute_structured_manifold_loss(
            z_mg, h,
            margin_root=args['margin_root'],
            margin_leaf=args['margin_leaf'],
            margin_random_base=args['margin_random_base'],
            lambda_main=args['lambda_main'],
            lambda_root=args['lambda_root'],
            lambda_leaf=args['lambda_leaf'],
            use_coupled_loss=args['use_coupled_loss'],
            device=device,
        )

        loss = loss * args['w_loss']

        print(f"Epoch {epoch+1:04d}: loss={loss.item():.6f}, loss_root={loss_root:.6f}, loss_leaf={loss_leaf:.6f}, loss_main={loss_main:.6f}")

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args['patience']:
                print(f"Early Stopping at Epoch {epoch+1}")
                break

    print(f"Loaded Best Model from Epoch {best_epoch}, loss={best_loss:.6f}")

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        z = model.embed(features, ADJ, MG, args['L_norm'])

    eval_x = z[test_idx].detach().cpu().numpy()
    eval_y = labels_t[test_idx].cpu().numpy()

    eval_detail = evaluate_results_nc_detail(eval_x, eval_y, num_classes_val, verbose=False)
    train_ratios = eval_detail['train_ratios']
    svm_macro_avg = eval_detail['macro_mean']
    svm_macro_std_avg = eval_detail['macro_std']
    svm_micro_avg = eval_detail['micro_mean']
    svm_micro_std_avg = eval_detail['micro_std']
    nmi_avg = eval_detail['nmi_mean']
    nmi_std_avg = eval_detail['nmi_std']
    ari_avg = eval_detail['ari_mean']
    ari_std_avg = eval_detail['ari_std']

    macro_str = ", ".join([f"{m:.4f}±{s:.4f}({r:.2f})" for m, s, r in zip(svm_macro_avg, svm_macro_std_avg, train_ratios)])
    micro_str = ", ".join([f"{m:.4f}±{s:.4f}({r:.2f})" for m, s, r in zip(svm_micro_avg, svm_micro_std_avg, train_ratios)])

    best_macro_idx = np.argmax(svm_macro_avg)
    best_micro_idx = np.argmax(svm_macro_avg)

    print("-" * 60)
    print(f"Macro-F1(mean±std): {macro_str}")
    print(f"Micro-F1(mean±std): {micro_str}")
    print(f"Best Macro-F1: {svm_macro_avg[best_macro_idx]:.4f}±{svm_macro_std_avg[best_macro_idx]:.4f}({train_ratios[best_macro_idx]:.2f})")
    print(f"Best Micro-F1: {svm_micro_avg[best_micro_idx]:.4f}±{svm_micro_std_avg[best_micro_idx]:.4f}({train_ratios[best_micro_idx]:.2f})")
    print(f"NMI: {nmi_avg:.6f}±{nmi_std_avg:.6f}")
    print(f"ARI: {ari_avg:.6f}±{ari_std_avg:.6f}")
    print("-" * 60)

    result_file = args.get('result_file', f"results/damarhgcl_results_{args['dataset']}.txt")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    with open(result_file, 'a') as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"model=DAMarHGCL(num_lt_pos={args['num_lt_pos']})\n")
        f.write(f"run_started_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"args={args}\n")
        f.write(f"dataset={args['dataset']}\n")
        f.write(f"seed={args['seed']}\n")
        f.write(f"margin_root={args['margin_root']}, margin_leaf={args['margin_leaf']}, lambda_root={args['lambda_root']}, lambda_leaf={args['lambda_leaf']}\n")
        f.write(f"use_coupled_loss={args['use_coupled_loss']}\n")
        f.write(f"best_epochs={best_epoch}\n")
        f.write(f"best_losses={best_loss:.6f}\n")
        if lt_report:
            f.write(f"{lt_report}\n")
        f.write(f"Macro-F1(mean±std): {macro_str}\n")
        f.write(f"Micro-F1(mean±std): {micro_str}\n")
        f.write(f"Best Macro-F1: {svm_macro_avg[best_macro_idx]:.4f}±{svm_macro_std_avg[best_macro_idx]:.4f}({train_ratios[best_macro_idx]:.2f})\n")
        f.write(f"Best Micro-F1: {svm_micro_avg[best_micro_idx]:.4f}±{svm_micro_std_avg[best_micro_idx]:.4f}({train_ratios[best_micro_idx]:.2f})\n")
        f.write(f"NMI: {nmi_avg:.6f}±{nmi_std_avg:.6f}\n")
        f.write(f"ARI: {ari_avg:.6f}±{ari_std_avg:.6f}\n")
        f.write(f"{'=' * 60}\n\n")

    print(f"Results Saved to {result_file}")

    return float(svm_macro_avg[best_macro_idx])


def main():
    parser = argparse.ArgumentParser(description='DAMarHGCL')
    parser.add_argument('--dataset', type=str, default='ACM', choices=list_datasets())
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--att_drop', type=float, default=None)
    parser.add_argument('--feat_drop', type=float, default=None)
    parser.add_argument('--L_norm', type=int, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--repeat', type=int, default=None)
    parser.add_argument('--w_loss', type=float, default=None)
    parser.add_argument('--lambda_root', type=float, default=None)
    parser.add_argument('--lambda_leaf', type=float, default=None)
    parser.add_argument('--lambda_main', type=float, default=None)
    parser.add_argument('--margin_root', type=float, default=None)
    parser.add_argument('--margin_leaf', type=float, default=None)
    parser.add_argument('--margin_random_base', type=float, default=None)
    parser.add_argument('--num_neg_trees', type=int, default=None)
    parser.add_argument('--num_leaves_per_tree', type=int, default=None)
    parser.add_argument('--num_random_per_tree', type=int, default=None)
    parser.add_argument('--num_lt_pos', type=int, default=None)
    parser.add_argument('--lt_num', type=int, default=None)
    parser.add_argument('--lt_dc', type=float, default=None)
    parser.add_argument('--lt_metric', type=str, default=None)
    parser.add_argument('--use_coupled_loss', type=str, default=None)
    parser.add_argument('--data_prefix', type=str, default=None)
    parser.add_argument('--result_file', type=str, default=None)
    parser.add_argument('--ratio', type=int, default=None)
    parser.add_argument('--split_file', type=str, default=None)
    args = parser.parse_args()

    params = get_dataset_params(args.dataset)

    override_keys = ['dataset', 'device', 'seed', 'lr', 'hidden_dim', 'att_drop', 'feat_drop',
                     'L_norm', 'weight_decay', 'patience', 'num_epochs', 'repeat', 'w_loss',
                     'lambda_root', 'lambda_leaf', 'lambda_main', 'margin_root', 'margin_leaf',
                     'margin_random_base', 'num_neg_trees', 'num_leaves_per_tree',
                     'num_random_per_tree', 'num_lt_pos', 'lt_num', 'lt_dc', 'lt_metric',
                     'use_coupled_loss', 'data_prefix', 'result_file', 'ratio', 'split_file']

    for key in override_keys:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val

    if isinstance(params.get('use_coupled_loss'), str):
        params['use_coupled_loss'] = params['use_coupled_loss'].lower() == 'true'

    print("=" * 60)
    print(f"Starting DAMarHGCL on Dataset: {params['dataset']}")
    print(f"[Method] DAMarHGCL, lr={params['lr']}")
    print(f"[Parameters] margin_root={params['margin_root']}, margin_leaf={params['margin_leaf']}, "
          f"lambda_main={params['lambda_main']}, lambda_root={params['lambda_root']}, lambda_leaf={params['lambda_leaf']}, "
          f"use_coupled_loss={params['use_coupled_loss']}")
    print(f"[Tuning Params] num_neg_trees={params['num_neg_trees']}, "
          f"num_leaves_per_tree={params['num_leaves_per_tree']}, num_random_per_tree={params['num_random_per_tree']}")
    print("=" * 60)

    set_random_seed(params['seed'])

    dataset_name = params['dataset']
    if dataset_name == 'ACM':
        data = load_ACM_data(prefix=params.get('data_prefix'), split_file=params.get('split_file'))
    elif dataset_name == 'DBLP':
        data = load_DBLP_data(prefix=params.get('data_prefix'))
    elif dataset_name == 'IMDB':
        data = load_IMDB_data(prefix=params.get('data_prefix'), ratio=params.get('ratio'),
                              split_file=params.get('split_file'))
    elif dataset_name == 'YELP':
        data = load_YELP_data(prefix=params.get('data_prefix'))
    elif dataset_name == 'Aminer':
        data = load_Aminer_data(ratio=params.get('ratio', 20))
    elif dataset_name == 'FreeBase':
        data = load_FreeBase_data(ratio=params.get('ratio', 20))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if len(data) == 10:
        ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask = data
    else:
        ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask, pos = data

    os.makedirs('checkpoint', exist_ok=True)

    for r in range(params.get('repeat', 1)):
        trainandeval(params, ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask)


if __name__ == '__main__':
    main()
