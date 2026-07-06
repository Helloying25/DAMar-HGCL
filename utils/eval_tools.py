import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

EVAL_TRAIN_RATIOS = [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]


def idx_to_one_hot(idx_arr):
    one_hot = np.zeros((idx_arr.shape[0], idx_arr.max() + 1))
    one_hot[np.arange(idx_arr.shape[0]), idx_arr] = 1
    return one_hot


def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def evaluate_results_nc_detail(embeddings, labels, num_classes, verbose=True):
    repeat = 10
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels, repeat=repeat)
    if verbose:
        print('Macro-F1: ' + ', '.join(['{:.4f}~{:.4f}({:.2f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                        (macro_f1_mean, macro_f1_std), train_size in
                                        zip(svm_macro_f1_list, EVAL_TRAIN_RATIOS)]))
        print('Micro-F1: ' + ', '.join(['{:.4f}~{:.4f}({:.2f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                        (micro_f1_mean, micro_f1_std), train_size in
                                        zip(svm_micro_f1_list, EVAL_TRAIN_RATIOS)]))

    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes, repeat=repeat)

    macro_mean = np.array([x for (x, _) in svm_macro_f1_list])
    macro_std = np.array([y for (_, y) in svm_macro_f1_list])
    micro_mean = np.array([x for (x, _) in svm_micro_f1_list])
    micro_std = np.array([y for (_, y) in svm_micro_f1_list])
    return {
        'train_ratios': np.array(EVAL_TRAIN_RATIOS, dtype=np.float64),
        'macro_mean': macro_mean,
        'macro_std': macro_std,
        'micro_mean': micro_mean,
        'micro_std': micro_std,
        'nmi_mean': float(nmi_mean),
        'nmi_std': float(nmi_std),
        'ari_mean': float(ari_mean),
        'ari_std': float(ari_std),
    }


def evaluate_results_nc(embeddings, labels, num_classes, verbose=True):
    detail = evaluate_results_nc_detail(embeddings, labels, num_classes, verbose=verbose)
    return detail['macro_mean'], detail['micro_mean'], detail['nmi_mean'], detail['ari_mean']
