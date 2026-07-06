import numpy as np
import scipy
import os
import torch
import torch.nn.functional as F
import torch as th
import scipy.sparse as sp

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'DATASET')


def _dataset_dir(dataset_name, prefix=None):
    return prefix if prefix is not None else os.path.join(DATASET_ROOT, dataset_name)


def load_ACM_data(prefix=None, split_file=None):
    prefix = _dataset_dir('ACM', prefix)

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()

    # features_0 = np.eye(4019)
    # features_1 =np.eye(7167)
    # features_2 =np.eye(60)
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)

    features_0 = F.normalize(features_0, dim=1, p=2)
    # features_1 = F.normalize(features_1, dim=1, p=2)
    # features_2 = F.normalize(features_2, dim=1, p=2)
    features = [features_0, features_1, features_2]
    pos = sp.load_npz(prefix + '/pos.npz')
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)

    if split_file is not None:
        train_val_test_idx = np.load(os.path.join(prefix, split_file))
    else:
        train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    type_mask = np.load(prefix + '/node_types.npy')
    num_classes = 3

    

    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz').A
    PP = ADJ[0:4019, 0:4019]
    PP = torch.from_numpy(PP).type(torch.FloatTensor)
    PP = F.normalize(PP, dim=1, p=2)
    ADJ = ADJ[:4019, 4019:]
    ADJ = torch.FloatTensor(ADJ)
    ADJ = F.normalize(ADJ, dim=1, p=2)

    
    nei_a = np.load(prefix + "/nei_a.npy", allow_pickle=True)
    nei_s = np.load(prefix + "/nei_s.npy", allow_pickle=True)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_s = [th.LongTensor(i) for i in nei_s]
    NS = [nei_a, nei_s]

    
    PAP = scipy.sparse.load_npz(prefix + '/pap.npz').A
    PAP = torch.from_numpy(PAP).type(torch.FloatTensor)
    PAP = F.normalize(PAP, dim=1, p=2)

    PSP = scipy.sparse.load_npz(prefix + '/psp.npz').A
    PSP = torch.from_numpy(PSP).type(torch.FloatTensor)
    PSP = F.normalize(PSP, dim=1, p=1)

    MG = [PAP, PSP,PP]

    return ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask,pos





def load_IMDB_data(prefix=None, ratio=None, split_file=None):
    prefix = _dataset_dir('IMDB', prefix)
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)

    features_0 = F.normalize(features_0, dim=1, p=2)
    features_1 = F.normalize(features_1, dim=1, p=2)
    features_2 = F.normalize(features_2, dim=1, p=2)

    features = [features_0, features_1, features_2]

    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    if split_file is not None:
        train_val_test_idx = np.load(os.path.join(prefix, split_file))
    elif ratio == 20:
        train_val_test_idx = np.load(prefix + '/train_val_test_idx_20.npz')
    elif ratio == 40:
        train_val_test_idx = np.load(prefix + '/train_val_test_idx_40.npz')
    elif ratio == 60:
        train_val_test_idx = np.load(prefix + '/train_val_test_idx_60.npz')
    else:
        train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    type_mask = np.load(prefix + '/node_types.npy')
    num_classes = 3

    
    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz').A
    ADJ = ADJ[:4278, 4278:]
    ADJ = torch.FloatTensor(ADJ)
    ADJ = F.normalize(ADJ, dim=1, p=1)

    
    nei_a = np.load(prefix + "/nei_a.npy", allow_pickle=True)
    nei_d = np.load(prefix + "/nei_d.npy", allow_pickle=True)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_d = [th.LongTensor(i) for i in nei_d]
    NS = [nei_d, nei_a]


    
    MDM = np.load(prefix + '/mdm.npy')
    MDM = torch.from_numpy(MDM).type(torch.FloatTensor)
    MDM = F.normalize(MDM, dim=1, p=2)

    MAM = np.load(prefix + '/mam.npy')
    MAM = torch.from_numpy(MAM).type(torch.FloatTensor)
    MAM = F.normalize(MAM, dim=1, p=2)

    MG = [MDM, MAM]
    pos = sp.load_npz(prefix + '/pos.npz')
    return ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask





def load_DBLP_data(prefix=None):
    prefix = _dataset_dir('DBLP', prefix)
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()#a
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()#p
    features_2 = np.load(prefix + '/features_2.npy')#t
    # features_0 = np.eye(4057)
    # features_1 = np.eye(14328)
    # features_2 = np.eye(7723)
    features_3 = np.eye(20)#c
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features_3 = torch.FloatTensor(features_3)
    pos = sp.load_npz(prefix + '/pos.npz')
    # features_0 = F.normalize(features_0, dim=1, p=0)
    #features_1 = F.normalize(features_1, dim=1, p=0)
    #features_2 = F.normalize(features_2, dim=1, p=0)

    features = [features_0, features_1, features_2, features_3]


    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)

    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    type_mask = np.load(prefix + '/node_types.npy')
    num_classes = 4

    
    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz').A
    ADJ = ADJ[:4057, 4057:18385]
    ADJ = torch.FloatTensor(ADJ)
    ADJ = F.normalize(ADJ, dim=1, p=1)

    
    nei_p = np.load(prefix + "/nei_p.npy", allow_pickle=True)
    nei_p = [th.LongTensor(i) for i in nei_p]
    # NS = [nei_p]

    
    nei_c = np.load(prefix + "/nei_c.npy", allow_pickle=True)
    nei_c = [th.LongTensor(i) for i in nei_c]
    nei_t = np.load(prefix + "/nei_t.npy", allow_pickle=True)
    nei_t = [th.LongTensor(i) for i in nei_t]
    NS = [nei_p, nei_t, nei_c]

    
    APA = scipy.sparse.load_npz(prefix + '/apa.npz').A
    APA = torch.from_numpy(APA).type(torch.FloatTensor)
    APA = APA - torch.diag_embed(torch.diag(APA))
    APA = F.normalize(APA, dim=1, p=1)

    APTPA = scipy.sparse.load_npz(prefix + '/aptpa.npz').A
    APTPA = torch.from_numpy(APTPA).type(torch.FloatTensor)
    APTPA = APTPA - torch.diag_embed(torch.diag(APTPA))
    APTPA = F.normalize(APTPA, dim=1, p=2)

    APCPA = scipy.sparse.load_npz(prefix + '/apvpa.npz').A
    APCPA = torch.from_numpy(APCPA).type(torch.FloatTensor)
    APCPA = APCPA - torch.diag_embed(torch.diag(APCPA))
    APCPA = F.normalize(APCPA, dim=1, p=2)

    MG = [APA, APTPA, APCPA]
    #MG = [APCPA]
    return ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask,pos





def load_YELP_data(prefix=None):
    prefix = _dataset_dir('YELP', prefix)
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_b.npz').toarray()#b
    features_1 = scipy.sparse.load_npz(prefix + '/features_1_u.npz').toarray()#u
    features_2 = scipy.sparse.load_npz(prefix + '/features_2_s.npz').toarray()#s
    features_3 = scipy.sparse.load_npz(prefix + '/features_3_l.npz').toarray()#l
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features_3 = torch.FloatTensor(features_3)

    features_0 = F.normalize(features_0, dim=1, p=2)
    # features_1 = F.normalize(features_1, dim=1, p=1)
    # features_2 = F.normalize(features_2, dim=1, p=1)
    # features_3 = F.normalize(features_3, dim=1, p=1)

    features = [features_0, features_1, features_2, features_3]

    pos = sp.load_npz(prefix + '/pos.npz')
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npy',allow_pickle=True)
    train_val_test_idx = train_val_test_idx.item()
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    type_mask = np.load(prefix + '/node_types.npy')
    num_classes = 3

    
    ADJ = scipy.sparse.load_npz(prefix + '/adjM.npz').A

    ADJ = ADJ[:2614, 2614:]
    ADJ = torch.FloatTensor(ADJ)
    ADJ = F.normalize(ADJ, dim=1, p=0)

    
    nei_u = np.load(prefix + "/nei_u.npy", allow_pickle=True)
    # print(nei_u.shape)
    nei_u = [th.LongTensor(i) for i in nei_u]
    # NS = [nei_p]

    
    nei_s = np.load(prefix + "/nei_s.npy", allow_pickle=True)
    nei_s = nei_s.astype(float)

    nei_s = [th.LongTensor(i) for i in nei_s]
    nei_l = np.load(prefix + "/nei_l.npy", allow_pickle=True)
    nei_l = nei_l.astype(float)
    nei_l = [th.LongTensor(i) for i in nei_l]
    NS = [nei_u, nei_s, nei_l]

    
    BUB = scipy.sparse.load_npz(prefix + '/adj_bub.npz').A
    BUB = torch.from_numpy(BUB).type(torch.FloatTensor)
    BUB = F.normalize(BUB, dim=1, p=2)

    BSB = scipy.sparse.load_npz(prefix + '/adj_bsb.npz').A
    BSB = torch.from_numpy(BSB).type(torch.FloatTensor)
    BSB = F.normalize( BSB, dim=1, p=2)

    BLB = scipy.sparse.load_npz(prefix + '/adj_blb.npz').A
    BLB = torch.from_numpy(BLB).type(torch.FloatTensor)
    BLB = F.normalize(BLB, dim=1, p=2)

    MG = [BUB, BSB,BLB]

    return ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask,pos


def load_Aminer_data(prefix=None, ratio=20):
    prefix = _dataset_dir('Aminer', prefix)

    features_0 = np.load(prefix + '/features_0.npy')
    features_1 = np.load(prefix + '/features_1.npy')
    features_2 = np.load(prefix + '/features_2.npy')

    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)

    features_0 = F.normalize(features_0, dim=1, p=2)
    features_1 = F.normalize(features_1, dim=1, p=2)
    features_2 = F.normalize(features_2, dim=1, p=2)

    features = [features_0, features_1, features_2]

    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)

    train_idx = np.load(prefix + f'/train_{ratio}.npy')
    val_idx = np.load(prefix + f'/val_{ratio}.npy')
    test_idx = np.load(prefix + f'/test_{ratio}.npy')

    type_mask = np.zeros(len(labels) + features_1.shape[0] + features_2.shape[0], dtype=np.int64)
    type_mask[:len(labels)] = 0  # Paper: 0
    type_mask[len(labels):len(labels)+features_1.shape[0]] = 1  # Author: 1
    type_mask[len(labels)+features_1.shape[0]:] = 2  # Reference: 2

    num_classes = 4

    pa = np.loadtxt(prefix + '/pa.txt', dtype=np.int64)
    pr = np.loadtxt(prefix + '/pr.txt', dtype=np.int64)

    n_paper = features_0.shape[0]
    n_author = features_1.shape[0]
    n_ref = features_2.shape[0]
    n_total = n_paper + n_author + n_ref

    ADJ = scipy.sparse.lil_matrix((n_total, n_total))
    pa_pairs = pa[:, :2] - 1
    pr_pairs = pr[:, :2] - 1
    for i, j in pa_pairs:
        if 0 <= i < n_total and 0 <= j < n_total:
            ADJ[i, j + n_paper] = 1
            ADJ[j + n_paper, i] = 1
    for i, j in pr_pairs:
        if 0 <= i < n_total and 0 <= j < n_total:
            ADJ[i, j + n_paper + n_author] = 1
            ADJ[j + n_paper + n_author, i] = 1
    ADJ = ADJ.tocsr()
    ADJ = ADJ[:n_paper, n_paper:]
    ADJ = torch.FloatTensor(ADJ.toarray())
    ADJ = F.normalize(ADJ, dim=1, p=1)

    nei_a = [[] for _ in range(n_paper)]
    for i, j in pa_pairs:
        if i < n_paper and j < n_author:
            nei_a[i].append(j)
    nei_r = [[] for _ in range(n_paper)]
    for i, j in pr_pairs:
        if i < n_paper and j < n_ref:
            nei_r[i].append(j)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_r = [th.LongTensor(i) for i in nei_r]
    NS = [nei_a, nei_r]

    PAP = scipy.sparse.lil_matrix((n_total, n_total))
    for i, j in pa_pairs:
        if i < n_paper and j < n_author:
            p1 = i
            p2 = i
            PAP[p1, p2] = 1
    for i, j in pa_pairs:
        if i < n_paper and j < n_author:
            p1 = i
            a = j + n_paper
            PAP[a, p1] = 1
            PAP[a, p1] = 1
    PAP = PAP.tocsr()
    PAP_coo = PAP.tocoo()
    PAP_dense = np.zeros((n_paper, n_paper))
    for idx in range(len(PAP_coo.row)):
        r, c = PAP_coo.row[idx], PAP_coo.col[idx]
        if r < n_paper and c < n_paper:
            PAP_dense[r, c] = 1
    PAP = torch.FloatTensor(PAP_dense)
    PAP = F.normalize(PAP, dim=1, p=2)

    PRP = scipy.sparse.lil_matrix((n_total, n_total))
    for i, j in pr_pairs:
        if i < n_paper and j < n_ref:
            p1 = i
            p2 = i
            PRP[p1, p2] = 1
    for i, j in pr_pairs:
        if i < n_paper and j < n_ref:
            p1 = i
            r = j + n_paper + n_author
            PRP[r, p1] = 1
            PRP[r, p1] = 1
    PRP = PRP.tocsr()
    PRP_coo = PRP.tocoo()
    PRP_dense = np.zeros((n_paper, n_paper))
    for idx in range(len(PRP_coo.row)):
        r, c = PRP_coo.row[idx], PRP_coo.col[idx]
        if r < n_paper and c < n_paper:
            PRP_dense[r, c] = 1
    PRP = torch.FloatTensor(PRP_dense)
    PRP = F.normalize(PRP, dim=1, p=2)

    MG = [PAP, PRP]

    return ADJ, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask


def get_freebase_neighbor_features(mam, mdm, mwm, n_movie, n_components=64):

    one_hot = np.eye(n_movie)

    deg_d = mdm.sum(axis=1).reshape(-1, 1) + 1e-8
    deg_a = mam.sum(axis=1).reshape(-1, 1) + 1e-8
    deg_w = mwm.sum(axis=1).reshape(-1, 1) + 1e-8

    feat_d = mdm.T @ one_hot / deg_d
    feat_a = mam.T @ one_hot / deg_a
    feat_w = mwm.T @ one_hot / deg_w

    features = np.concatenate([feat_d, feat_a, feat_w], axis=1)

    if n_components and n_components < features.shape[1]:
        svd = TruncatedSVD(n_components=n_components, random_state=164)
        features = svd.fit_transform(features)

    return features


def load_FreeBase_data(prefix=None, ratio=20):
    prefix = _dataset_dir('freebase', prefix)

    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    num_classes = len(np.unique(labels))

    train_idx = np.load(prefix + f'/train_{ratio}.npy')
    val_idx = np.load(prefix + f'/val_{ratio}.npy')
    test_idx = np.load(prefix + f'/test_{ratio}.npy')

    n_movie = len(labels)

    MAM = scipy.sparse.load_npz(prefix + '/mam.npz').toarray()
    MDM = scipy.sparse.load_npz(prefix + '/mdm.npz').toarray()
    MWM = scipy.sparse.load_npz(prefix + '/mwm.npz').toarray()

    features_0 = get_freebase_neighbor_features(MAM, MDM, MWM, n_movie, n_components=64)
    features_0 = torch.FloatTensor(features_0)
    features_0 = F.normalize(features_0, dim=1, p=2)
    features = [features_0]

    ADJ_movie = MAM
    ADJ_movie = torch.FloatTensor(ADJ_movie)
    ADJ_movie = F.normalize(ADJ_movie, dim=1, p=1)

    nei_d = np.load(prefix + "/nei_d.npy", allow_pickle=True)
    nei_a = np.load(prefix + "/nei_a.npy", allow_pickle=True)
    nei_w = np.load(prefix + "/nei_w.npy", allow_pickle=True)
    nei_d = [torch.LongTensor(i) for i in nei_d]
    nei_a = [torch.LongTensor(i) for i in nei_a]
    nei_w = [torch.LongTensor(i) for i in nei_w]
    NS = [nei_d, nei_a, nei_w]

    MAM = torch.FloatTensor(MAM)
    MDM = torch.FloatTensor(MDM)
    MWM = torch.FloatTensor(MWM)

    MAM = F.normalize(MAM, dim=1, p=2)
    MDM = F.normalize(MDM, dim=1, p=2)
    MWM = F.normalize(MWM, dim=1, p=2)

    MG = [MAM, MDM, MWM]

    pos = scipy.sparse.load_npz(prefix + '/pos.npz')

    type_mask = np.zeros(n_movie, dtype=np.int64)

    return ADJ_movie, NS, MG, features, labels, num_classes, train_idx, val_idx, test_idx, type_mask, pos
