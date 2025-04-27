import numpy as np
import scipy
import torch
import dgl
import scipy.sparse as sp
import torch.nn.functional as F
from scipy.sparse import csr_matrix


def load_ACM_data(prefix=r'D:\桌面文件\模型\复现数据集\ACM_processed'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_p.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1_a.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2_s.npz').toarray()
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features = features_0
    # features_dim =  [features_0.shape[1], features_1.shape[1], features_2.shape[1]]

    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引

    PP = scipy.sparse.load_npz("D:/桌面文件/模型/复现数据集/ACM_processed/P_P.npz")
    PP1 = scipy.sparse.load_npz("D:/桌面文件/模型/复现数据集/ACM_processed/P_P1.npz")

    # 获取数据
    PSP = scipy.sparse.load_npz(prefix + '/PSP.npz')
    PAP = scipy.sparse.load_npz(prefix + '/PAP.npz')
    #存储节点、边及其特征
    g1 = dgl.DGLGraph(PSP)
    g2 = dgl.DGLGraph(PAP)
    g3 = dgl.DGLGraph(PP)
    g4 = dgl.DGLGraph(PP1)

    g = [g1,g2,g3,g4]

    labels = torch.LongTensor(labels)
    num_classes = 3
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    adj = torch.LongTensor(sp.load_npz(prefix +"/adjM.npz").toarray())
    PA = (adj[:4019, 4019:4019 + 7167] > 0) * 1
    PS = (adj[:4019, 4019 + 7167:] > 0) * 1
    NS = [PA,PS]

    return g, features,NS, labels, num_classes, train_idx, val_idx, test_idx
    # return g, features,NS, labels, num_classes, train_idx, val_idx, test_idx,features_dim


def load_DBLP_data(prefix=r'D:\桌面文件\模型\复现数据集\DBLP_processed'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_A.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1_P.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2_T.npz').toarray()
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features = features_0
    # features_dim =  [features_0.shape[1], features_1.shape[1], features_2.shape[1]]

    labels = np.load(prefix + '/labels.npy')#加载标签，4019
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')#加载训练集，验证集，测试集的索引
    AA = scipy.sparse.load_npz("D:/桌面文件/模型/复现数据集/DBLP_processed/A_A.npz")
    AA1 = scipy.sparse.load_npz("D:/桌面文件/模型/复现数据集/DBLP_processed/A_A1.npz")

    # 获取数据
    apa = scipy.sparse.load_npz(prefix + '/apa.npz')
    apcpa = scipy.sparse.load_npz(prefix + '/apcpa.npz')
    aptpa = scipy.sparse.load_npz(prefix + '/aptpa.npz')
    g1 = dgl.DGLGraph(apa)      #存储节点、边及其特征
    g2 = dgl.DGLGraph(apcpa)
    g3 = dgl.DGLGraph(aptpa)
    g4 = dgl.DGLGraph(AA)
    g5 = dgl.DGLGraph(AA1)
    g = [g1,g2,g3,g4,g5]

    # g = [g5]

    labels = torch.LongTensor(labels)
    num_classes = 4
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    adj = torch.LongTensor(sp.load_npz(prefix +"/adjM.npz").toarray())

    NS = []

    return g, features,NS, labels, num_classes, train_idx, val_idx, test_idx

def load_imdb(prefix=r"D:\桌面文件\模型\复现数据集\IMDB_processed"):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_M.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1_D.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2_A.npz').toarray()
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features = features_0

    #标签 训练集，验证集，测试集 分类数量
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    train_idx = torch.from_numpy(train_idx).to(torch.int64)
    val_idx = torch.from_numpy(val_idx).to(torch.int64)
    test_idx = torch.from_numpy(test_idx).to(torch.int64)
    num_classes = 3

    adj = torch.LongTensor(sp.load_npz("./acm/adjM.npz").toarray())
    MD = (adj[:4278, 4278:4278 + 2081] > 0) * 1
    MA = (adj[:4278, 4278 + 2081:] > 0) * 1
    NS = []

    MAM = sp.load_npz(prefix + '/MAM.npz')
    MDM = scipy.sparse.load_npz(prefix + '/MDM.npz')
    MM = scipy.sparse.load_npz(prefix +"/M_M.npz")
    MM1 = scipy.sparse.load_npz(prefix +"/M_M1.npz")
    g1 = dgl.DGLGraph(MAM)  # 存储节点、边及其特征
    g2 = dgl.DGLGraph(MDM)
    g3 = dgl.DGLGraph(MM)
    g4 = dgl.DGLGraph(MM1)

    g = [g1,g2,g3,g4]

    return g, features,NS, labels, num_classes, train_idx, val_idx, test_idx

def load_Yelp(prefix=r"D:\桌面文件\模型\复现数据集\4_Yelp"):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_b.npz').toarray()  #B:2614
    features_1 = scipy.sparse.load_npz(prefix + '/features_1_u.npz').toarray()  #U:1286
    features_2 = scipy.sparse.load_npz(prefix + '/features_2_s.npz').toarray()  #S:4
    features_3 = scipy.sparse.load_npz(prefix + '/features_3_l.npz').toarray()  #L:9
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features_3 = torch.FloatTensor(features_3)
    features = features_0

    #标签 训练集，验证集，测试集 分类数量
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npy', allow_pickle=True)
    train_idx = train_val_test_idx.item()['train_idx'].astype(int)
    val_idx = train_val_test_idx.item()['val_idx']
    test_idx = train_val_test_idx.item()['test_idx']

    train_idx = torch.from_numpy(train_idx).to(torch.int64)
    val_idx = torch.from_numpy(val_idx).to(torch.int64)
    test_idx = torch.from_numpy(test_idx).to(torch.int64)
    num_classes = 3

    BUB = sp.load_npz(prefix + '/adj_bub_one.npz')
    BSB = scipy.sparse.load_npz(prefix + '/adj_bsb_one.npz')
    BLB = scipy.sparse.load_npz(prefix + '/adj_blb_one.npz')
    g1 = dgl.DGLGraph(BUB)  # 存储节点、边及其特征
    g2 = dgl.DGLGraph(BSB)
    g3 = dgl.DGLGraph(BLB)
    g = [g1,g2,g3]

    adj = torch.LongTensor(sp.load_npz(prefix+"/adjM.npz").toarray())
    BU = (adj[:2614, 2614:2614 + 1286] > 0) * 1
    BS = (adj[:2614, 2614 + 1286:2614 + 1286+4] > 0) * 1
    BL = (adj[:2614, 2614+1286+4:2614 + 1286+4+9] > 0) * 1
    NS = [BU, BS]
    return g, features,NS, labels, num_classes, train_idx, val_idx, test_idx

def load_data(prefix=r'D:\桌面文件\模型\阿里复现数据集\三分类'):
    features_0 = scipy.sparse.load_npz(prefix + '/product_feature.npz').toarray()  # features_0为商品特征 features_1为用户特征
    features_1 = scipy.sparse.load_npz(prefix + '/user_feature.npz').toarray()
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features = features_0

    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)

    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    num_classes = 3
    # meta_path
    item_buy_user_item = scipy.sparse.load_npz(prefix + '/item_buy_user_item.npz').toarray()
    item_cart_user_item = scipy.sparse.load_npz(prefix + '/item_cart_user_item.npz').toarray()
    item_pav_user_item = scipy.sparse.load_npz(prefix + '/item_fav_user_item.npz').toarray()
    item_pv_user_item = scipy.sparse.load_npz(prefix + '/item_pv_user_item.npz').toarray()
    # i_i = scipy.sparse.load_npz(prefix + '/i_i.npz')

    item_buy_user_item = scipy.sparse.load_npz(prefix + '/item_buy_user_item.npz')
    item_cart_user_item = scipy.sparse.load_npz(prefix + '/item_cart_user_item.npz')
    item_pav_user_item = scipy.sparse.load_npz(prefix + '/item_fav_user_item.npz')
    item_pv_user_item = scipy.sparse.load_npz(prefix + '/item_pv_user_item.npz')
    g1 = dgl.DGLGraph(item_buy_user_item)
    g2 = dgl.DGLGraph(item_cart_user_item)
    g3 = dgl.DGLGraph(item_pav_user_item)
    g4 = dgl.DGLGraph(item_pv_user_item)
    # g5 = dgl.DGLGraph(i_i)
    g1 = dgl.add_self_loop(g1)
    g2 = dgl.add_self_loop(g2)
    g3 = dgl.add_self_loop(g3)
    g4 = dgl.add_self_loop(g4)
    # g5 = dgl.add_self_loop(g5)

    g=[g1,g2,g3,g4]
    NS=[]

    return g, features,NS, labels, num_classes, train_idx, val_idx, test_idx


def load_data_dblp(prefix=r'D:\桌面文件\dblp_3'):
    features_0 = scipy.sparse.load_npz(prefix + '/paper_features.npz').toarray()  # 加载特征
    features_0 = torch.FloatTensor(features_0)

    features = features_0

    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)

    train_val_test_idx = np.load(prefix + '/train_val_test_idx1.npz')
    train_idx = torch.LongTensor(train_val_test_idx['train_idx'])
    val_idx = torch.LongTensor(train_val_test_idx['val_idx'])
    test_idx = torch.LongTensor(train_val_test_idx['test_idx'])
    num_classes = labels.max().item() + 1

    # 加载稀疏矩阵并转换为 COO 格式
    P_A_P_important = scipy.sparse.load_npz(prefix + '/P_A_P_important.npz').tocoo()
    P_A_P_ordinary = scipy.sparse.load_npz(prefix + '/P_A_P_ordinary.npz').tocoo()

    # 创建 DGL 图
    g1 = dgl.graph((P_A_P_important.row, P_A_P_important.col))
    g2 = dgl.graph((P_A_P_ordinary.row, P_A_P_ordinary.col))

    # 添加自环边
    g1 = dgl.add_self_loop(g1)
    g2 = dgl.add_self_loop(g2)

    g = [g1, g2]
    NS = []

    return g, features, NS, labels, num_classes, train_idx, val_idx, test_idx
def load_music_data(prefix=r'D:\桌面文件\音乐数据集2.5(双类多关系)\音乐数据集2.5(双类多关系)'):
    features_0 = scipy.sparse.load_npz(prefix + '/song_features.npz').toarray()  # features_0为商品特征 features_1为用户特征
    features_1 = scipy.sparse.load_npz(prefix + '/user_features.npz').toarray()
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features = features_0

    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)

    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    num_classes = 10
    # meta_path

    # i_i = scipy.sparse.load_npz(prefix + '/i_i.npz')

    item_buy_user_item = scipy.sparse.load_npz(prefix + '/song_user_play_song.npz')
    item_cart_user_item = scipy.sparse.load_npz(prefix + '/song_user_download_song.npz')
    item_pav_user_item = scipy.sparse.load_npz(prefix + '/song_user_collect_song.npz')

    g1 = dgl.DGLGraph(item_buy_user_item)
    g2 = dgl.DGLGraph(item_cart_user_item)
    g3 = dgl.DGLGraph(item_pav_user_item)

    # g5 = dgl.DGLGraph(i_i)
    g1 = dgl.add_self_loop(g1)
    g2 = dgl.add_self_loop(g2)
    g3 = dgl.add_self_loop(g3)

    # g5 = dgl.add_self_loop(g5)

    g=[g1,g2,g3]
    NS=[]

    return g, features,NS, labels, num_classes, train_idx, val_idx, test_idx

if __name__ == "__main__":
    load_data()