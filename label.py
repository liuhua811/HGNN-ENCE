import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity

# 加载 labels 和 idx (包括训练集、验证集和测试集)
labels = np.load("D:/桌面文件/模型/复现数据集/ACM_processed/labels.npy")
idx_data = np.load("D:/桌面文件/模型/复现数据集/ACM_processed/train_val_test_idx.npz")

train_idx = idx_data['train_idx']
val_idx = idx_data['val_idx']
test_idx = idx_data['test_idx']
print(f"训练集长度: {len(train_idx)}")
print(f"验证集长度: {len(val_idx)}")
print(f"测试集长度: {len(test_idx)}")

# 提取训练集和验证集的标签
train_labels = labels[train_idx]
val_labels = labels[val_idx]

# 提取所有节点的标签
all_labels = labels  # 所有节点的标签

# 加载节点特征（数据部分）
features_npz = np.load("D:/桌面文件/模型/复现数据集/ACM_processed/features_0_p.npz")
features = csr_matrix((features_npz['data'], features_npz['indices'], features_npz['indptr']), shape=features_npz['shape'])

# 将稀疏矩阵转换为密集矩阵（NumPy数组）
dense_features = features.toarray()

# 计算所有节点特征之间的余弦相似度
cos_sim_matrix = cosine_similarity(dense_features)

# 创建一个 4019 x 4019 的邻接矩阵，初始为全零
num_nodes = 4019
adj_matrix = np.zeros((num_nodes, num_nodes))

# 合并训练集和验证集的索引
all_train_val_idx = np.concatenate([train_idx])
all_train_val_labels = labels[all_train_val_idx]

# 设置相似度阈值，只有当相似度大于该值时才连接节点
threshold = 0.55 # 可以调整阈值

# 遍历训练集和验证集中的所有节点对，如果它们的标签相同且特征相似度高，则连接它们
for i in range(len(all_train_val_idx)):
    for j in range(i + 1, len(all_train_val_idx)):
        # 检查标签是否相同且特征相似度大于阈值
        if all_train_val_labels[i] == all_train_val_labels[j] and cos_sim_matrix[all_train_val_idx[i], all_train_val_idx[j]] > threshold:
            # 在邻接矩阵中对应位置设置为1
            adj_matrix[all_train_val_idx[i], all_train_val_idx[j]] = 1
            adj_matrix[all_train_val_idx[j], all_train_val_idx[i]] = 1  # 因为是无向图

# 添加自环：将对角线元素设置为1
np.fill_diagonal(adj_matrix, 1)

# 将邻接矩阵转换为稀疏矩阵 (CSR 格式)
sparse_adj_matrix = csr_matrix(adj_matrix)

# 计算非零元素的数量
non_zero_elements = sparse_adj_matrix.nnz  # 非零元素数量

# 计算稀疏性：非零元素数占总元素数的比例
total_elements = sparse_adj_matrix.shape[0] * sparse_adj_matrix.shape[1]  # 总元素数量
sparsity = (1 - non_zero_elements / total_elements) * 100  # 稀疏性（百分比）

# 输出非零元素的数量和稀疏性
print(f"非零元素的数量：{non_zero_elements}")
print(f"邻接矩阵的稀疏性：{sparsity:.2f}%")

# 使用scipy的save_npz保存稀疏矩阵
save_npz("D:/桌面文件/模型/复现数据集/ACM_processed/P_P1.npz", sparse_adj_matrix)
print(train_labels)
print("稀疏矩阵已保存为 P_P1.npz")






###DBLP
# import numpy as np
# from scipy.sparse import csr_matrix, save_npz
# from sklearn.metrics.pairwise import cosine_similarity
#
# # 加载 labels 和 idx (包括训练集、验证集和测试集)
# labels = np.load("D:/桌面文件/模型/复现数据集/DBLP_processed/labels.npy")
# idx_data = np.load("D:/桌面文件/模型/复现数据集/DBLP_processed/train_val_test_idx.npz")
#
# train_idx = idx_data['train_idx']
# val_idx = idx_data['val_idx']
# test_idx = idx_data['test_idx']
#
# # 提取训练集和验证集的标签
# train_labels = labels[train_idx]
# val_labels = labels[val_idx]
#
# # 提取所有节点的标签
# all_labels = labels  # 所有节点的标签
#
# # 加载节点特征（数据部分）
# features_npz = np.load("D:/桌面文件/模型/复现数据集/DBLP_processed/features_0_A.npz")
# features = csr_matrix((features_npz['data'], features_npz['indices'], features_npz['indptr']), shape=features_npz['shape'])
#
# # 将稀疏矩阵转换为密集矩阵（NumPy数组）
# dense_features = features.toarray()
#
# # 计算所有节点特征之间的余弦相似度
# cos_sim_matrix = cosine_similarity(dense_features)
#
# print(f"训练集长度: {len(train_idx)}")
# print(f"验证集长度: {len(val_idx)}")
# print(f"测试集长度: {len(test_idx)}")
#
# num_nodes = 4057
# adj_matrix = np.zeros((num_nodes, num_nodes))
#
# # 合并训练集和验证集的索引
# all_train_val_idx = np.concatenate([train_idx])
# all_train_val_labels = labels[all_train_val_idx]
#
# # 设置相似度阈值，只有当相似度大于该值时才连接节点
# threshold = 0.35  # 可以调整阈值
#
# # 遍历训练集和验证集中的所有节点对，如果它们的标签相同且特征相似度高，则连接它们
# for i in range(len(all_train_val_idx)):
#     for j in range(i + 1, len(all_train_val_idx)):
#         # 检查标签是否相同且特征相似度大于阈值
#         if all_train_val_labels[i] == all_train_val_labels[j] and cos_sim_matrix[all_train_val_idx[i], all_train_val_idx[j]] > threshold:
#             # 在邻接矩阵中对应位置设置为1
#             adj_matrix[all_train_val_idx[i], all_train_val_idx[j]] = 1
#             adj_matrix[all_train_val_idx[j], all_train_val_idx[i]] = 1  # 因为是无向图
#
# # 添加自环：将对角线元素设置为1
# np.fill_diagonal(adj_matrix, 1)
#
# # 将邻接矩阵转换为稀疏矩阵 (CSR 格式)
# sparse_adj_matrix = csr_matrix(adj_matrix)
#
# # 计算非零元素的数量
# non_zero_elements = sparse_adj_matrix.nnz  # 非零元素数量
#
# # 计算稀疏性：非零元素数占总元素数的比例
# total_elements = sparse_adj_matrix.shape[0] * sparse_adj_matrix.shape[1]  # 总元素数量
# sparsity = (1 - non_zero_elements / total_elements) * 100  # 稀疏性（百分比）
#
# # 输出非零元素的数量和稀疏性
# print(f"非零元素的数量：{non_zero_elements}")
# print(f"邻接矩阵的稀疏性：{sparsity:.2f}%")
#
# # 使用scipy的save_npz保存稀疏矩阵
# save_npz("D:/桌面文件/模型/复现数据集/DBLP_processed/A_A1.npz", sparse_adj_matrix)
# print(train_labels)
# print("稀疏矩阵已保存为 A_A1.npz")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import numpy as np
# from scipy.sparse import csr_matrix, save_npz
# from sklearn.metrics.pairwise import cosine_similarity
#
# # 加载 labels 和 idx (包括训练集、验证集和测试集)
# labels = np.load("D:/桌面文件/模型/复现数据集/IMDB_processed/labels.npy")
# idx_data = np.load("D:/桌面文件/模型/复现数据集/IMDB_processed/train_val_test_idx.npz")
# print(idx_data)
#
# train_idx = idx_data['train_idx']
# val_idx = idx_data['val_idx']
# test_idx = idx_data['test_idx']
# print(f"训练集长度: {len(train_idx)}")
# print(f"验证集长度: {len(val_idx)}")
# print(f"测试集长度: {len(test_idx)}")
#
# # 提取训练集和验证集的标签
# train_labels = labels[train_idx]
# val_labels = labels[val_idx]
#
# # 提取所有节点的标签
# all_labels = labels  # 所有节点的标签
#
# # 加载节点特征（数据部分）
# features_npz = np.load("D:/桌面文件/模型/复现数据集/IMDB_processed/features_0_M.npz")
# features = csr_matrix((features_npz['data'], features_npz['indices'], features_npz['indptr']), shape=features_npz['shape'])
#
# # 将稀疏矩阵转换为密集矩阵（NumPy数组）
# dense_features = features.toarray()
#
# # 计算所有节点特征之间的余弦相似度
# cos_sim_matrix = cosine_similarity(dense_features)
#
# # 创建一个 4019 x 4019 的邻接矩阵，初始为全零
# num_nodes = 4278
# adj_matrix = np.zeros((num_nodes, num_nodes))
#
# # 合并训练集和验证集的索引
# all_train_val_idx = np.concatenate([train_idx])
# all_train_val_labels = labels[all_train_val_idx]
#
# # 设置相似度阈值，只有当相似度大于该值时才连接节点
# threshold = 0.4  # 可以调整阈值
#
# # 遍历训练集和验证集中的所有节点对，如果它们的标签相同且特征相似度高，则连接它们
# for i in range(len(all_train_val_idx)):
#     for j in range(i + 1, len(all_train_val_idx)):
#         # 检查标签是否相同且特征相似度大于阈值
#         if all_train_val_labels[i] == all_train_val_labels[j] and cos_sim_matrix[all_train_val_idx[i], all_train_val_idx[j]] > threshold:
#             # 在邻接矩阵中对应位置设置为1
#             adj_matrix[all_train_val_idx[i], all_train_val_idx[j]] = 1
#             adj_matrix[all_train_val_idx[j], all_train_val_idx[i]] = 1  # 因为是无向图
#
# # 添加自环：将对角线元素设置为1
# np.fill_diagonal(adj_matrix, 1)
#
# # 将邻接矩阵转换为稀疏矩阵 (CSR 格式)
# sparse_adj_matrix = csr_matrix(adj_matrix)
#
# # 计算非零元素的数量
# non_zero_elements = sparse_adj_matrix.nnz  # 非零元素数量
#
# # 计算稀疏性：非零元素数占总元素数的比例
# total_elements = sparse_adj_matrix.shape[0] * sparse_adj_matrix.shape[1]  # 总元素数量
# sparsity = (1 - non_zero_elements / total_elements) * 100  # 稀疏性（百分比）
#
# # 输出非零元素的数量和稀疏性
# print(f"非零元素的数量：{non_zero_elements}")
# print(f"邻接矩阵的稀疏性：{sparsity:.2f}%")
#
# # 使用scipy的save_npz保存稀疏矩阵
# save_npz("D:/桌面文件/模型/复现数据集/IMDB_processed/M_M1.npz", sparse_adj_matrix)
#
# print("稀疏矩阵已保存为 M_M1.npz")
# print(train_labels)