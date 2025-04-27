import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# Step 1: 加载特征矩阵
npz_file = np.load("D:/桌面文件/模型/复现数据集/ACM_processed/features_0_p.npz")
data = npz_file['data']
indices = npz_file['indices']
indptr = npz_file['indptr']
shape = tuple(npz_file['shape'])  # 矩阵形状
sparse_matrix = csr_matrix((data, indices, indptr), shape=shape)

print("特征矩阵加载完成")
print("矩阵形状:", sparse_matrix.shape)

# Step 2: 降维处理
print("降维处理...")
svd = TruncatedSVD(n_components=50, random_state=42)  # 将特征维度降到50
reduced_data = svd.fit_transform(sparse_matrix)
print("降维完成，降维后形状:", reduced_data.shape)

# Step 3: Agglomerative Clustering 聚类
print("执行 Agglomerative Clustering 聚类...")
agg_clustering = AgglomerativeClustering(n_clusters=70)  # 设置聚类数
labels = agg_clustering.fit_predict(reduced_data)
print("聚类完成")

from sklearn.cluster import KMeans

# Step 3: 使用 K-means 聚类
# print("执行 K-means 聚类...")
# kmeans = KMeans(n_clusters=70, random_state=42)  # 设置聚类数
# labels = kmeans.fit_predict(reduced_data)
# print("聚类完成")


# Step 4: 使用PCA将降维后的数据映射到2D空间以便可视化
print("使用PCA将数据映射到2D空间...")
pca = PCA(n_components=2)
reduced_data_2d = pca.fit_transform(reduced_data)

# # Step 5: 绘制聚类图
# plt.figure(figsize=(8, 6))
# plt.scatter(reduced_data_2d[:, 0], reduced_data_2d[:, 1], c=labels, cmap='viridis', marker='o')
# plt.title('Agglomerative Clustering Results')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.colorbar(label='Cluster Label')
# plt.show()

# Step 6: 创建邻接矩阵
adjacency_matrix = np.zeros((shape[0], shape[0]), dtype=np.int8)  # 初始化为全零矩阵

print("构建邻接矩阵...")

# 对每一类节点进行遍历，并在邻接矩阵中建立连接
for cluster_label in np.unique(labels):
    # 获取当前类别的所有节点索引
    nodes_in_cluster = np.where(labels == cluster_label)[0]

    # 将同一类中的点两两相连
    for i in nodes_in_cluster:
        for j in nodes_in_cluster:
            if i != j:  # 避免自环
                adjacency_matrix[i, j] = 1
np.fill_diagonal(adjacency_matrix, 1)

print("邻接矩阵构建完成")

# Step 7: 保存邻接矩阵为稀疏矩阵
output_sparse_path = "D:/桌面文件/模型/复现数据集/ACM_processed/P_P.npz"
sparse_adjacency_matrix = csr_matrix(adjacency_matrix)
from scipy.sparse import save_npz
save_npz(output_sparse_path, sparse_adjacency_matrix)

print(f"稀疏邻接矩阵已保存到 {output_sparse_path}")

# Step 8: 检查邻接矩阵的稀疏性
non_zero_elements = np.count_nonzero(adjacency_matrix)
total_elements = adjacency_matrix.size
print(f"邻接矩阵稀疏性: {100 * (1 - non_zero_elements / total_elements):.2f}%")

##DBLP


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# # Step 1: 加载特征矩阵
# npz_file = np.load("D:/桌面文件/模型/复现数据集/DBLP_processed/features_0_A.npz")
# data = npz_file['data']
# indices = npz_file['indices']
# indptr = npz_file['indptr']
# shape = tuple(npz_file['shape'])  # 矩阵形状
# sparse_matrix = csr_matrix((data, indices, indptr), shape=shape)
#
# print("特征矩阵加载完成")
# print("矩阵形状:", sparse_matrix.shape)
#
# # Step 2: 降维处理
# print("降维处理...")
# svd = TruncatedSVD(n_components=50, random_state=42)  # 将特征维度降到50
# reduced_data = svd.fit_transform(sparse_matrix)
# print("降维完成，降维后形状:", reduced_data.shape)
#
# # Step 3: Agglomerative Clustering 聚类
# print("执行 Agglomerative Clustering 聚类...")
# agg_clustering = AgglomerativeClustering(n_clusters=44)  # 设置聚类数
# labels = agg_clustering.fit_predict(reduced_data)
# print("聚类完成")
#
# # # Step 4: 使用PCA将降维后的数据映射到2D空间以便可视化
# # print("使用PCA将数据映射到2D空间...")
# # pca = PCA(n_components=2)
# # reduced_data_2d = pca.fit_transform(reduced_data)
# #
# # # Step 5: 绘制聚类图
# # plt.figure(figsize=(8, 6))
# # plt.scatter(reduced_data_2d[:, 0], reduced_data_2d[:, 1], c=labels, cmap='viridis', marker='o')
# # plt.title('Agglomerative Clustering Results')
# # plt.xlabel('PCA Component 1')
# # plt.ylabel('PCA Component 2')
# # plt.colorbar(label='Cluster Label')
# # plt.show()
#
# # Step 6: 创建邻接矩阵
# adjacency_matrix = np.zeros((shape[0], shape[0]), dtype=np.int8)  # 初始化为全零矩阵
#
# print("构建邻接矩阵...")
#
# # 对每一类节点进行遍历，并在邻接矩阵中建立连接
# for cluster_label in np.unique(labels):
#     # 获取当前类别的所有节点索引
#     nodes_in_cluster = np.where(labels == cluster_label)[0]
#
#     # 将同一类中的点两两相连
#     for i in nodes_in_cluster:
#         for j in nodes_in_cluster:
#             if i != j:  # 避免自环
#                 adjacency_matrix[i, j] = 1
#
# np.fill_diagonal(adjacency_matrix, 1)
#
# print("邻接矩阵构建完成")
#
# # Step 7: 保存邻接矩阵为稀疏矩阵
# output_sparse_path = "D:/桌面文件/模型/复现数据集/DBLP_processed/A_A.npz"
# sparse_adjacency_matrix = csr_matrix(adjacency_matrix)
# from scipy.sparse import save_npz
# save_npz(output_sparse_path, sparse_adjacency_matrix)
#
# print(f"稀疏邻接矩阵已保存到 {output_sparse_path}")
#
# # Step 8: 检查邻接矩阵的稀疏性
# non_zero_elements = np.count_nonzero(adjacency_matrix)
# total_elements = adjacency_matrix.size
# print(f"邻接矩阵稀疏性: {100 * (1 - non_zero_elements / total_elements):.2f}%")





###IMDB
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix
# from sklearn.decomposition import TruncatedSVD
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.decomposition import PCA
#
# # Step 1: 加载特征矩阵
# npz_file = np.load("D:/桌面文件/模型/复现数据集/IMDB_processed/features_0_M.npz")
# data = npz_file['data']
# indices = npz_file['indices']
# indptr = npz_file['indptr']
# shape = tuple(npz_file['shape'])  # 矩阵形状
# sparse_matrix = csr_matrix((data, indices, indptr), shape=shape)
#
# print("特征矩阵加载完成")
# print("矩阵形状:", sparse_matrix.shape)
#
# # Step 2: 降维处理
# print("降维处理...")
# svd = TruncatedSVD(n_components=50, random_state=42)  # 将特征维度降到50
# reduced_data = svd.fit_transform(sparse_matrix)
# print("降维完成，降维后形状:", reduced_data.shape)
#
# # Step 3: Agglomerative Clustering 聚类
# print("执行 Agglomerative Clustering 聚类...")
# agg_clustering = AgglomerativeClustering(n_clusters=52)  # 设置聚类数
# labels = agg_clustering.fit_predict(reduced_data)
# print("聚类完成")
#
# # # Step 4: 使用PCA将降维后的数据映射到2D空间以便可视化
# # print("使用PCA将数据映射到2D空间...")
# # pca = PCA(n_components=2)
# # reduced_data_2d = pca.fit_transform(reduced_data)
# #
# # # Step 5: 绘制聚类图
# # plt.figure(figsize=(8, 6))
# # plt.scatter(reduced_data_2d[:, 0], reduced_data_2d[:, 1], c=labels, cmap='viridis', marker='o')
# # plt.title('Agglomerative Clustering Results')
# # plt.xlabel('PCA Component 1')
# # plt.ylabel('PCA Component 2')
# # plt.colorbar(label='Cluster Label')
# # plt.show()
#
# # Step 6: 创建邻接矩阵
# adjacency_matrix = np.zeros((shape[0], shape[0]), dtype=np.int8)  # 初始化为全零矩阵
#
# print("构建邻接矩阵...")
#
# # 对每一类节点进行遍历，并在邻接矩阵中建立连接
# for cluster_label in np.unique(labels):
#     # 获取当前类别的所有节点索引
#     nodes_in_cluster = np.where(labels == cluster_label)[0]
#
#     # 将同一类中的点两两相连
#     for i in nodes_in_cluster:
#         for j in nodes_in_cluster:
#             if i != j:  # 避免自环
#                 adjacency_matrix[i, j] = 1
#
# # 保证每个节点都有自环
# np.fill_diagonal(adjacency_matrix, 1)
#
# print("邻接矩阵构建完成")
#
# # Step 7: 保存邻接矩阵为稀疏矩阵
# output_sparse_path = "D:/桌面文件/模型/复现数据集/IMDB_processed/M_M.npz"
# sparse_adjacency_matrix = csr_matrix(adjacency_matrix)
# from scipy.sparse import save_npz
# save_npz(output_sparse_path, sparse_adjacency_matrix)
#
# print(f"稀疏邻接矩阵已保存到 {output_sparse_path}")
#
# # Step 8: 检查邻接矩阵的稀疏性
# non_zero_elements = np.count_nonzero(adjacency_matrix)
# total_elements = adjacency_matrix.size
#
# print(f"邻接矩阵稀疏性: {100 * (1 - non_zero_elements / total_elements):.2f}%")




# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix
# from sklearn.decomposition import TruncatedSVD
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.decomposition import PCA
#
# # Step 1: 加载特征矩阵
# npz_file = np.load("D:/桌面文件/模型/复现数据集/DBLP_processed/features_0_A.npz")
# data = npz_file['data']
# indices = npz_file['indices']
# indptr = npz_file['indptr']
# shape = tuple(npz_file['shape'])  # 矩阵形状
# sparse_matrix = csr_matrix((data, indices, indptr), shape=shape)
#
# print("特征矩阵加载完成")
# print("矩阵形状:", sparse_matrix.shape)
#
# # Step 2: 降维处理
# print("降维处理...")
# svd = TruncatedSVD(n_components=50, random_state=42)  # 将特征维度降到50
# reduced_data = svd.fit_transform(sparse_matrix)
# print("降维完成，降维后形状:", reduced_data.shape)
#
# # Step 3: Agglomerative Clustering 聚类
# print("执行 Agglomerative Clustering 聚类...")
# agg_clustering = AgglomerativeClustering(n_clusters=44)  # 设置聚类数
# labels = agg_clustering.fit_predict(reduced_data)
# print("聚类完成")
#
# # # Step 4: 使用PCA将降维后的数据映射到2D空间以便可视化
# # print("使用PCA将数据映射到2D空间...")
# # pca = PCA(n_components=2)
# # reduced_data_2d = pca.fit_transform(reduced_data)
# #
# # # Step 5: 绘制聚类图
# # plt.figure(figsize=(8, 6))
# # plt.scatter(reduced_data_2d[:, 0], reduced_data_2d[:, 1], c=labels, cmap='viridis', marker='o')
# # plt.title('Agglomerative Clustering Results')
# # plt.xlabel('PCA Component 1')
# # plt.ylabel('PCA Component 2')
# # plt.colorbar(label='Cluster Label')
# # plt.show()
#
# # Step 6: 创建邻接矩阵
# adjacency_matrix = np.zeros((shape[0], shape[0]), dtype=np.int8)  # 初始化为全零矩阵
#
# print("构建邻接矩阵...")
#
# # 对每一类节点进行遍历，并在邻接矩阵中建立连接
# for cluster_label in np.unique(labels):
#     # 获取当前类别的所有节点索引
#     nodes_in_cluster = np.where(labels == cluster_label)[0]
#
#     # 将同一类中的点两两相连
#     for i in nodes_in_cluster:
#         for j in nodes_in_cluster:
#             if i != j:  # 避免自环
#                 adjacency_matrix[i, j] = 1
#
# np.fill_diagonal(adjacency_matrix, 1)
#
# print("邻接矩阵构建完成")
#
# # Step 7: 保存邻接矩阵为稀疏矩阵
# output_sparse_path = "D:/桌面文件/模型/复现数据集/DBLP_processed/A_A.npz"
# sparse_adjacency_matrix = csr_matrix(adjacency_matrix)
# from scipy.sparse import save_npz
# save_npz(output_sparse_path, sparse_adjacency_matrix)
#
# print(f"稀疏邻接矩阵已保存到 {output_sparse_path}")
#
# # Step 8: 检查邻接矩阵的稀疏性
# non_zero_elements = np.count_nonzero(adjacency_matrix)
# total_elements = adjacency_matrix.size
# print(f"邻接矩阵稀疏性: {100 * (1 - non_zero_elements / total_elements):.2f}%")

