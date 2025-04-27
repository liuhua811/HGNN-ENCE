# import numpy as np
# import argparse
# from scipy.sparse import csr_matrix
# from sklearn.decomposition import TruncatedSVD
# from sklearn.cluster import AgglomerativeClustering
#
# # 解析命令行参数
# def parse_args():
#     parser = argparse.ArgumentParser(description="聚类数量参数")
#     parser.add_argument('n_clusters', type=int, help="聚类的数量")
#     return parser.parse_args()
#
# # 主函数
# def run_clustering(n_clusters):
#     # Step 1: 加载特征矩阵
#     npz_file = np.load("D:/桌面文件/模型/复现数据集/ACM_processed/features_0_p.npz")
#     data = npz_file['data']
#     indices = npz_file['indices']
#     indptr = npz_file['indptr']
#     shape = tuple(npz_file['shape'])  # 矩阵形状
#     sparse_matrix = csr_matrix((data, indices, indptr), shape=shape)
#
#     print(f"特征矩阵加载完成，矩阵形状: {sparse_matrix.shape}")
#
#     # Step 2: 降维处理
#     print("降维处理...")
#     svd = TruncatedSVD(n_components=50, random_state=42)  # 将特征维度降到50
#     reduced_data = svd.fit_transform(sparse_matrix)
#     print(f"降维完成，降维后形状: {reduced_data.shape}")
#
#     # Step 3: Agglomerative Clustering 聚类
#     print(f"执行 Agglomerative Clustering 聚类... 使用 {n_clusters} 个聚类")
#     agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)  # 设置聚类数
#     labels = agg_clustering.fit_predict(reduced_data)
#     print(f"聚类完成，聚类数量: {n_clusters}")
#
#     # Step 6: 创建邻接矩阵
#     adjacency_matrix = np.zeros((shape[0], shape[0]), dtype=np.int8)  # 初始化为全零矩阵
#
#     print("构建邻接矩阵...")
#
#     # 对每一类节点进行遍历，并在邻接矩阵中建立连接
#     for cluster_label in np.unique(labels):
#         # 获取当前类别的所有节点索引
#         nodes_in_cluster = np.where(labels == cluster_label)[0]
#
#         # 将同一类中的点两两相连
#         for i in nodes_in_cluster:
#             for j in nodes_in_cluster:
#                 if i != j:  # 避免自环
#                     adjacency_matrix[i, j] = 1
#
#     # 保证每个节点都有自环
#     np.fill_diagonal(adjacency_matrix, 1)
#
#     print(f"邻接矩阵构建完成，聚类数 {n_clusters}")
#
#     # Step 7: 保存邻接矩阵为稀疏矩阵
#     output_sparse_path = f"D:/桌面文件/模型/复现数据集/ACM_processed/P_P.npz"
#     sparse_adjacency_matrix = csr_matrix(adjacency_matrix)
#     from scipy.sparse import save_npz
#     save_npz(output_sparse_path, sparse_adjacency_matrix)
#
#     print(f"稀疏邻接矩阵已保存到 {output_sparse_path}")
#
#     # Step 8: 检查邻接矩阵的稀疏性
#     non_zero_elements = np.count_nonzero(adjacency_matrix)
#     total_elements = adjacency_matrix.size
#     print(f"邻接矩阵稀疏性: {100 * (1 - non_zero_elements / total_elements):.2f}%")
#     print("-" * 50)
#
# if __name__ == "__main__":
#     args = parse_args()  # 获取命令行参数
#     n_clusters = args.n_clusters  # 获取聚类数
#     run_clustering(n_clusters)  # 运行聚类





import numpy as np
import argparse
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="聚类数量参数")
    parser.add_argument('n_clusters', type=int, help="聚类的数量")
    return parser.parse_args()

# 主函数
def run_clustering(n_clusters):
    # Step 1: 加载特征矩阵
    npz_file = np.load("D:/桌面文件/模型/复现数据集/4_Yelp/features_0_b.npz")
    data = npz_file['data']
    indices = npz_file['indices']
    indptr = npz_file['indptr']
    shape = tuple(npz_file['shape'])  # 矩阵形状
    sparse_matrix = csr_matrix((data, indices, indptr), shape=shape)

    print(f"特征矩阵加载完成，矩阵形状: {sparse_matrix.shape}")

    # Step 2: 降维处理
    print("降维处理...")
    svd = TruncatedSVD(n_components=50, random_state=42)  # 将特征维度降到50
    reduced_data = svd.fit_transform(sparse_matrix)
    print(f"降维完成，降维后形状: {reduced_data.shape}")

    # Step 3: KMeans 聚类
    print(f"执行 KMeans 聚类... 使用 {n_clusters} 个聚类")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # 设置聚类数
    labels = kmeans.fit_predict(reduced_data)
    print(f"聚类完成，聚类数量: {n_clusters}")

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

    # 保证每个节点都有自环
    np.fill_diagonal(adjacency_matrix, 1)

    print(f"邻接矩阵构建完成，聚类数 {n_clusters}")

    # Step 7: 保存邻接矩阵为稀疏矩阵
    output_sparse_path = f"D:/桌面文件/模型/复现数据集/4_Yelp/b_b.npz"
    sparse_adjacency_matrix = csr_matrix(adjacency_matrix)
    from scipy.sparse import save_npz
    save_npz(output_sparse_path, sparse_adjacency_matrix)

    print(f"稀疏邻接矩阵已保存到 {output_sparse_path}")

    # Step 8: 检查邻接矩阵的稀疏性
    non_zero_elements = np.count_nonzero(adjacency_matrix)
    total_elements = adjacency_matrix.size
    print(f"邻接矩阵稀疏性: {100 * (1 - non_zero_elements / total_elements):.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    args = parse_args()  # 获取命令行参数
    n_clusters = args.n_clusters  # 获取聚类数
    run_clustering(n_clusters)  # 运行聚类
