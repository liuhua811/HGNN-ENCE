import numpy as np
import scipy.sparse as sp

# 读取 .npy 文件
npy_file_path = 'MDMDM.npy'
dense_data = np.load(npy_file_path)

# 将其转换为稀疏矩阵（如 COO 格式）
sparse_data = sp.coo_matrix(dense_data)

# 保存为 .npz 文件
npz_file_path = 'MDMDM.npz'
sp.save_npz(npz_file_path, sparse_data)

print(f"Sparse data from {npy_file_path} has been saved to {npz_file_path}")
