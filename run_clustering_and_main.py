import subprocess

# 定义一个函数来调用聚类.py和main.py
def run_clustering_and_main(n_clusters):
    # Step 1: 调用聚类.py 文件，传递聚类数量
    print(f"开始执行聚类，聚类数: {n_clusters}")

    # 使用 subprocess 执行聚类.py，并将聚类数量作为参数传入
    result = subprocess.run(['python', 'clustering.py', str(n_clusters)], capture_output=True, text=True, encoding='utf-8')  # 添加 encoding='utf-8'

    # 输出聚类结果到控制台
    print(f"聚类结果（聚类数={n_clusters}）:\n", result.stdout)

    # Step 2: 调用 main.py
    print("开始执行 main.py")
    subprocess.run(['python', 'main.py'])  # 调用 main.py，假设它在聚类后自动运行



# 从 3 到 100 进行循环
for n_clusters in range(3, 101):
    run_clustering_and_main(n_clusters)
