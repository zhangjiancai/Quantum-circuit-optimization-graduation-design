# config.py

# 配置参数
N_QUBITS = 5           # 量子比特数量
N_MOMENTS = 15         # 电路中的时刻数（时间步）
N_GATE_CLASSES = 4     # 门类数量
N_RULES = 6            # 规则数量
N_STEPS = 10          # 每集的最大步数
EPOCHS = 3            # 训练的迭代次数
STEPS_PER_EPOCH = 10   # 每个迭代中的步数
LEARNING_RATE = 5e-4   # 优化器的学习率
gamma = 0.99          # 折扣因子
clip_epsilon = 0.2     # 截断误差率
batch_size = 2        # 批处理大小
