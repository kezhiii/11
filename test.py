# import torch
#
# # 创建一个包含 NaN 的张量
# x = torch.tensor([[1.0, 2.0, 3.0],
#                   [4.0, float('nan'), 6.0],
#                   [7.0, 8.0, 9.0]])
#
# # 计算 nan_mask
# nan_mask = x.isnan().any(axis=-1)
# print("nan_mask:")
# print(nan_mask)
#
# # 将包含 NaN 的行置为 0
# x[nan_mask] = 0
# print("处理后的 x:")
# print(x)
import torch
import torch.nn as nn

# 定义 swish 激活函数
def swish(x):
    return x * torch.sigmoid(x)

# 假设输入特征 x 的形状为 (batch_size, num_f_maps, sequence_length)
batch_size = 2
num_f_maps = 3
sequence_length = 4
x = torch.randn(batch_size, num_f_maps, sequence_length)

# 假设时间步嵌入向量 time_emb 的形状为 (batch_size, time_emb_dim)
time_emb_dim = 5
time_emb = torch.randn(batch_size, time_emb_dim)

# 定义线性层 self.time_proj
time_proj = nn.Linear(time_emb_dim, num_f_maps)

# 计算经过处理的时间步嵌入向量
time_emb = swish(time_emb)
time_emb = time_proj(time_emb)
time_emb = time_emb[:,:,None]  # 在最后一个维度上增加一个维度

# 进行相加操作
result = x + time_emb

print("输入特征 x 的形状:", x.shape)
print(x)
print("经过处理的时间步嵌入向量的形状:", time_emb.shape)
print(time_emb)
print("相加结果的形状:", result.shape)
print(result)