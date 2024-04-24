import matplotlib.pyplot as plt

# 原始数据
data = """
0.0078
0.40625
0.4765
0.4922
0.5266
0.5479
0.5544
0.5574
0.5598
0.5625
0.5804
0.5854
0.5876
0.588
0.5908
0.5938
0.5999
0.6055
0.6078
0.6104
0.6116
0.6129
0.6132
0.6135
0.6141
0.6143
0.615
0.6153
0.6154
0.6156
0.616
0.616
0.6156
"""

# 将回车转换为逗号，并按逗号分割成列表
data_list = list(map(float, data.strip().split("\n")))

# 绘制折线图
plt.plot(data_list)

# 添加标题和标签
plt.title('Performance analysis diagram')
plt.xlabel('Communication round')
plt.ylabel('Communication round')

# 显示图形
plt.show()
