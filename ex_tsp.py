import pulp as pp
import math
import networkx as nx
import matplotlib.pyplot as plt

#データの読み込み
directory = "/Users/yoshidayuto/Desktop/00_研究/40_コード集/10_Training_Takanaga/dataset/tsp_dataset_20.csv"
p_list = [[0,0]]
with open(directory, 'r') as file:
    name1 = file.readline()
    cus_num = int(file.readline())
    name2 = file.readline()
    for i in range(cus_num):
        p_list.append(list(map(float,file.readline().split(","))))

n = len(p_list)

#モデルの構築
model_name = "tsp_model"
model = pp.LpProblem(name = model_name, sense = 1)

#決定変数x_ijを管理する辞書xと定数距離d_ijの辞書d
x,d,u = {},{},{}
for i in range(n):
    u[i] = pp.LpVariable(name = 'u_{0}'.format(i), cat = 'Integer')
    for j in range(n):
        if i != j:
            d[i,j] = round(math.sqrt(abs(p_list[i][0] - p_list[j][0])**2 + abs(p_list[i][1] - p_list[j][1])**2),4)
            x[i,j] = pp.LpVariable(name = 'x_{0}_{1}'.format(i,j), cat = 'Binary')
#目的関数
model.setObjective(obj = pp.lpSum(d[i,j] * x[i,j] for i in range(n) for j in range(n) if i != j))
#制約式(1)
for j in range(n):
    model += pp.lpSum(x[i,j] for i in range(n) if i != j) == 1
#制約式(2)
for i in range(n):
    model += pp.lpSum(x[i,j] for j in range(n) if i != j) == 1
#制約式(3)
model += u[0] == 1
#制約式(4)
for i in range(1,n):
    model += u[i] <= n
    model += u[i] >= 2
#制約式(5)
for i in range(1,n):
    for j in range(1,n):
        if i != j:
            model += u[i] - u[j] + (n-1) * x[i,j] <= n-2

model.writeLP(filename = "{0}.lp".format(model_name))
#model.solve()

#グラフネットワークに出力
G = nx.DiGraph()
for i in range(n):
    G.add_node((p_list[i][0],p_list[i][1]))

for i in range(n):
    for j in range(n):
        if i != j:
            if x[i,j].value() == 1:
                G.add_edge((p_list[i][0],p_list[i][1]),(p_list[j][0],p_list[j][1]))

pos = {n: (n[0], n[1]) for n in G.nodes()}
node_size = [30 for i in range(n)]
nx.draw_networkx_nodes(G, pos,node_size = node_size)
nx.draw_networkx_edges(G, pos)

#出力するグラフの軸ラベルとかの設定
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.tick_params(labelbottom=True,labelleft=True)
plt.xticks([n for n in range(-10,11,2)])
plt.yticks([n for n in range(-10,11,2)])
plt.savefig('tsp.png')

print("hello")
