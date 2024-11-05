import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from math import sqrt
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import simpledialog

plt.rcParams['font.family'] = 'Malgun Gothic'  
plt.rcParams['axes.unicode_minus'] = False

G = nx.Graph()

# 노드의 위치를 지정
pos = {
    '농장문': (507.3915257404467, -102.84458963441489),
    '조형관(도로)': (471.4276838269645, -188.7378505358978),
    '약대 오른(도로)': (541.6651739832625, -228.4589968099731),
    '약대 왼(도로)': (474.0834545721495, -268.90475968019643),
    '신관 옆(도로)': (545.3897905794104, -290.5243723388636),
    '도서관 앞': (480.980185795705, -343.17667276197596),
    '사대 앞(도로)': (555.4242135048919, -347.58801270671773),
    '첨성관(도로)': (688.6599685001538, -362.83070834612454),
    '제4합(도로)': (637.0403558414866, -379.24204829086636),
    '사과대(도로)': (686.4861187117099, -413.13704435338553),
    '경상대': (643.8320831275612, -479.7893447764979),
    '서문': (50, -541),
    '청룡관(도로)': (159, -498),
    '대운동장(도로)': (252, -618),
    '생물학관(도로)': (206, -512),
    '미래융합관과제1과학관사이(도로)': (226, -446),
    '복현회관(도로)': (235, -394),
    '출판동204동(도로)': (317, -381),
    '농대201동(도로)': (323, -318),
    '북문': (354, -287),
    '지도못(도로)': (348, -512),
    '대학원동(도로)': (372, -398),
    '인문대(도로)': (385, -342),
    '쪽문 사거리(도로)': (356.89256198347096, -718.5413223140497),
    '교수 아파트': (341.93388429752054, -714.5743801652894),
    '보람관(도로)': (334.93388429752054, -667.5991735537192),
    '진리관(도로)': (338.9008264462809, -618.9628099173556),
    'IT대 2호관(도로)': (482.08264462809916, -614.5743801652894),
    '수의대(도로)': (540.36363636363626, -669.3429752066118),
    'IT대 1호관(도로)': (577.7272727272725, -598.8388429752067),
    '학생 주차장(도로)': (596.4297520661155, -569.6983471074382),
    '정문': (621.1074380165287, -739.2024793388431),
    '센트럴 파크(도로)': (668.5454545454545, -639.1198347107439),
    'IT대 5호관(도로)': (467.53719008264443, -525.3842975206613),
    '일청담': (478.9834710743801, -481.0702479338844),
    '대운동장테니스장(도로)' : (225, -495),
    '테니스장(도로)' : (358, -464)
}

# 도로 지점과 주요 지점 간의 거리로 엣지를 추가
edges_with_weights = [
    ('경상대', '사과대(도로)', 53.40),
    ('농장문', '조형관(도로)', 63.84),
    ('농장문', '약대 오른(도로)', 115.16),
    ('도서관 앞', '신관 옆(도로)', 87.80),
    ('사과대(도로)', '첨성관(도로)', 73.62),
    ('사과대(도로)', '제4합(도로)', 51.10),
    ('사대 앞(도로)', '신관 옆(도로)', 92.07),
    ('사대 앞(도로)', '제4합(도로)', 81.04),
    ('신관 옆(도로)', '약대 오른(도로)', 62.61),
    ('신관 옆(도로)', '약대 왼(도로)', 71.76),
    ('신관 옆(도로)', '첨성관(도로)', 120.25),
    ('약대 오른(도로)', '약대 왼(도로)', 88.92),
    ('약대 왼(도로)', '조형관(도로)', 95.73),
    ('제4합(도로)', '첨성관(도로)', 46.32),
    ('서문', '청룡관(도로)', 98.49),
    ('미래융합관과제1과학관사이(도로)', '복현회관(도로)', 51.04),
    ('농대201동(도로)', '북문', 41.44),
    ('농대201동(도로)', '출판동204동(도로)', 66.41),
    ('대학원동(도로)', '출판동204동(도로)', 57.57),
    ('농대201동(도로)', '인문대(도로)', 49.04),
    ('북문', '인문대(도로)', 68.60),
    ('대학원동(도로)', '인문대(도로)', 67.27),
    ('대학원동(도로)', '지도못(도로)', 110.16),
    ('복현회관(도로)', '출판동204동(도로)', 64.29),
    ('IT대 1호관(도로)', '수의대(도로)', 69.74),
    ('IT대 1호관(도로)', '학생 주차장(도로)', 42.55),
    ('IT대 1호관(도로)', '정문', 121.34),
    ('IT대 1호관(도로)', '센트럴 파크(도로)', 111.84),
    ('IT대 1호관(도로)', 'IT대 5호관(도로)', 117.18),
    ('IT대 2호관(도로)', '진리관(도로)', 138.34),
    ('IT대 2호관(도로)', '수의대(도로)', 53.97),
    ('IT대 5호관(도로)', '일청담', 49.90),
    ('교수 아파트', '쪽문 사거리(도로)', 45.13),
    ('교수 아파트', '보람관(도로)', 42.98),
    ('보람관(도로)', '진리관(도로)', 43.82),
    ('센트럴 파크(도로)', '학생 주차장(도로)', 95.87),
    ('센트럴 파크(도로)', '정문', 97.26),
    ('수의대(도로)', '정문', 90.76),
    ('일청담', '학생 주차장(도로)', 135.99),
    ('지도못(도로)', '진리관(도로)', 104.14),
    ('대운동장(도로)', '진리관(도로)', 80.90),
    ('IT대 5호관(도로)', '지도못(도로)', 118.30),
    ('경상대', '학생 주차장(도로)', 90.96),
    ('사대 앞(도로)', '일청담', 100.36),
    ('농장문', '북문', 299.66),
    ('도서관 앞', '인문대(도로)', 154.13),
    ('북문', '약대 왼(도로)', 199.68),
    ('북문', '조형관(도로)', 243.36),
    ('대운동장테니스장(도로)', '미래융합관과제1과학관사이(도로)', 51),
    ('대운동장테니스장(도로)', '대운동장(도로)', 151),
    ('청룡관(도로)','대운동장테니스장(도로)', 70),
    ('테니스장(도로)', '대운동장테니스장(도로)', 150),
    ('테니스장(도로)', '대학원동(도로)', 100),
    ('테니스장(도로)', '지도못(도로)', 50),
    ('테니스장(도로)', '일청담', 150)
]

# 그래프에 엣지 추가
for edge in edges_with_weights:
    G.add_edge(edge[0], edge[1], weight=edge[2])

# 다익스트라 알고리즘
def dijkstra(graph, start, finish):
    path = nx.dijkstra_path(graph, start, finish)
    length = nx.dijkstra_path_length(graph, start, finish)
    return path, length

def heuristic(node, goal):
    x1, y1 = pos[node]
    x2, y2 = pos[goal]
    return 2*(sqrt((x2 - x1)**2 + (y2 - y1)**2))  # 휴리스틱에 가중치 2를 곱해 영향력 강화

# IDA* 알고리즘
def ida_star(graph, start, goal):
    bound = heuristic(start, goal)
    path = [start]

    def search(path, g, bound):
        node = path[-1]
        f = g + heuristic(node, goal)
        if f > bound:
            return f, None
        if node == goal:
            return g, list(path)
        min_bound = float('inf')
        for neighbor in graph.neighbors(node):
            if neighbor not in path: 
                path.append(neighbor)
                t, result_path = search(path, g + graph[node][neighbor]['weight'], bound)
                if result_path is not None:
                    return t, result_path
                if t < min_bound:
                    min_bound = t
                path.pop()
        return min_bound, None

    while True:
        t, result_path = search(path, 0, bound)
        if result_path is not None:
            return result_path, t
        if t == float('inf'):
            return None, float('inf')
        bound = t

# 경로 및 길이 변수 초기화
dijkstra_path = []
dijkstra_length = 0
ida_path = []
ida_length = 0
ani = None

background_image = mpimg.imread(r"C:\Users\SON\Desktop\program\경북대 지도.jpg")  

# 애니메이션 함수
def update(frame):
    global dijkstra_path, ida_path, axs

    for i, ax in enumerate(axs):
        ax.clear()
        ax.imshow(background_image, extent=[0, 800, -800, 0], alpha=0.3)  
        ax.axis('off') 

    # 다익스트라 경로 애니메이션
    if len(dijkstra_path) > 0 and frame < len(dijkstra_path) - 1:
        axs[0].imshow(background_image, extent=[0, 800, -800, 0], alpha=0.3) 
        nx.draw_networkx_nodes(G, pos, ax=axs[0], nodelist=[node for node in G.nodes if '도로' not in node], node_color='lightblue', node_size=300)
        nx.draw_networkx_nodes(G, pos, ax=axs[0], nodelist=[node for node in G.nodes if '도로' in node], node_color='lightgray', node_size=10)
        nx.draw_networkx_edges(G, pos, ax=axs[0], edge_color='gray')
        nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes if '도로' not in node}, ax=axs[0], font_family='Malgun Gothic')
        axs[0].set_title(f'Dijkstra: Length={dijkstra_length}', fontdict={'family': 'Malgun Gothic'})

        for i in range(frame + 1):
            if i+1 < len(dijkstra_path):
                edge = [(dijkstra_path[i], dijkstra_path[i + 1])]
                nx.draw_networkx_edges(G, pos, edgelist=edge, ax=axs[0], edge_color='red', width=2)
                nx.draw_networkx_nodes(G, pos, nodelist=[dijkstra_path[i]], ax=axs[0], node_color='red', node_size=300)

    # A* 경로 애니메이션
    if len(ida_path) > 0 and frame < len(ida_path) - 1:
        axs[1].imshow(background_image, extent=[0, 800, -800, 0], alpha=0.3)  
        nx.draw_networkx_nodes(G, pos, ax=axs[1], nodelist=[node for node in G.nodes if '도로' not in node], node_color='lightgreen', node_size=300)
        nx.draw_networkx_nodes(G, pos, ax=axs[1], nodelist=[node for node in G.nodes if '도로' in node], node_color='lightgray', node_size=10)
        nx.draw_networkx_edges(G, pos, ax=axs[1], edge_color='gray')
        nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes if '도로' not in node}, ax=axs[1], font_family='Malgun Gothic')
        axs[1].set_title(f'A*: Length={ida_length}', fontdict={'family': 'Malgun Gothic'})

        for i in range(frame + 1):
            if i+1 < len(ida_path):
                edge = [(ida_path[i], ida_path[i + 1])]
                nx.draw_networkx_edges(G, pos, edgelist=edge, ax=axs[1], edge_color='blue', width=2)
                nx.draw_networkx_nodes(G, pos, nodelist=[ida_path[i]], ax=axs[1], node_color='blue', node_size=300)

# Tkinter를 이용하여 입력 받기
root = tk.Tk()
root.withdraw()  # 메인 윈도우 숨기기

available_nodes = [node for node in pos.keys() if '도로' not in node]
available_nodes_str = ', '.join(available_nodes)

start_node = simpledialog.askstring("시작 노드", f"시작 노드를 선택하세요:\n사용 가능한 노드:\n{available_nodes_str}")
finish_node = simpledialog.askstring("끝 노드", f"끝 노드를 선택하세요:\n사용 가능한 노드:\n{available_nodes_str}")

if start_node not in pos.keys():
    print(f"잘못된 시작 노드입니다: {start_node}")
    exit()
if finish_node not in pos.keys():
    print(f"잘못된 끝 노드입니다: {finish_node}")
    exit()

# 경로 및 길이 계산
dijkstra_path, dijkstra_length = dijkstra(G, start_node, finish_node)
ida_path, ida_length = ida_star(G, start_node, finish_node)

# 애니메이션 실행
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
ani = FuncAnimation(fig, update, frames=max(len(dijkstra_path), len(ida_path)) - 1, interval=1000, repeat=True)

# 초기 그래프 그리기
for i, ax in enumerate(axs):
    ax.imshow(background_image, extent=[0, 800, -800, 0], alpha=0.3)  
    ax.axis('off') 
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[node for node in G.nodes if '도로' not in node], node_color='lightblue', node_size=300)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[node for node in G.nodes if '도로' in node], node_color='lightgray', node_size=10)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray')
    nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes if '도로' not in node}, ax=ax, font_family='Malgun Gothic')

plt.show()
