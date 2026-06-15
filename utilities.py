import math
import random
import matplotlib.pyplot as plt

# 공구 교체 페널티 (Constant Tool Change Penalty: T_ATC)
T_ATC = 2.0 
FEED  = 20.0 # mm/s

class Hole:
    def __init__(self, id, x, y, tool):
        self.id = id
        self.x = x
        self.y = y
        self.tool = tool

    def __repr__(self):
        return f"Hole({self.id}: {self.tool} @ {self.x},{self.y})"

def get_tool_radius(tool_name):
    """
    예: 'M4' -> 지름 4mm -> 반지름 2.0 / 'M10' -> 지름 10mm -> 반지름 5.0
    """
    try:
        # 공구 이름이 'M' 또는 'm'으로 시작하는지 확인
        if tool_name.upper().startswith('M'):
            diameter = float(tool_name[1:])
            return diameter / 2.0
    except ValueError:
        pass
        
    return 0.0  # 기본값

def get_distance(h1, h2):
    """두 홀 사이의 순수 물리적 유클리드 거리를 계산합니다."""
    return math.hypot(h1.x - h2.x, h1.y - h2.y)

def get_cost(h1, h2):
    """
    [핵심] 상태 의존적 비용 계산 함수
    물리적 거리에 더해, 두 홀의 필요 공구가 다를 경우 T_ATC 페널티를 부과합니다.
    """
    cost = get_distance(h1, h2) / FEED  # 이동 시간으로 변환
    if h1.tool != h2.tool:
        cost += T_ATC
    return cost

def calculate_path_cost(path, holes, return_to_start=True):
    """
    주어진 경로(순서)의 전체 사이클 타임(물리적 이동 + 누적 T_ATC 페널티)을 계산합니다.
    """
    if not path:
        return 0.0
        
    total_cost = 0.0
    for i in range(len(path) - 1):
        total_cost += get_cost(holes[path[i]], holes[path[i+1]])
        
    if return_to_start and len(path) > 1:
        # 마지막 노드에서 다시 시작 노드로 돌아오는 비용 (TSP 순환)
        total_cost += get_cost(holes[path[-1]], holes[path[0]])
        
    return total_cost

def generate_random_holes(num_holes, width, height, tool_types):
    """
    무작위 좌표와 공구 종류(Tool type)를 가진 홀(Hole) 데이터 세트를 생성합니다.
    물리적인 겹침을 방지하기 위해 공구의 반지름과 안전 여유 공간을 고려합니다.
    """
    holes = []

    # 1. 시작점 (Depot) 설정 - 고정 좌표 (0.0, 0.0)
    start_tool = tool_types[0]
    holes.append(Hole(id=0, x=0.0, y=0.0, tool=start_tool))

    padding = 1.0 # 홀과 홀 사이의 최소 안전 여유 공간 (mm)

    # 2. 나머지 홀 생성 (ID 중복을 피하기 위해 1부터 시작)
    for i in range(1, num_holes):
        tool = random.choice(tool_types)
        radius = get_tool_radius(tool)
        
        valid_position = False
        max_attempts = 1000 # 무한 루프(공간 부족) 방지를 위한 최대 시도 횟수
        attempts = 0
        
        while not valid_position and attempts < max_attempts:
            x = round(random.uniform(0, width), 2)
            y = round(random.uniform(0, height), 2)
            
            # 겹침 확인 (Collision Detection)
            collision = False
            for h in holes:
                existing_radius = get_tool_radius(h.tool)
                
                # 최소 허용 거리 = (내 반지름 + 상대 반지름 + 안전 여유 공간)
                min_dist = radius + existing_radius + padding
                
                # 두 홀 사이의 유클리드 거리 계산
                dist = math.hypot(x - h.x, y - h.y)
                
                if dist < min_dist:
                    collision = True
                    break # 하나라도 겹치면 더 검사할 필요 없이 즉시 파기
                    
            if not collision:
                valid_position = True
                holes.append(Hole(id=i, x=x, y=y, tool=tool))
            
            attempts += 1
            
        if not valid_position:
            print(f"[경고] 도면 공간이 부족하여 {i}번째 홀을 배치하지 못했습니다.")
            
    return holes


def plot_path(ax, holes, path, title, cost, exec_time):
    """
    각 알고리즘이 찾은 경로와 홀 위치를 서브플롯에 그리는 헬퍼 함수
    """
    # 1. 경로 그리기 (회색 실선)
    if path:
        # 경로의 끝에서 다시 시작점으로 돌아오는 순환 경로 완성
        path_x = [holes[i].x for i in path] + [holes[path[0]].x]
        path_y = [holes[i].y for i in path] + [holes[path[0]].y]
        ax.plot(path_x, path_y, linestyle='-', color='gray', alpha=0.6, zorder=1)

    # 2. 홀 그리기 (종류에 따라 색상과 크기(반지름) 차등 부여)
    tool_types = sorted(set(h.tool for h in holes))  # 고유한 공구 종류 추출
    cmap = plt.get_cmap('tab10')  # 최대 10가지 색상 지원하는 컬러맵
    for h in holes:
        diameter = int(h.tool[1:])  # 'M4' -> 4, 'M8' -> 8
        s = diameter * 10
        color_idx = tool_types.index(h.tool)
        c = cmap(color_idx)
            
        ax.scatter(h.x, h.y, color=c, s=s, zorder=2, edgecolors='black')
        # 홀 번호를 표시하고 싶다면 아래 주석을 해제하세요
        # ax.text(h.x, h.y, f" {h.id}", fontsize=8, verticalalignment='bottom')
    
    ax.scatter(holes[0].x, holes[0].y, color='gold', s=200, zorder=3, edgecolors='black', label='Origin')  # 시작점 강조
    # ax.text(holes[0].x, holes[0].y - 5, "Depot (0)", fontsize=10, fontweight='bold', ha='center', color='black')

    # 타이틀 설정 (알고리즘 이름, 최종 비용, 실행 시간)
    ax.set_title(f"{title}\nCost: {cost:.2f} | Time: {exec_time:.4f}s", fontsize=11)
    ax.set_aspect('equal') # X, Y축 비율을 맞춰서 왜곡 방지
    ax.grid(True, linestyle=':', alpha=0.6)