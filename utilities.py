import random
import math
import pandas as pd
import matplotlib.pyplot as plt

def generate_hole_coordinates(N, x_max, y_max, r_hole, max_attempts=100000):
    
    """
    겹치지 않는 N개의 무작위 홀 중심 좌표를 생성합니다.
    
    Parameters:
    - N (int): 생성할 홀의 개수
    - x_max (float): 그리드의 가로 크기 (0 ~ x_max)
    - y_max (float): 그리드의 세로 크기 (0 ~ y_max)
    - r_hole (float): 홀의 반지름
    - max_attempts (int): 무한 루프 방지를 위한 최대 샘플링 시도 횟수
    
    Returns:
    - list of tuple: [(x1, y1), (x2, y2), ...] 형태의 홀 중심 좌표 리스트
    """

    random.seed(42)  # 재현 가능한 결과를 위해 시드 설정

    centers = []
    min_distance = 2 * r_hole
    min_distance_sq = min_distance ** 2  # 제곱 연산으로 루트 계산을 생략하여 속도 최적화
    
    attempts = 0
    while len(centers) < N and attempts < max_attempts:
        attempts += 1
        
        # 1. 지정된 직사각형 그리드 영역 내에서 무작위 좌표 샘플링
        x = random.uniform(r_hole, x_max - r_hole)
        y = random.uniform(r_hole, y_max - r_hole)
        
        # 2. 기존에 생성된 모든 홀들과의 거리 제약 조건 검사 (겹침 확인)
        overlap = False
        for cx, cy in centers:
            # 두 중심점 사이의 유클리드 거리의 제곱 계산
            dist_sq = (x - cx)**2 + (y - cy)**2
            if dist_sq <= min_distance_sq:
                overlap = True
                break
                
        # 3. 겹치지 않는 경우에만 좌표 리스트에 추가
        if not overlap:
            centers.append((x, y))
            
    # 지정한 횟수 내에 N개를 채우지 못한 경우 경고 출력
    if len(centers) < N:
        print(f"[Warning] 밀도 제한으로 인해 목표치({N}개) 중 {len(centers)}개의 홀만 생성되었습니다.")
        print("팁: 그리드 크기(x_max, y_max)를 키우거나 홀 반지름(r_hole)을 줄여보세요.")
        
    return centers

def print_time(start_time, end_time):
    elapsed_time = end_time - start_time
    
    if elapsed_time < 1:
        print(f"소요 시간: {elapsed_time * 1000:.2f} ms")
    else:
        print(f"소요 시간: {elapsed_time:.2f} s")
        

def visualize_four_paths(hole_positions, greedy_path, two_opt_path, held_karp_path, proposed_path):
    """
    4개의 알고리즘 결과를 2x2 서브플롯으로 시각화하는 함수입니다.
    경로 리스트가 비어있어도 에러가 발생하지 않도록 유연하게 설계되었습니다.
    """
    
    # 1. 시각화를 위한 2x2 서브플롯 캔버스 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparison of Tool Path Optimization Algorithms', fontsize=18, fontweight='bold')
    
    # 2. 타공 지점(Holes)의 X, Y 좌표 추출
    # (어떤 경로를 그리든 배경에 동일한 홀 배치를 보여주기 위함)
    holes_x = [p[0] for p in hole_positions]
    holes_y = [p[1] for p in hole_positions]
    
    # 3. 그릴 경로들과 제목을 리스트로 묶어 반복문 처리
    path_data = [
        ("Greedy Algorithm", greedy_path, axes[0, 0]),
        ("2-Opt Heuristic", two_opt_path, axes[0, 1]),
        ("Held-Karp (Exact DP)", held_karp_path, axes[1, 0]),
        ("Proposed (Relay Recursive DP)", proposed_path, axes[1, 1])
    ]
    
    # 4. 각 서브플롯에 순차적으로 그리기
    for title, path, ax in path_data:
        # 배경에 모든 홀을 회색 점으로 표시
        ax.scatter(holes_x, holes_y, c='gray', s=30, zorder=2, label='Holes')
        
        # 원점(0,0)을 빨간색 별표로 특별히 표시 (공구 출발점)
        ax.scatter(0, 0, c='red', s=150, marker='*', zorder=3, label='Depot (0,0)')
        
        # 경로가 존재하는 경우에만 선 긋기 (길이가 달라도 안전함)
        if path and len(path) > 1:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            
            # 경로를 파란색 선으로 연결
            ax.plot(path_x, path_y, c='blue', linewidth=1.5, alpha=0.7, zorder=1)
            
            # 경로의 방향성을 보여주기 위해 시작과 끝을 살짝 표시
            ax.plot(path_x[0], path_y[0], 'go', markersize=6, label='Start') # 녹색 점: 시작점
            ax.plot(path_x[-1], path_y[-1], 'ro', markersize=6, label='End') # 빨간 점: 끝점
            
        else:
            # Held-karp 등 연산 실패/생략으로 빈 배열이 들어온 경우
            ax.text(0.5, 0.5, 'Path Not Available\n(Time Out or Empty)', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')
            
        # 축 및 제목 설정
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)

    # 레이아웃 간격 자동 조정 및 출력
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 메인 타이틀을 위한 여백 확보
    plt.show()

# ==============================================================
# [실행 방법]
# 모든 알고리즘을 돌려 변수들이 준비되었다고 가정하고 아래처럼 호출합니다.
# 

# ==============================================================