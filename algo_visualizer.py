import tkinter as tk
from tkinter import ttk
import time
import threading

class AlgoTab:
    def __init__(self, notebook, title, default_input, code_lines, step_generator, visual_drawer):
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text=title)

        self.code_lines = code_lines
        self.current_step = 0
        self.steps = []
        self.running = False
        self.highlight_map = {}
        self.step_generator = step_generator
        self.visual_drawer = visual_drawer

        # Left Panel
        left = tk.Frame(self.frame)
        left.pack(side="left", fill="y")

        self.code_box = tk.Text(left, width=40, height=15, font=("Courier", 10))
        self.code_box.pack(pady=5)
        self.tags = []
        for i, line in enumerate(code_lines):
            tag = f"line{i}"
            self.code_box.insert("end", line + "\n", tag)
            self.tags.append(tag)
            self.code_box.tag_bind(tag, "<Enter>", lambda e, i=i: self.highlight_step(i))
            self.code_box.tag_bind(tag, "<Leave>", lambda e: self.clear_highlight())
        self.code_box.config(state="disabled")

        self.input_label = tk.Label(left, text="Input (comma-separated):")
        self.input_label.pack()
        self.input_entry = tk.Entry(left)
        self.input_entry.insert(0, default_input)
        self.input_entry.pack()

        btn_frame = tk.Frame(left)
        btn_frame.pack(pady=5)
        self.play_btn = tk.Button(btn_frame, text="▶ Play", command=self.play)
        self.step_btn = tk.Button(btn_frame, text="⏭ Step", command=self.step)
        self.reset_btn = tk.Button(btn_frame, text="🔁 Reset", command=self.reset)
        self.play_btn.pack(side="left", padx=2)
        self.step_btn.pack(side="left", padx=2)
        self.reset_btn.pack(side="left", padx=2)

        # Canvas
        self.canvas = tk.Canvas(self.frame, width=400, height=300, bg="white")
        self.canvas.pack(side="right", fill="both", expand=True)

        self.init_visual()

    def init_visual(self):
        self.arr = self.get_input_array()
        self.steps = self.step_generator(self.arr[:])
        self.current_step = 0
        self.draw_array()

    def get_input_array(self):
        try:
            return list(map(int, self.input_entry.get().split(',')))
        except:
            return [1, 3, 5, 7, 9]

    def draw_array(self, highlights=[]):
        self.canvas.delete("all")
        self.visual_drawer(self.canvas, self.arr, highlights)

    def highlight_step(self, step):
        if self.highlight_map:
            self.draw_array(self.highlight_map.get(step, []))

    def clear_highlight(self):
        self.draw_array()

    def play(self):
        if self.running:
            return
        self.running = True
        def runner():
            while self.current_step < len(self.steps):
                self.step()
                time.sleep(0.6)
            self.running = False
        threading.Thread(target=runner, daemon=True).start()

    def step(self):
        if self.current_step >= len(self.steps):
            return
        highlights, snapshot = self.steps[self.current_step]
        self.arr = snapshot
        self.highlight_map = {0: [], 1: highlights, 2: highlights}
        self.draw_array(highlights)
        self.current_step += 1

    def reset(self):
        self.running = False
        self.init_visual()


# Bubble Sort Step Generator
def generate_bubble_steps(arr):
    steps = []
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
            else:
                swapped = False
            steps.append(([j, j + 1], arr[:]))
    return steps

# Bubble Sort Drawer
def draw_bubble(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    for i, val in enumerate(arr):
        color = "red" if i in highlights else "skyblue"
        x0 = i * (w + 10) + 20
        y0 = 250 - val * 20
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0 + x1) // 2, y0 - 10, text=str(val))

# Binary Search Step Generator
def generate_binary_steps(arr):
    steps = []
    l, r = 0, len(arr) - 1
    while l <= r:
        mid = (l + r) // 2
        steps.append(([mid], arr[:]))
        if arr[mid] == 7: break
        elif arr[mid] < 7:
            l = mid + 1
        else:
            r = mid - 1
    return steps

# Binary Search Drawer
def draw_binary(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    for i, val in enumerate(arr):
        color = "green" if i in highlights else "lightblue"
        x0 = i * (w + 10) + 20
        y0 = 250 - val * 10
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0 + x1) // 2, y0 - 10, text=str(val))




# Fast & Slow Pointers Step Generator
def generate_fast_slow_steps(arr):
    steps = []
    slow = 0
    fast = 0
    while fast < len(arr) and fast + 1 < len(arr):
        steps.append(([slow, fast], arr[:]))
        slow += 1
        fast += 2
    return steps

# Fast & Slow Pointers Drawer
def draw_fast_slow(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    for i, val in enumerate(arr):
        color = "purple" if i in highlights else "orange"
        x0 = i * (w + 10) + 20
        y0 = 250 - val * 15
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0 + x1) // 2, y0 - 10, text=str(val))

# Prefix Sum Step Generator
def generate_prefix_steps(arr):
    steps = []
    prefix = []
    total = 0
    for i, num in enumerate(arr):
        total += num
        prefix.append(total)
        steps.append(([i], prefix[:]))
    return steps

# Prefix Sum Drawer
def draw_prefix(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    for i, val in enumerate(arr):
        color = "teal" if i in highlights else "lightgray"
        x0 = i * (w + 10) + 20
        y0 = 250 - val * 10
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0 + x1) // 2, y0 - 10, text=str(val))

# Depth-First Search (DFS) Step Generator
def generate_dfs_steps(arr):
    steps = []
    visited = [False] * len(arr)

    def dfs(i):
        if i >= len(arr) or visited[i]:
            return
        visited[i] = True
        steps.append(([i], visited[:]))
        dfs(i + 1)

    dfs(0)
    return steps

# DFS Drawer
def draw_dfs(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    for i, val in enumerate(arr):
        color = "blue" if i in highlights else "gray"
        x0 = i * (w + 10) + 20
        y0 = 250 - val * 10
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0 + x1)//2, y0 - 10, text=str(val))

# Breadth-First Search (BFS) Step Generator
from collections import deque
def generate_bfs_steps(arr):
    steps = []
    visited = [False] * len(arr)
    queue = deque()
    queue.append(0)

    while queue:
        i = queue.popleft()
        if i >= len(arr) or visited[i]:
            continue
        visited[i] = True
        steps.append(([i], visited[:]))
        if i + 1 < len(arr):
            queue.append(i + 1)
        if i + 2 < len(arr):
            queue.append(i + 2)

    return steps

# BFS Drawer
def draw_bfs(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    for i, val in enumerate(arr):
        color = "green" if i in highlights else "lightgray"
        x0 = i * (w + 10) + 20
        y0 = 250 - val * 10
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0 + x1)//2, y0 - 10, text=str(val))

# Backtracking Step Generator
def generate_backtracking_steps(arr):
    steps = []
    path = []

    def backtrack(start):
        steps.append((path[:], arr[:]))
        for i in range(start, len(arr)):
            path.append(arr[i])
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return steps

# Backtracking Drawer
def draw_backtracking(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    for i, val in enumerate(arr):
        color = "pink" if val in highlights else "lightgray"
        x0 = i * (w + 10) + 20
        y0 = 250 - val * 10
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0 + x1)//2, y0 - 10, text=str(val))


# Dynamic Programming (1D) Step Generator
def generate_dp_steps(arr):
    steps = []
    n = len(arr)
    dp = [0] * n
    for i in range(n):
        dp[i] = arr[i]
        if i > 0:
            dp[i] = max(dp[i], dp[i-1] + arr[i])
        steps.append(([i], dp[:]))
    return steps

# DP Drawer
def draw_dp(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    for i, val in enumerate(arr):
        color = "purple" if i in highlights else "lightgray"
        x0 = i * (w + 10) + 20
        y0 = 250 - val * 15
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0 + x1)//2, y0 - 10, text=str(val))

# Kadane’s Algorithm Step Generator
def generate_kadane_steps(arr):
    steps = []
    max_current = max_global = arr[0]
    dp = [max_current]
    steps.append(([0], dp[:]))
    for i in range(1, len(arr)):
        max_current = max(arr[i], max_current + arr[i])
        max_global = max(max_global, max_current)
        dp.append(max_current)
        steps.append(([i], dp[:]))
    return steps

# Kadane Drawer (reuse DP drawer for simplicity)
def draw_kadane(canvas, arr, highlights):
    draw_dp(canvas, arr, highlights)

# Topological Sort Step Generator
def generate_topo_steps(arr):
    # For demo, interpret arr as adjacency list encoded like: 
    # arr = [ [1,2], [3], [3], [] ] flattened as something? 
    # Simplify: assume arr contains node count, edges hardcoded.
    steps = []
    graph = {
        0: [1, 2],
        1: [3],
        2: [3],
        3: []
    }
    visited = set()
    stack = []
    def dfs(u):
        visited.add(u)
        steps.append(([u], list(stack)))
        for v in graph[u]:
            if v not in visited:
                dfs(v)
        stack.append(u)
        steps.append(([], list(stack)))
    for node in graph:
        if node not in visited:
            dfs(node)
    # steps show visiting and stack state (topo order)
    return steps

# Topological Sort Drawer
def draw_topo(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    # Show fixed nodes 0 to 3, color visited in highlights, stack at bottom
    for i in range(4):
        color = "orange" if i in highlights else "lightgray"
        x0 = i * (w + 10) + 20
        y0 = 150
        x1 = x0 + w
        y1 = 200
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0 + x1)//2, y0 - 10, text=str(i))
    # Show stack items at bottom
    stack = [str(x) for x in arr]
    canvas.create_text(200, 250, text="Stack: " + ", ".join(stack), font=("Arial", 12))


# 12. Monotonic Stack Step Generator
def generate_monotonic_stack_steps(arr):
    steps = []
    stack = []
    for i, val in enumerate(arr):
        while stack and arr[stack[-1]] < val:
            popped = stack.pop()
            steps.append(([], arr[:]))  # Show state after pop
        stack.append(i)
        steps.append((stack[:], arr[:]))  # Show current stack
    return steps

# Monotonic Stack Drawer
def draw_monotonic_stack(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    # Draw bars
    for i, val in enumerate(arr):
        color = "lightgray"
        if i in highlights:
            color = "orange"
        x0 = i * (w + 10) + 20
        y0 = 250 - val * 10
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0+x1)//2, y0-10, text=str(val))
    # Draw stack indexes as blue rectangles below
    for pos in highlights:
        x0 = pos * (w + 10) + 20
        y0 = 260
        x1 = x0 + w
        y1 = 280
        canvas.create_rectangle(x0, y0, x1, y1, fill="blue")
        canvas.create_text((x0+x1)//2, y0+10, text=str(pos), fill="white")

# 13. Dynamic Programming (2D) Step Generator
def generate_dp2d_steps(arr):
    # Assume arr is a flattened 3x3 matrix for simplicity
    steps = []
    n, m = 3, 3
    dp = [[0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            val = arr[i*m + j] if i*m + j < len(arr) else 0
            if i == 0 and j == 0:
                dp[i][j] = val
            elif i == 0:
                dp[i][j] = dp[i][j-1] + val
            elif j == 0:
                dp[i][j] = dp[i-1][j] + val
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + val
            # Flatten dp to 1D to show snapshot
            flat_dp = [dp[x][y] for x in range(n) for y in range(m)]
            steps.append(([(i*m)+j], flat_dp))
    return steps

# DP 2D Drawer
def draw_dp2d(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    n, m = 3, 3
    for i in range(n):
        for j in range(m):
            idx = i*m + j
            val = arr[idx] if idx < len(arr) else 0
            color = "purple" if idx in highlights else "lightgray"
            x0 = j * (w + 10) + 20
            y0 = 250 - val * 15
            x1 = x0 + w
            y1 = 250 - i * 80
            canvas.create_rectangle(x0, y1, x1, y0, fill=color)
            canvas.create_text((x0+x1)//2, y1 - 10, text=str(val))

# 14. Union Find (Disjoint Set Union) Step Generator
def generate_union_find_steps(arr):
    # Simplified DSU for 5 elements
    parent = list(range(5))
    rank = [0]*5
    steps = []
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a,b):
        rootA = find(a)
        rootB = find(b)
        if rootA != rootB:
            if rank[rootA] < rank[rootB]:
                parent[rootA] = rootB
            elif rank[rootB] < rank[rootA]:
                parent[rootB] = rootA
            else:
                parent[rootB] = rootA
                rank[rootA] += 1
            steps.append((parent[:], rank[:]))
    # Example unions
    union(0,1)
    union(1,2)
    union(3,4)
    union(2,4)
    return steps

# Union Find Drawer
def draw_union_find(canvas, parent, highlights):
    canvas.delete("all")
    w = 50
    for i, p in enumerate(parent):
        color = "lightgray"
        if i in highlights:
            color = "red"
        x0 = i * (w + 10) + 20
        y0 = 250 - 50
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0+x1)//2, y0 - 10, text=f"{i}")
        canvas.create_text((x0+x1)//2, y1 + 10, text=f"p:{p}")

# 15. Trie (Prefix Tree) Step Generator
def generate_trie_steps(arr):
    steps = []
    trie = {}
    for word in ["to", "tea", "ted", "ten"]:
        node = trie
        for ch in word:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
            steps.append(([ch], list(trie.keys())))
    return steps

# Trie Drawer

def draw_trie(canvas, arr, highlights):
    canvas.delete("all")
    keys = ", ".join(str(x) for x in arr) if arr else "None"
    canvas.create_text(200, 150, text=f"Trie keys: {keys}", font=("Arial", 14))



# 16. Divide and Conquer Step Generator
def generate_divide_conquer_steps(arr):
    steps = []
    def divide(l, r):
        if l >= r:
            steps.append((list(range(l, r+1)), arr[l:r+1]))
            return
        mid = (l + r) // 2
        steps.append((list(range(l, r+1)), arr[l:r+1]))
        divide(l, mid)
        divide(mid+1, r)
        steps.append((list(range(l, r+1)), arr[l:r+1]))
    divide(0, len(arr)-1)
    return steps

# Divide and Conquer Drawer
def draw_divide_conquer(canvas, arr, highlights):
    canvas.delete("all")
    w = 40
    for i, val in enumerate(arr):
        color = "cyan" if i in highlights else "lightgray"
        x0 = i * (w + 15) + 20
        y0 = 250 - val * 15
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0+x1)//2, y0-10, text=str(val))


# 17. Dynamic Programming on Trees Step Generator
def generate_dp_tree_steps(arr):
    # For simplicity, simulate a tree of 5 nodes, edges hardcoded
    steps = []
    n = 5
    tree = {
        0: [1, 2],
        1: [3],
        2: [4],
        3: [],
        4: []
    }
    dp = [0]*n
    visited = [False]*n

    def dfs(u):
        visited[u] = True
        total = arr[u] if u < len(arr) else 0
        for v in tree.get(u, []):
            if not visited[v]:
                dfs(v)
            total += dp[v]
        dp[u] = total
        steps.append(([u], dp[:]))

    dfs(0)
    return steps

# DP on Trees Drawer
def draw_dp_tree(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    n = 5
    for i in range(n):
        val = arr[i] if i < len(arr) else 0
        color = "purple" if i in highlights else "lightgray"
        x0 = i * (w + 10) + 20
        y0 = 250 - val * 15
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0+x1)//2, y0 - 10, text=str(val))
    canvas.create_text(300, 20, text="DP Tree sums visualized", font=("Arial", 12))


# 18. Greedy Algorithms Step Generator
def generate_greedy_steps(arr):
    steps = []
    arr_sorted = sorted(arr)
    used = [False]*len(arr)
    result = []
    for val in arr_sorted:
        for i, x in enumerate(arr):
            if not used[i] and x == val:
                used[i] = True
                result.append(x)
                steps.append(([i], result[:]))
                break
    return steps

# Greedy Drawer
def draw_greedy(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    for i, val in enumerate(arr):
        color = "gold" if i in highlights else "lightgray"
        x0 = i * (w + 10) + 20
        y0 = 250 - val * 10
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0+x1)//2, y0 - 10, text=str(val))


# 19. Binary Search on Answer Step Generator
def generate_binary_search_answer_steps(arr):
    steps = []
    low, high = min(arr), max(arr)
    def condition(x):
        # Dummy condition: can we find a number <= x?
        return any(val <= x for val in arr)

    while low <= high:
        mid = (low + high) // 2
        steps.append(([mid], arr[:]))
        if condition(mid):
            high = mid - 1
        else:
            low = mid + 1
    return steps

# Binary Search on Answer Drawer
def draw_binary_search_answer(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    for i, val in enumerate(arr):
        color = "lightgreen" if i in highlights else "lightgray"
        x0 = i * (w + 10) + 20
        y0 = 250 - val * 10
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0+x1)//2, y0 - 10, text=str(val))
    if highlights:
        canvas.create_text(300, 20, text=f"Mid = {highlights[0]}", font=("Arial", 12))


# 20. Dynamic Programming with Bitmask Step Generator
def generate_bitmask_dp_steps(arr):
    n = len(arr)
    steps = []
    dp = [False] * (1 << n)
    dp[0] = True
    for mask in range(1 << n):
        if dp[mask]:
            steps.append(([mask], dp[:]))
            for i in range(n):
                if not (mask & (1 << i)):
                    dp[mask | (1 << i)] = True
    return steps

# Bitmask DP Drawer
def draw_bitmask_dp(canvas, arr, highlights):
    canvas.delete("all")
    w = 20
    n = len(arr)
    max_mask = (1 << n)
    # Display as squares for each mask, highlight current
    for mask in range(max_mask):
        color = "red" if mask in highlights else "lightgray"
        row = mask // 8
        col = mask % 8
        x0 = col * (w + 5) + 20
        y0 = row * (w + 5) + 200
        x1 = x0 + w
        y1 = y0 + w
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0+x1)//2, (y0+y1)//2, text=bin(mask)[2:].zfill(n), font=("Courier", 8))


# 21. Dijkstra’s Algorithm Step Generator
import heapq
def generate_dijkstra_steps(arr):
    # Simple graph as adjacency list with weights (hardcoded)
    graph = {
        0: [(1, 4), (2, 1)],
        1: [(3, 1)],
        2: [(1, 2), (3, 5)],
        3: []
    }
    dist = {node: float('inf') for node in graph}
    dist[0] = 0
    steps = []
    heap = [(0,0)]
    visited = set()
    while heap:
        cur_dist, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        steps.append(([u], list(dist.values())))
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
                steps.append(([v], list(dist.values())))
    return steps

# Dijkstra Drawer
def draw_dijkstra(canvas, arr, highlights):
    canvas.delete("all")
    w = 60
    nodes = 4
    dist = arr if arr else [float('inf')] * nodes
    for i in range(nodes):
        val = dist[i] if i < len(dist) else float('inf')
        color = "yellow" if i in highlights else "lightgray"
        x0 = i * (w + 20) + 20
        y0 = 150
        x1 = x0 + w
        y1 = 200
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        text_val = str(val) if val != float('inf') else "∞"
        canvas.create_text((x0+x1)//2, y0 - 10, text=f"Node {i}")
        canvas.create_text((x0+x1)//2, y1 + 10, text=f"Dist: {text_val}")


# 22. Bellman-Ford Algorithm Step Generator
def generate_bellman_ford_steps(arr):
    # Simplified graph for demonstration (4 nodes)
    edges = [
        (0, 1, 4),
        (0, 2, 5),
        (1, 2, -2),
        (2, 3, 3),
        (1, 3, 4),
    ]
    dist = [float('inf')] * 4
    dist[0] = 0
    steps = []
    for _ in range(3):  # Relax edges |V|-1 times
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                steps.append(([v], dist[:]))
    return steps

# Bellman-Ford Drawer
def draw_bellman_ford(canvas, arr, highlights):
    canvas.delete("all")
    w = 60
    n = 4
    dist = arr if arr else [float('inf')] * n
    for i in range(n):
        val = dist[i]
        color = "orange" if i in highlights else "lightgray"
        x0 = i * (w + 20) + 20
        y0 = 150
        x1 = x0 + w
        y1 = 200
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        text_val = str(val) if val != float('inf') else "∞"
        canvas.create_text((x0 + x1)//2, y0 - 10, text=f"Node {i}")
        canvas.create_text((x0 + x1)//2, y1 + 10, text=f"Dist: {text_val}")



# 23. Floyd-Warshall Algorithm Step Generator
def generate_floyd_warshall_steps(arr):
    # 4-node graph with weighted adjacency matrix
    INF = float('inf')
    n = 4
    dist = [[INF]*n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    edges = [
        (0, 1, 3),
        (0, 2, 10),
        (1, 2, 1),
        (2, 3, 2),
        (1, 3, 7)
    ]
    for u, v, w in edges:
        dist[u][v] = w
    steps = []

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    # flatten dist matrix for visualization (just dist from i to j)
                    flat_dist = [dist[x][y] if dist[x][y] != INF else -1 for x in range(n) for y in range(n)]
                    steps.append(([i*n + j], flat_dist))
    return steps

# Floyd-Warshall Drawer
def draw_floyd_warshall(canvas, arr, highlights):
    canvas.delete("all")
    n = 4
    w = 30
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            val = arr[idx] if arr and idx < len(arr) else -1
            color = "purple" if idx in highlights else "lightgray"
            x0 = j * (w + 10) + 50
            y0 = i * (w + 10) + 50
            x1 = x0 + w
            y1 = y0 + w
            canvas.create_rectangle(x0, y0, x1, y1, fill=color)
            text = str(val) if val != -1 else "∞"
            canvas.create_text((x0+x1)//2, (y0+y1)//2, text=text, font=("Courier", 10))



# 24. A* Search Step Generator
import heapq
def generate_astar_steps(arr):
    # Simple grid graph for demo (5 nodes), edges with costs, heuristic ignored for simplicity
    graph = {
        0: [(1, 1), (2, 4)],
        1: [(3, 7)],
        2: [(3, 2)],
        3: [(4, 1)],
        4: []
    }
    dist = {node: float('inf') for node in graph}
    dist[0] = 0
    steps = []
    heap = [(0, 0)]  # (cost, node)
    visited = set()
    while heap:
        cost, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        steps.append(([u], list(dist.values())))
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
                steps.append(([v], list(dist.values())))
    return steps

# A* Drawer (similar to Dijkstra)
def draw_astar(canvas, arr, highlights):
    canvas.delete("all")
    w = 60
    nodes = 5
    dist = arr if arr else [float('inf')] * nodes
    for i in range(nodes):
        val = dist[i] if i < len(dist) else float('inf')
        color = "cyan" if i in highlights else "lightgray"
        x0 = i * (w + 15) + 20
        y0 = 150
        x1 = x0 + w
        y1 = 200
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        text_val = str(val) if val != float('inf') else "∞"
        canvas.create_text((x0 + x1)//2, y0 - 10, text=f"Node {i}")
        canvas.create_text((x0 + x1)//2, y1 + 10, text=f"Cost: {text_val}")



# 25. Ternary Search Step Generator
def generate_ternary_search_steps(arr):
    steps = []
    left, right = 0, len(arr) - 1
    def f(x):
        # Sample unimodal function using arr values
        return arr[x]

    while right - left > 2:
        m1 = left + (right - left) // 3
        m2 = right - (right - left) // 3
        steps.append(([m1, m2], arr[:]))
        if f(m1) < f(m2):
            left = m1 + 1
        else:
            right = m2 - 1
    steps.append(([], arr[:]))
    return steps

# Ternary Search Drawer
def draw_ternary_search(canvas, arr, highlights):
    canvas.delete("all")
    w = 50
    for i, val in enumerate(arr):
        color = "magenta" if i in highlights else "lightgray"
        x0 = i * (w + 10) + 20
        y0 = 250 - val * 10
        x1 = x0 + w
        y1 = 250
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        canvas.create_text((x0 + x1)//2, y0 - 10, text=str(val))
    
# Run GUI
def run_visualizer():
    root = tk.Tk()
    root.title("AlgoMonster Multi-Visualizer (Phase 2)")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    AlgoTab(
        notebook,
        "Bubble Sort",
        "5,2,9,1,6",
        [
            "for i in range(n):",
            "  for j in range(n - i - 1):",
            "    if arr[j] > arr[j + 1]:",
            "      arr[j], arr[j + 1] = arr[j + 1], arr[j]"
        ],
        generate_bubble_steps,
        draw_bubble
    )

    AlgoTab(
        notebook,
        "Binary Search",
        "1,3,5,7,9,11",
        [
            "while l <= r:",
            "  mid = (l + r) // 2",
            "  if arr[mid] == target:",
            "    return mid"
        ],
        generate_binary_steps,
        draw_binary
    )

    AlgoTab(
        notebook,
        "Fast & Slow Pointers",
        "1,2,3,4,5,6,7,8",
        [
            "while fast and fast.next:",
            "  slow = slow.next",
            "  fast = fast.next.next"
        ],
        generate_fast_slow_steps,
        draw_fast_slow
    )

    AlgoTab(
        notebook,
        "Prefix Sum",
        "1,2,3,4,5",
        [
            "prefix[0] = arr[0]",
            "for i in range(1, n):",
            "  prefix[i] = prefix[i-1] + arr[i]"
        ],
        generate_prefix_steps,
        draw_prefix
    )

    AlgoTab(
        notebook,
        "DFS",
        "1,2,3,4,5",
        [
            "def dfs(node):",
            "  if not node: return",
            "  visit(node)",
            "  dfs(node.left)",
            "  dfs(node.right)"
        ],
        generate_dfs_steps,
        draw_dfs
    )

    AlgoTab(
        notebook,
        "BFS",
        "1,2,3,4,5",
        [
            "queue = [root]",
            "while queue:",
            "  node = queue.pop(0)",
            "  visit(node)",
            "  add children to queue"
        ],
        generate_bfs_steps,
        draw_bfs
    )

    AlgoTab(
        notebook,
        "Backtracking",
        "1,2,3",
        [
            "def backtrack(path):",
            "  if end_condition:",
            "    output(path)",
            "  for choice in choices:",
            "    make_choice",
            "    backtrack(path)",
            "    undo_choice"
        ],
        generate_backtracking_steps,
        draw_backtracking
    )


    AlgoTab(
        notebook,
        "Dynamic Programming (1D)",
        "1,-1,2,3,-2,1",
        [
            "dp = [0]*n",
            "for i in range(n):",
            "  dp[i] = max(arr[i], dp[i-1] + arr[i]) if i > 0 else arr[i]"
        ],
        generate_dp_steps,
        draw_dp
    )

    AlgoTab(
        notebook,
        "Kadane’s Algorithm",
        "1,-2,3,4,-1,2,1,-5,4",
        [
            "max_current = max_global = arr[0]",
            "for i in range(1, n):",
            "  max_current = max(arr[i], max_current + arr[i])",
            "  max_global = max(max_global, max_current)",
            "return max_global"
        ],
        generate_kadane_steps,
        draw_kadane
    )

    AlgoTab(
        notebook,
        "Topological Sort",
        "N/A",
        [
            "visited = set()",
            "stack = []",
            "def dfs(node):",
            "  visited.add(node)",
            "  for neighbor in graph[node]:",
            "    if neighbor not in visited:",
            "      dfs(neighbor)",
            "  stack.append(node)"
        ],
        generate_topo_steps,
        draw_topo
    )



    
    AlgoTab(
        notebook,
        "Monotonic Stack",
        "2,1,5,6,2,3",
        [
            "stack = []",
            "for x in arr:",
            "  while stack and stack[-1] < x:",
            "    stack.pop()",
            "  stack.append(x)"
        ],
        generate_monotonic_stack_steps,
        draw_monotonic_stack
    )

    AlgoTab(
        notebook,
        "Dynamic Programming (2D)",
        "1,2,3,4,5,6,7,8,9",
        [
            "dp = [[0]*m for _ in range(n)]",
            "for i in range(n):",
            "  for j in range(m):",
            "    dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + arr[i][j]"
        ],
        generate_dp2d_steps,
        draw_dp2d
    )

    AlgoTab(
        notebook,
        "Union Find",
        "N/A",
        [
            "parent = [0,1,2,3,4]",
            "def find(x):",
            "  if parent[x] != x:",
            "    parent[x] = find(parent[x])",
            "def union(a,b):",
            "  # union logic here"
        ],
        generate_union_find_steps,
        draw_union_find
    )

    AlgoTab(
        notebook,
        "Trie (Prefix Tree)",
        "N/A",
        [
            "trie = {}",
            "for word in words:",
            "  node = trie",
            "  for ch in word:",
            "    if ch not in node:",
            "      node[ch] = {}",
            "    node = node[ch]"
        ],
        generate_trie_steps,
        draw_trie
    )


    
    AlgoTab(
        notebook,
        "Divide and Conquer",
        "7,2,5,3,8,1",
        [
            "def divide(l, r):",
            "  if l >= r: return",
            "  mid = (l + r) // 2",
            "  divide(l, mid)",
            "  divide(mid+1, r)"
        ],
        generate_divide_conquer_steps,
        draw_divide_conquer
    )

    AlgoTab(
        notebook,
        "DP on Trees",
        "2,3,1,4,5",
        [
            "def dfs(u):",
            "  dp[u] = arr[u]",
            "  for v in tree[u]:",
            "    dfs(v)",
            "    dp[u] += dp[v]"
        ],
        generate_dp_tree_steps,
        draw_dp_tree
    )

    AlgoTab(
        notebook,
        "Greedy Algorithms",
        "5,3,6,2,10",
        [
            "arr.sort()",
            "for x in arr:",
            "  pick x if possible"
        ],
        generate_greedy_steps,
        draw_greedy
    )

    AlgoTab(
        notebook,
        "Binary Search on Answer",
        "1,5,9,12,15",
        [
            "low, high = min(arr), max(arr)",
            "while low <= high:",
            "  mid = (low + high) // 2",
            "  if condition(mid): high = mid - 1",
            "  else: low = mid + 1"
        ],
        generate_binary_search_answer_steps,
        draw_binary_search_answer
    )

    AlgoTab(
        notebook,
        "DP with Bitmask",
        "1,0,1",
        [
            "dp = [False]*(1 << n)",
            "dp[0] = True",
            "for mask in range(1 << n):",
            "  if dp[mask]:",
            "    for i in range(n):",
            "      if not mask & (1 << i):",
            "        dp[mask | (1 << i)] = True"
        ],
        generate_bitmask_dp_steps,
        draw_bitmask_dp
    )

    AlgoTab(
        notebook,
        "Dijkstra’s Algorithm",
        "N/A",
        [
            "dist = [∞]*n",
            "dist[start] = 0",
            "while heap:",
            "  u = pop_min_dist()",
            "  for v in neighbors[u]:",
            "    relax edges"
        ],
        generate_dijkstra_steps,
        draw_dijkstra
    )


    
    AlgoTab(
        notebook,
        "Bellman-Ford",
        "N/A",
        [
            "dist = [∞]*n",
            "dist[start] = 0",
            "for _ in range(n-1):",
            "  for u,v,w in edges:",
            "    relax edges"
        ],
        generate_bellman_ford_steps,
        draw_bellman_ford
    )

    AlgoTab(
        notebook,
        "Floyd-Warshall",
        "N/A",
        [
            "dist = [[∞]*n for _ in range(n)]",
            "for i in range(n):",
            "  dist[i][i] = 0",
            "for k in range(n):",
            "  for i in range(n):",
            "    for j in range(n):",
            "      dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])"
        ],
        generate_floyd_warshall_steps,
        draw_floyd_warshall
    )

    AlgoTab(
        notebook,
        "A* Search",
        "N/A",
        [
            "dist = {node:∞ for node in graph}",
            "dist[start] = 0",
            "while heap:",
            "  u = pop node with lowest cost",
            "  for v in neighbors[u]:",
            "    relax edges with heuristic"
        ],
        generate_astar_steps,
        draw_astar
    )

    AlgoTab(
        notebook,
        "Ternary Search",
        "5,10,15,20,25",
        [
            "while right - left > 2:",
            "  m1 = left + (right-left)//3",
            "  m2 = right - (right-left)//3",
            "  if f(m1) < f(m2): left = m1 + 1",
            "  else: right = m2 - 1"
        ],
        generate_ternary_search_steps,
        draw_ternary_search
    )


    root.mainloop()

run_visualizer()
