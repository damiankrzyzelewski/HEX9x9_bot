import os
import tkinter as tk
import math
import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import random
import collections
import threading
import tkinter.ttk as ttk

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.c1(x)))
        out = self.bn2(self.c2(out))
        out += identity
        return torch.relu(out)

class HexCNN(BaseFeaturesExtractor):
    def __init__(self, obs_space, features_dim=256):
        super().__init__(obs_space, features_dim)
        self.net = nn.Sequential(
            nn.Conv2d(obs_space.shape[0], 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128), ResidualBlock(128), ResidualBlock(128),
            ResidualBlock(128), ResidualBlock(128),
            nn.Flatten()
        )
        with torch.no_grad():
            n = self.net(torch.as_tensor(obs_space.sample()[None]).float()).shape[1]
        self.lin = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

    def forward(self, x):
        return self.lin(self.net(x))

class HexState:
    def __init__(self, board_size=9):
        self.size = board_size
        self.board = [-1] * (board_size * board_size)
        self._current_player = 0
        self._is_terminal = False
        self._returns = [0.0, 0.0]

        self.moves_played = 0
        self.first_move_action = None
        self.swap_action = board_size * board_size

    def current_player(self): return self._current_player
    def is_terminal(self): return self._is_terminal
    def returns(self): return self._returns

    def legal_actions(self):
        actions = [i for i, s in enumerate(self.board) if s == -1]
        if self.moves_played == 1:
            actions.append(self.swap_action)
        return actions

    def apply_action(self, action):
        if self._is_terminal:
            return

        if action == self.swap_action:
            self.board[self.first_move_action] = 1
            self._current_player = 0
            self.moves_played += 1
            return

        if self.board[action] != -1:
            return

        self.board[action] = self._current_player

        if self.moves_played == 0:
            self.first_move_action = action

        if self._check_win(self._current_player):
            self._is_terminal = True
            self._returns = [1, -1] if self._current_player == 0 else [-1, 1]
        else:
            self._current_player = 1 - self._current_player

        self.moves_played += 1

    def _get_neighbors(self, action):
        r, c = action // self.size, action % self.size
        dirs = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        out = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                out.append(nr * self.size + nc)
        return out

    def _check_win(self, player):
        visited = set()
        stack = []

        for i in range(self.size):
            node = i if player == 0 else i * self.size
            if self.board[node] == player:
                stack.append(node)

        while stack:
            curr = stack.pop()
            if curr in visited:
                continue

            visited.add(curr)
            r, c = curr // self.size, curr % self.size

            if (player == 0 and r == self.size - 1) or \
               (player == 1 and c == self.size - 1):
                return True

            for n in self._get_neighbors(curr):
                if self.board[n] == player and n not in visited:
                    stack.append(n)

        return False

class PPOAgent:
    def __init__(self, model_path, board_size=9):
        self.size = board_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.valid = False

        if os.path.exists(model_path):
            print(f"Ładowanie modelu: {model_path}")
            self.model = MaskablePPO.load(model_path, device=self.device)
            self.valid = True
            print("Model gotowy")
        else:
            print(f"Brak modelu PPO: {model_path} (Będzie grał losowo)")

    def _obs(self, state, p, rot):
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)
        my, op = p, 1 - p

        for r in range(self.size):
            for c in range(self.size):
                rr, rc = (self.size-1-r, self.size-1-c) if rot else (r, c)
                if p == 1:
                    rr, rc = rc, rr

                val = state.board[rr * self.size + rc]

                if val == my:
                    obs[0, r, c] = 1
                elif val == op:
                    obs[1, r, c] = 1
                else:
                    obs[2, r, c] = 1

        return obs

    def _map(self, a, p, rot):
        if a == self.size * self.size:
            return a

        r, c = a // self.size, a % self.size

        if rot:
            r, c = self.size - 1 - r, self.size - 1 - c

        if p == 1:
            r, c = c, r

        return r * self.size + c

    def valid_action_mask(self, state, p, rot):
        mask = np.zeros(self.size * self.size + 1, dtype=np.bool_)
        for a in range(self.size * self.size + 1):
            if self._map(a, p, rot) in state.legal_actions():
                mask[a] = True
        return mask

    def step(self, state, p, rot):
        if not self.valid:
            return random.choice(state.legal_actions())

        obs = self._obs(state, p, rot)
        mask = self.valid_action_mask(state, p, rot)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=False)
        return self._map(int(action), p, rot)

def generate_fair_openings(size):
    openings = []
    center = size // 2
    
    # Przeszukujemy planszę, omijając skrajne krawędzie
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            # Liczymy dystans od środka (tzw. dystans Manhattan)
            dist = abs(r - center) + abs(c - center)
            
            # Bierzemy pola, które są oddalone od środka o 2 lub 3 kratki
            if dist == 2 or dist == 3:
                openings.append(r * size + c)
                
    return openings

FAIR_OPENINGS = generate_fair_openings(9)

class LinearHexBot2:
    def __init__(self):
        self.N_CELLS = 81  # dla 9x9

        self.WEIGHTS = np.array([
        2.294,  0.703, -1.046, -0.566,
        -1.145, -0.177,  1.360,  0.480,
        0.383, -0.308, -0.359,  0.089
        ], dtype=np.float32)
    
    def _features_as_player(self, state, action, player, cache):
        return self._features(state, action, player, cache)

    def step(self, state, p=None, rot=None):
        player = state.current_player()
        legal = state.legal_actions()
        if not legal:
            return None

        cache = self._cache(state, player)

        best_score = None
        best_action = None

        for action in legal:
            if action == state.swap_action:
                continue
            feat = self._features(state, action, player, cache)
            score = float(np.dot(self.WEIGHTS, feat))
            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        if state.swap_action in legal:
            first = state.first_move_action

            import copy
            empty_state = copy.copy(state)
            empty_state.board = [-1] * (state.size * state.size)

            swap_cache = self._cache(empty_state, 0)
            feat = self._features(empty_state, first, 0, swap_cache)
            swap_score = float(np.dot(self.WEIGHTS, feat))

            print(f"DEBUG swap={swap_score:.3f} best={best_score:.3f} first={first}")

            if best_score is None or swap_score > best_score:
                return state.swap_action

        return best_action

    def _distances(self, state, player, start_edge):
        size = state.size
        dist = [float("inf")] * (size * size)
        q = collections.deque()

        for i in range(size):
            if player == 0:
                node = i if start_edge else (size - 1) * size + i
            else:
                node = i * size if start_edge else i * size + (size - 1)

            if state.board[node] == player:
                dist[node] = 0
                q.appendleft(node)
            elif state.board[node] == -1:
                dist[node] = 1
                q.append(node)

        while q:
            curr = q.popleft()
            d = dist[curr]

            for n in state._get_neighbors(curr):
                if state.board[n] == 1 - player:
                    continue

                cost = 0 if state.board[n] == player else 1
                nd = d + cost

                if nd < dist[n]:
                    dist[n] = nd
                    if cost == 0:
                        q.appendleft(n)
                    else:
                        q.append(n)

        return dist

    def _components(self, state, color):
        size = state.size
        comp_id = [-1] * (size * size)
        comp_sizes = []
        cid = 0

        for i in range(size * size):
            if state.board[i] != color or comp_id[i] != -1:
                continue

            q = collections.deque([i])
            comp_id[i] = cid
            nodes = [i]

            while q:
                curr = q.popleft()
                for n in state._get_neighbors(curr):
                    if state.board[n] == color and comp_id[n] == -1:
                        comp_id[n] = cid
                        q.append(n)
                        nodes.append(n)

            comp_sizes.append(len(nodes))
            cid += 1

        return comp_id, comp_sizes

    def _bridge(self, state, action, color):
        nbs = [n for n in state._get_neighbors(action) if state.board[n] == color]
        if len(nbs) < 2:
            return 0

        count = 0
        for i in range(len(nbs)):
            for j in range(i + 1, len(nbs)):
                u, v = nbs[i], nbs[j]
                inter = set(state._get_neighbors(u)).intersection(state._get_neighbors(v))

                if action in inter and len(inter) == 2:
                    other = next(iter(inter - {action}))
                    if state.board[other] != (1 - color):
                        count += 1

        return count

    def _cache(self, state, player):
        opp = 1 - player

        my_ds = self._distances(state, player, True)
        my_de = self._distances(state, player, False)
        opp_ds = self._distances(state, opp, True)
        opp_de = self._distances(state, opp, False)

        my_comp, my_sizes = self._components(state, player)

        return {
            "my_ds": my_ds,
            "my_de": my_de,
            "opp_ds": opp_ds,
            "opp_de": opp_de,
            "my_comp": my_comp,
            "my_sizes": my_sizes,
        }

    def _features(self, state, action, player, cache):
        size = state.size
        opp = 1 - player

        if action == state.swap_action:
            return np.zeros(12, dtype=np.float32)

        r, c = action // size, action % size

        # --- center ---
        max_center = 2 * (size - 1)
        center_dist = abs(r - size // 2) + abs(c - size // 2)
        center_score = 1.0 - center_dist / max_center

        # --- paths ---
        norm = 2 * size
        my_path = cache["my_ds"][action] + cache["my_de"][action]
        opp_path = cache["opp_ds"][action] + cache["opp_de"][action]

        # --- neighbors ---
        neigh = state._get_neighbors(action)
        friend = [n for n in neigh if state.board[n] == player]
        enemy = [n for n in neigh if state.board[n] == opp]
        empty = [n for n in neigh if state.board[n] == -1]

        # --- components ---
        comp_id = cache["my_comp"]
        comp_sizes = cache["my_sizes"]

        touched = set(comp_id[n] for n in friend if comp_id[n] != -1)
        comp_size = 1 + sum(comp_sizes[c] for c in touched)

        return np.array([
            1 - min(my_path, norm * 2) / (norm * 2),
            1 - min(opp_path, norm * 2) / (norm * 2),
            (opp_path - my_path) / norm,
            center_score,
            len(friend) / 6.0,
            len(enemy) / 6.0,
            len(touched) / 6.0,
            comp_size / float(self.N_CELLS),
            self._bridge(state, action, player) / 6.0,
            self._bridge(state, action, opp) / 6.0,
            len(empty) / 6.0,
            0.0,
        ], dtype=np.float32)
        
        
class LinearHexBotTest:
    def __init__(self):
        self.N_CELLS = 81  # dla 9x9

        self.WEIGHTS = np.array([
            8.064,  24.642, -3.659, -5.511,
            -4.437,   1.050,  6.780,  8.348,
            1.973,  -4.067, -1.587,  0.115
        ], dtype=np.float32)
    
    def _features_as_player(self, state, action, player, cache):
        return self._features(state, action, player, cache)

    def step(self, state, p=None, rot=None):
        player = state.current_player()
        legal = state.legal_actions()
        if not legal:
            return None

        cache = self._cache(state, player)

        best_score = None
        best_action = None

        for action in legal:
            if action == state.swap_action:
                continue
            feat = self._features(state, action, player, cache)
            score = float(np.dot(self.WEIGHTS, feat))
            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        if state.swap_action in legal:
            first = state.first_move_action

            import copy
            empty_state = copy.copy(state)
            empty_state.board = [-1] * (state.size * state.size)

            swap_cache = self._cache(empty_state, 0)
            feat = self._features(empty_state, first, 0, swap_cache)
            swap_score = float(np.dot(self.WEIGHTS, feat))

            print(f"DEBUG swap={swap_score:.3f} best={best_score:.3f} first={first}")

            if best_score is None or swap_score > best_score:
                return state.swap_action

        return best_action

    def _distances(self, state, player, start_edge):
        size = state.size
        dist = [float("inf")] * (size * size)
        q = collections.deque()

        for i in range(size):
            if player == 0:
                node = i if start_edge else (size - 1) * size + i
            else:
                node = i * size if start_edge else i * size + (size - 1)

            if state.board[node] == player:
                dist[node] = 0
                q.appendleft(node)
            elif state.board[node] == -1:
                dist[node] = 1
                q.append(node)

        while q:
            curr = q.popleft()
            d = dist[curr]

            for n in state._get_neighbors(curr):
                if state.board[n] == 1 - player:
                    continue

                cost = 0 if state.board[n] == player else 1
                nd = d + cost

                if nd < dist[n]:
                    dist[n] = nd
                    if cost == 0:
                        q.appendleft(n)
                    else:
                        q.append(n)

        return dist

    def _components(self, state, color):
        size = state.size
        comp_id = [-1] * (size * size)
        comp_sizes = []
        cid = 0

        for i in range(size * size):
            if state.board[i] != color or comp_id[i] != -1:
                continue

            q = collections.deque([i])
            comp_id[i] = cid
            nodes = [i]

            while q:
                curr = q.popleft()
                for n in state._get_neighbors(curr):
                    if state.board[n] == color and comp_id[n] == -1:
                        comp_id[n] = cid
                        q.append(n)
                        nodes.append(n)

            comp_sizes.append(len(nodes))
            cid += 1

        return comp_id, comp_sizes

    def _bridge(self, state, action, color):
        nbs = [n for n in state._get_neighbors(action) if state.board[n] == color]
        if len(nbs) < 2:
            return 0

        count = 0
        for i in range(len(nbs)):
            for j in range(i + 1, len(nbs)):
                u, v = nbs[i], nbs[j]
                inter = set(state._get_neighbors(u)).intersection(state._get_neighbors(v))

                if action in inter and len(inter) == 2:
                    other = next(iter(inter - {action}))
                    if state.board[other] != (1 - color):
                        count += 1

        return count

    def _cache(self, state, player):
        opp = 1 - player

        my_ds = self._distances(state, player, True)
        my_de = self._distances(state, player, False)
        opp_ds = self._distances(state, opp, True)
        opp_de = self._distances(state, opp, False)

        my_comp, my_sizes = self._components(state, player)

        return {
            "my_ds": my_ds,
            "my_de": my_de,
            "opp_ds": opp_ds,
            "opp_de": opp_de,
            "my_comp": my_comp,
            "my_sizes": my_sizes,
        }

    def _features(self, state, action, player, cache):
        size = state.size
        opp = 1 - player

        if action == state.swap_action:
            return np.zeros(12, dtype=np.float32)

        r, c = action // size, action % size

        # --- center ---
        max_center = 2 * (size - 1)
        center_dist = abs(r - size // 2) + abs(c - size // 2)
        center_score = 1.0 - center_dist / max_center

        # --- paths ---
        norm = 2 * size
        my_path = cache["my_ds"][action] + cache["my_de"][action]
        opp_path = cache["opp_ds"][action] + cache["opp_de"][action]

        # --- neighbors ---
        neigh = state._get_neighbors(action)
        friend = [n for n in neigh if state.board[n] == player]
        enemy = [n for n in neigh if state.board[n] == opp]
        empty = [n for n in neigh if state.board[n] == -1]

        # --- components ---
        comp_id = cache["my_comp"]
        comp_sizes = cache["my_sizes"]

        touched = set(comp_id[n] for n in friend if comp_id[n] != -1)
        comp_size = 1 + sum(comp_sizes[c] for c in touched)

        return np.array([
            1 - min(my_path, norm * 2) / (norm * 2),
            1 - min(opp_path, norm * 2) / (norm * 2),
            (opp_path - my_path) / norm,
            center_score,
            len(friend) / 6.0,
            len(enemy) / 6.0,
            len(touched) / 6.0,
            comp_size / float(self.N_CELLS),
            self._bridge(state, action, player) / 6.0,
            self._bridge(state, action, opp) / 6.0,
            len(empty) / 6.0,
            0.0,
        ], dtype=np.float32)

class HeuristicAgent:
    def step(self, state, p=None, rot=None):
        legal = state.legal_actions()
        if not legal: return None
        
        # 0. ZASADA OTWARCIA (75% uczciwe, 25% agresywny środek)
        if state.moves_played == 0:
            if random.random() < 0.25:
                # Losuje dowolne pole w promieniu 2 kratek od idealnego środka
                center = state.size // 2
                center_moves = []
                for r in range(center - 2, center + 3):
                    for c in range(center - 2, center + 3):
                        move = r * state.size + c
                        if move in legal:
                            center_moves.append(move)
                
                if center_moves:
                    return random.choice(center_moves)
            
            # 75% szans na zbalansowane FAIR_OPENINGS z książki
            valid_openings = [m for m in FAIR_OPENINGS if m in legal]
            if valid_openings:
                return random.choice(valid_openings)
        
        if state.moves_played == 1 and state.swap_action in legal:
            r = state.first_move_action // state.size
            c = state.first_move_action % state.size
            center = state.size // 2
            if abs(r - center) <= 2 and abs(c - center) <= 2:
                return state.swap_action
                
        player = state.current_player()
        size = state.size
        
        dist_start = self._get_distances(state, player, is_start_edge=True)
        dist_end = self._get_distances(state, player, is_start_edge=False)
        
        best_score = float('inf')
        best_actions = []
        
        for action in legal:
            if action == state.swap_action: continue
                
            path_length = dist_start[action] + dist_end[action]
            r, c = action // size, action % size
            center_dist = abs(r - size//2) + abs(c - size//2)
            final_score = path_length + (center_dist * 0.01)
            
            if final_score < best_score:
                best_score = final_score
                best_actions = [action]
            elif final_score == best_score:
                best_actions.append(action)
                
        if best_actions:
            return random.choice(best_actions)
        return random.choice(legal)

    def _get_distances(self, state, player, is_start_edge):
        size = state.size
        dist = [float('inf')] * (size * size)
        q = collections.deque()
        
        for i in range(size):
            if player == 0: 
                node = i if is_start_edge else (size - 1) * size + i
            else:           
                node = i * size if is_start_edge else i * size + (size - 1)
                
            if state.board[node] == player:
                dist[node] = 0
                q.appendleft(node)  
            elif state.board[node] == -1:
                dist[node] = 1
                q.append(node)      
                
        while q:
            curr = q.popleft()
            curr_d = dist[curr]
            
            for n in state._get_neighbors(curr):
                if state.board[n] == 1 - player: continue
                cost = 0 if state.board[n] == player else 1
                if curr_d + cost < dist[n]:
                    dist[n] = curr_d + cost
                    if cost == 0: q.appendleft(n)
                    else: q.append(n)
        return dist

class HeuristicAgent2:
    def step(self, state, p=None, rot=None):
        legal = state.legal_actions()
        if not legal: return None
        
        # 0. ZASADA OTWARCIA (75% uczciwe, 25% agresywny środek)
        if state.moves_played == 0:
            if random.random() < 0.25:
                # Losuje dowolne pole w promieniu 2 kratek od idealnego środka
                center = state.size // 2
                center_moves = []
                for r in range(center - 2, center + 3):
                    for c in range(center - 2, center + 3):
                        move = r * state.size + c
                        if move in legal:
                            center_moves.append(move)
                
                if center_moves:
                    return random.choice(center_moves)
            
            # 75% szans na zbalansowane FAIR_OPENINGS z książki
            valid_openings = [m for m in FAIR_OPENINGS if m in legal]
            if valid_openings:
                return random.choice(valid_openings)
        
        if state.moves_played == 1 and state.swap_action in legal:
            r = state.first_move_action // state.size
            c = state.first_move_action % state.size
            center = state.size // 2
            if abs(r - center) <= 2 and abs(c - center) <= 2:
                return state.swap_action
                
        player = state.current_player()
        opponent = 1 - player
        size = state.size
        
        bot_dist_start = self._get_distances(state, player, is_start_edge=True)
        bot_dist_end = self._get_distances(state, player, is_start_edge=False)
        
        opp_dist_start = self._get_distances(state, opponent, is_start_edge=True)
        opp_dist_end = self._get_distances(state, opponent, is_start_edge=False)
        
        best_score = None
        best_actions = []
        
        for action in legal:
            if action == state.swap_action: continue
                
            bot_path = bot_dist_start[action] + bot_dist_end[action]
            opp_path = opp_dist_start[action] + opp_dist_end[action]
            
            critical_score = min(bot_path, opp_path + 0.1)
            total_paths = bot_path + opp_path
            
            r, c = action // size, action % size
            center_dist = abs(r - size//2) + abs(c - size//2)
            
            score_tuple = (critical_score, total_paths, center_dist)
            
            if best_score is None or score_tuple < best_score:
                best_score = score_tuple
                best_actions = [action]
            elif score_tuple == best_score:
                best_actions.append(action)
                
        if best_actions:
            return random.choice(best_actions)
        return random.choice(legal)

    def _get_distances(self, state, player, is_start_edge):
        size = state.size
        dist = [float('inf')] * (size * size)
        q = collections.deque()
        
        for i in range(size):
            if player == 0: 
                node = i if is_start_edge else (size - 1) * size + i
            else:           
                node = i * size if is_start_edge else i * size + (size - 1)
                
            if state.board[node] == player:
                dist[node] = 0
                q.appendleft(node)  
            elif state.board[node] == -1:
                dist[node] = 1
                q.append(node)      
                
        while q:
            curr = q.popleft()
            curr_d = dist[curr]
            
            for n in state._get_neighbors(curr):
                if state.board[n] == 1 - player: continue 
                cost = 0 if state.board[n] == player else 1
                if curr_d + cost < dist[n]:
                    dist[n] = curr_d + cost
                    if cost == 0: q.appendleft(n)
                    else: q.append(n)
        return dist
    
class ValueMapGUI(tk.Frame):
    def __init__(self, master, agent, agent_name, board_size=9):
        super().__init__(master)
        self.pack()
        self.size = board_size
        self.agent = agent
        self.agent_name = agent_name

        self.info_label = tk.Label(self, text=f"Generowanie mapy WARTOŚCI dla: {agent_name}...", font=("Arial", 16, "bold"))
        self.info_label.pack(pady=10)

        self.canvas = tk.Canvas(self, width=800, height=600)
        self.canvas.pack()

        self.btn_back = tk.Button(self, text="⬅ Powrót do Menu", font=("Arial", 12), command=self.go_back)
        self.btn_back.pack(pady=10)

        self.after(50, self.generate_map)

    def generate_map(self):
        self.values = {}
        state = HexState(self.size)
        player = 0 # Oceniamy pustą planszę jako Czarny

        # Sprawdzamy, czy agent to nasz bot heurystyczny z wagami
        if hasattr(self.agent, "WEIGHTS") and hasattr(self.agent, "_features"):
            cache = self.agent._cache(state, player)
            for action in range(self.size * self.size):
                feat = self.agent._features(state, action, player, cache)
                score = float(np.dot(self.agent.WEIGHTS, feat))
                self.values[action] = score
        else:
            self.info_label.config(text=f"Błąd: Mapa WARTOŚCI działa tylko dla agentów CMA-ES!", fg="red")
            return

        self.info_label.config(text=f"Wartość pól (Score) na start dla: {self.agent_name}")
        self.draw_heatmap()

    def draw_heatmap(self):
        vals = list(self.values.values())
        min_v = min(vals)
        max_v = max(vals)
        rng = max_v - min_v if max_v != min_v else 1.0

        size = 25
        w = math.sqrt(3) * size
        h = 2 * size

        for r in range(self.size):
            for c in range(self.size):
                cx = 100 + c * w + r * (w / 2)
                cy = 50 + r * (h * 0.75)

                pts = []
                for i in range(6):
                    angle = math.pi / 180 * (60 * i - 30)
                    pts += [cx + size * math.cos(angle), cy + size * math.sin(angle)]

                action = r * self.size + c
                val = self.values[action]
                
                norm = (val - min_v) / rng
                
                red = 255
                green = int(255 * (1 - norm))
                blue = int(255 * (1 - norm))
                color = f"#{red:02x}{green:02x}{blue:02x}"

                self.canvas.create_polygon(pts, fill=color, outline="gray")
                
                text_color = "white" if norm > 0.5 else "black"
                label_text = f"{action}\n{val:.2f}"
                self.canvas.create_text(cx, cy, text=label_text, fill=text_color, font=("Arial", 8, "bold"), justify=tk.CENTER)

    def go_back(self):
        self.destroy()
        MainMenu(self.master)
    
class FirstMoveMapGUI(tk.Frame):
    def __init__(self, master, agent, agent_name, board_size=9, samples=100):
        super().__init__(master)
        self.pack()
        self.size = board_size
        self.agent = agent
        self.samples = samples
        self.agent_name = agent_name

        self.info_label = tk.Label(self, text=f"Generowanie mapy OTWARCIA dla: {agent_name}...", font=("Arial", 16, "bold"))
        self.info_label.pack(pady=10)

        self.canvas = tk.Canvas(self, width=800, height=600)
        self.canvas.pack()

        self.btn_back = tk.Button(self, text="⬅ Powrót do Menu", font=("Arial", 12), command=self.go_back)
        self.btn_back.pack(pady=10)

        self.after(50, self.generate_map)

    def generate_map(self):
        self.probs = {i: 0 for i in range(self.size * self.size)}
        
        for _ in range(self.samples):
            # Zawsze pusta plansza
            state = HexState(self.size)
            rot = random.choice([True, False])
            
            action = self.agent.step(state, p=0, rot=rot)
            
            if action is not None and action != state.swap_action:
                self.probs[action] += 1
        
        # Przeliczamy zliczenia na procenty
        for i in self.probs:
            self.probs[i] = self.probs[i] / self.samples

        self.info_label.config(text=f"Mapa prawdopodobieństwa OTWARCIA: {self.agent_name}")
        self.draw_heatmap()

    def get_color(self, prob):
        if prob == 0:
            return "#ffffff" # Biały dla zera
            
        intensity = min(1.0, prob * 1.5) 
        r = 255
        g = int(255 * (1 - intensity))
        b = int(255 * (1 - intensity))
        return f"#{r:02x}{g:02x}{b:02x}"

    def draw_heatmap(self):
        size = 25
        w = math.sqrt(3) * size
        h = 2 * size

        for r in range(self.size):
            for c in range(self.size):
                cx = 100 + c * w + r * (w / 2)
                cy = 50 + r * (h * 0.75)

                pts = []
                for i in range(6):
                    angle = math.pi / 180 * (60 * i - 30)
                    pts += [cx + size * math.cos(angle), cy + size * math.sin(angle)]

                action = r * self.size + c
                prob = self.probs[action]
                color = self.get_color(prob)

                self.canvas.create_polygon(pts, fill=color, outline="gray")
                
                text_color = "white" if prob > 0.4 else "black"
                label_text = f"{action}\n{int(prob*100)}%"
                self.canvas.create_text(cx, cy, text=label_text, fill=text_color, font=("Arial", 8, "bold"), justify=tk.CENTER)

    def go_back(self):
        self.destroy()
        MainMenu(self.master)    

class SwapMapGUI(tk.Frame):
    def __init__(self, master, agent, agent_name, board_size=9, samples=20):
        super().__init__(master)
        self.pack()
        self.size = board_size
        self.agent = agent
        self.samples = samples
        self.agent_name = agent_name

        self.info_label = tk.Label(self, text=f"Generowanie mapy SWAP dla: {agent_name}...", font=("Arial", 16, "bold"))
        self.info_label.pack(pady=10)

        self.canvas = tk.Canvas(self, width=800, height=600)
        self.canvas.pack()

        self.btn_back = tk.Button(self, text="⬅ Powrót do Menu", font=("Arial", 12), command=self.go_back)
        self.btn_back.pack(pady=10)

        self.after(50, self.generate_map)

    def generate_map(self):
        self.probs = {}
        for first_move in range(self.size * self.size):
            swaps = 0
            for _ in range(self.samples):
                state = HexState(self.size)
                state.apply_action(first_move) # Gracz 0 robi ruch
                
                rot = random.choice([True, False])
                # Pytamy agenta (grającego jako Gracz 1) o ruch
                action = self.agent.step(state, p=1, rot=rot)
                
                if action == state.swap_action:
                    swaps += 1
            
            self.probs[first_move] = swaps / self.samples

        self.info_label.config(text=f"Mapa prawdopodobieństwa SWAP: {self.agent_name}")
        self.draw_heatmap()

    def get_color(self, prob):
        # Od białego (0%) do czerwonego (100%)
        r = 255
        g = int(255 * (1 - prob))
        b = int(255 * (1 - prob))
        return f"#{r:02x}{g:02x}{b:02x}"

    def draw_heatmap(self):
        size = 25
        w = math.sqrt(3) * size
        h = 2 * size

        for r in range(self.size):
            for c in range(self.size):
                cx = 100 + c * w + r * (w / 2)
                cy = 50 + r * (h * 0.75)

                pts = []
                for i in range(6):
                    angle = math.pi / 180 * (60 * i - 30)
                    pts += [cx + size * math.cos(angle), cy + size * math.sin(angle)]

                action = r * self.size + c
                prob = self.probs[action]
                color = self.get_color(prob)

                self.canvas.create_polygon(pts, fill=color, outline="gray")
                

                text_color = "white" if prob > 0.6 else "black"
                label_text = f"{action}\n{int(prob*100)}%"
                self.canvas.create_text(cx, cy, text=label_text, fill=text_color, font=("Arial", 8, "bold"), justify=tk.CENTER)

    def go_back(self):
        self.destroy()
        MainMenu(self.master)
        
class HexGameGUI(tk.Frame):
    def __init__(self, master, agent0, agent1, board_size=9, delay_ms=500):
        super().__init__(master)
        self.pack()
        self.size = board_size
        self.state = HexState(board_size)
        
        # None oznacza człowieka
        self.agents = {0: agent0, 1: agent1}
        self.delay_ms = delay_ms
        self.rot = np.random.choice([True, False])

        self.info_label = tk.Label(self, font=("Arial", 14))
        self.info_label.pack(pady=10)

        # SWAP
        self.swap_btn = tk.Button(self, text="🔄 SWAP", command=self.play_swap,
                                 state=tk.DISABLED, bg="orange", font=("Arial", 12, "bold"))
        self.swap_btn.pack(pady=5)

        self.canvas = tk.Canvas(self, width=800, height=600)
        self.canvas.pack()

        self.draw_board()
        self.check_turn()

    def draw_board(self):
        self.canvas.delete("all")
        self.hex_ids = {}

        R = 25
        dx = math.sqrt(3) * R
        dy = 1.5 * R

        ox = 120
        oy = 60

        def center(r, c):
            return ox + c * dx + r * (dx / 2), oy + r * dy

        def hex_vertices(cx, cy):
            pts = []
            for i in range(6):
                angle = math.radians(60 * i - 30)
                pts.append((cx + R * math.cos(angle), cy + R * math.sin(angle)))
            return pts

        n = self.size

        # --- 1. NAJPIERW RYSUJEMY WSZYSTKIE HEKSAGONY ---
        for r in range(n):
            for c in range(n):
                cx, cy = center(r, c)
                pts = hex_vertices(cx, cy)
                flat = [v for p in pts for v in p]

                action = r * n + c
                pid = self.canvas.create_polygon(
                    flat,
                    fill="lightgray",
                    outline="gray",
                    width=1
                )
                self.hex_ids[action] = pid
                self.canvas.tag_bind(pid, "<Button-1>", lambda e, a=action: self.click(a))

        # --- 2. RYSOWANIE ZĄBKOWANYCH KRAWĘDZI NA WIERZCHU ---
        border_w = 6  # Grubość kolorowej ramki

        def draw_edge_line(r, c, v_start, v_end, color):
            # Funkcja pomocnicza rysująca linię między dwoma wierzchołkami heksa
            cx, cy = center(r, c)
            pts = hex_vertices(cx, cy)
            p1 = pts[v_start]
            p2 = pts[v_end]
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1],
                                    fill=color, width=border_w, capstyle=tk.ROUND)

        # Czarne krawędzie (Gracz Czarny: Góra i Dół)
        for c in range(n):
            # Górny rząd (wierzchołki 4-5 i 5-0)
            draw_edge_line(0, c, 4, 5, "black")
            draw_edge_line(0, c, 5, 0, "black")
            
            # Dolny rząd (wierzchołki 1-2 i 2-3)
            draw_edge_line(n - 1, c, 1, 2, "black")
            draw_edge_line(n - 1, c, 2, 3, "black")

        for r in range(n):
            draw_edge_line(r, 0, 3, 4, "red")
            if r < n - 1:
                draw_edge_line(r, 0, 2, 3, "red")
            
            draw_edge_line(r, n - 1, 0, 1, "red")
            if r > 0:
                draw_edge_line(r, n - 1, 5, 0, "red")

    def update_board(self):
        for i in range(self.size * self.size):
            val = self.state.board[i]
            color = "lightgray"
            if val == 0:
                color = "black"
            elif val == 1:
                color = "red"

            self.canvas.itemconfig(self.hex_ids[i], fill=color)

        cp = self.state.current_player()
        color_name = "CZARNY (Góra-Dół)" if cp == 0 else "CZERWONY (Lewo-Prawo)"
        player_type = "Człowiek" if self.agents[cp] is None else "Bot"
        self.info_label.config(text=f"Tura: {color_name} [{player_type}]")

    def click(self, action):
        cp = self.state.current_player()
        if self.agents[cp] is not None:
            return
        if action not in self.state.legal_actions():
            return

        self.state.apply_action(action)
        self.check_turn()

    def play_swap(self):
        cp = self.state.current_player()
        if self.agents[cp] is not None:
            return
        if self.state.swap_action not in self.state.legal_actions():
            return

        self.state.apply_action(self.state.swap_action)
        self.check_turn()

    def check_turn(self):
        self.update_board()

        cp = self.state.current_player()
        
        # SWAP aktywacja tylko dla człowieka
        if self.state.moves_played == 1 and self.agents[cp] is None:
            self.swap_btn.config(state=tk.NORMAL)
        else:
            self.swap_btn.config(state=tk.DISABLED)

        if self.state.is_terminal():
            winner = "CZARNY" if self.state.returns()[0] == 1 else "CZERWONY"
            self.info_label.config(text=f"KONIEC GRY! Wygrywa {winner}", fg="blue")
            print("KONIEC:", self.state.returns())
            return

        # Jeśli kolej bota, zaplanuj jego ruch
        if self.agents[cp] is not None:
            self.after(self.delay_ms, self.bot_move)

    def bot_move(self):
        if self.state.is_terminal(): return
        
        cp = self.state.current_player()
        bot = self.agents[cp]
        
        a = bot.step(self.state, cp, self.rot) 
        self.state.apply_action(a)
        self.check_turn()


# MENU GŁÓWNE I TURNIEJ W TLE
class MainMenu(tk.Frame):
    def __init__(self, master, checkpoint_dir="league_checkpoints"):
        super().__init__(master)
        self.master = master
        self.checkpoint_dir = checkpoint_dir
        self.pack(pady=50)

        tk.Label(self, text="HEX 9x9", font=("Arial", 20, "bold")).pack(pady=20)

        # Wybór trybu
        self.mode_var = tk.StringVar(value="Human_vs_Bot")
        tk.Radiobutton(self, text="Zagraj z wybranym botem", variable=self.mode_var, value="Human_vs_Bot", command=self.update_ui, font=("Arial", 12)).pack()
        tk.Radiobutton(self, text="Bot vs Bot (Oglądaj mecz)", variable=self.mode_var, value="Bot_vs_Bot", command=self.update_ui, font=("Arial", 12)).pack()
        tk.Radiobutton(self, text="Turniej wszystkich botów w tle", variable=self.mode_var, value="Background_Tournament", command=self.update_ui, font=("Arial", 12, "bold"), fg="blue").pack()
        tk.Radiobutton(self, text="Analiza SWAP (Mapa cieplna otwarć)", variable=self.mode_var, value="Swap_Map", command=self.update_ui, font=("Arial", 12, "bold"), fg="purple").pack()
        tk.Radiobutton(self, text="Analiza OTWARCIA (Gdzie zagra na pustej planszy)", variable=self.mode_var, value="First_Move_Map", command=self.update_ui, font=("Arial", 12, "bold"), fg="darkgreen").pack()
        tk.Radiobutton(self, text="Analiza WARTOŚCI (Oceny pól dla botów CMA-ES)", variable=self.mode_var, value="Value_Map", command=self.update_ui, font=("Arial", 12, "bold"), fg="brown").pack()
        # Dynamiczne wczytywanie dostępnych botów
        self.bot_options = ["Heuristic Agent", "CMA-ES", "Blind Heuristic Agent"]
        
        # Skanowanie folderu w poszukiwaniu modeli PPO (.zip)
        if os.path.exists(self.checkpoint_dir):
            ppo_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".zip")]
            # Sortowanie alfabetyczne/numeryczne (żeby kroki były po kolei)
            ppo_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
            
            for f in ppo_files:
                self.bot_options.append(f"PPO: {f}")
        else:
            print(f"Brak folderu {self.checkpoint_dir}. Boty PPO nie zostaną załadowane.")

        # Zabezpieczenie, jeśli w ogóle nie ma botów
        if not self.bot_options:
            self.bot_options = ["Brak botów"]

        # Ramka wyboru botów (dla trybów z GUI)
        self.frame_bots = tk.Frame(self)
        self.frame_bots.pack(pady=20)

        self.lbl_bot1 = tk.Label(self.frame_bots, text="Bot 1 (Czarny):")
        self.lbl_bot1.grid(row=0, column=0, padx=10)
        self.bot1_var = tk.StringVar(value=self.bot_options[0])
        self.bot1_menu = tk.OptionMenu(self.frame_bots, self.bot1_var, *self.bot_options)
        self.bot1_menu.grid(row=0, column=1)

        self.lbl_bot2 = tk.Label(self.frame_bots, text="Bot 2 (Czerwony):")
        self.lbl_bot2.grid(row=1, column=0, padx=10, pady=10)
        
        # Domyślnie ustaw drugi na liście jako Czerwony (jeśli istnieje)
        default_bot2 = self.bot_options[1] if len(self.bot_options) > 1 else self.bot_options[0]
        self.bot2_var = tk.StringVar(value=default_bot2)
        self.bot2_menu = tk.OptionMenu(self.frame_bots, self.bot2_var, *self.bot_options)
        self.bot2_menu.grid(row=1, column=1)

        # Ramka dla turnieju w tle
        self.frame_tourney = tk.Frame(self)
        tk.Label(self.frame_tourney, text="Ile meczów w każdej parze (np. A vs B)?", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.games_var = tk.IntVar(value=10)
        tk.Entry(self.frame_tourney, textvariable=self.games_var, width=5, font=("Arial", 10)).pack(side=tk.LEFT)

        self.btn_start = tk.Button(self, text="▶ ROZPOCZNIJ", font=("Arial", 14, "bold"), bg="green", fg="white", command=self.start_game)
        self.btn_start.pack(pady=20)
        
        self.info_lbl = tk.Label(self, text="", font=("Arial", 12), fg="red")
        self.info_lbl.pack()

        self.update_ui()

    def update_ui(self):
        mode = self.mode_var.get()
        self.info_lbl.config(text="")
        
        if mode == "Human_vs_Bot":
            self.frame_bots.pack(pady=20)
            self.frame_tourney.pack_forget()
            self.bot1_menu.config(state=tk.NORMAL)
            self.bot2_menu.config(state=tk.DISABLED)
            self.lbl_bot1.config(text="Wybór bota przeciwnika:")
            self.lbl_bot2.config(text="(Ignorowane)")
            self.btn_start.config(state=tk.NORMAL, text="▶ GRAJ Z BOTEM")
            
        elif mode == "Bot_vs_Bot":
            self.frame_bots.pack(pady=20)
            self.frame_tourney.pack_forget()
            self.bot1_menu.config(state=tk.NORMAL)
            self.bot2_menu.config(state=tk.NORMAL)
            self.lbl_bot1.config(text="Bot 1 (Czarny):")
            self.lbl_bot2.config(text="Bot 2 (Czerwony):")
            self.btn_start.config(state=tk.NORMAL, text="▶ OGLĄDAJ MECZ")
            
        elif mode == "Background_Tournament":
            self.frame_bots.pack_forget()
            self.frame_tourney.pack(pady=20)
            self.btn_start.config(state=tk.NORMAL, text="▶ START TURNIEJU")
        
        elif mode == "Swap_Map":
            self.frame_bots.pack(pady=20)
            self.frame_tourney.pack_forget()
            self.bot1_menu.config(state=tk.NORMAL)
            self.bot2_menu.config(state=tk.DISABLED)
            self.lbl_bot1.config(text="Wybierz agenta do analizy:")
            self.lbl_bot2.config(text="(Ignorowane)")
            self.btn_start.config(state=tk.NORMAL, text="GENERUJ MAPĘ")
        
        elif mode in ["Swap_Map", "First_Move_Map", "Value_Map"]:
            self.frame_bots.pack(pady=20)
            self.frame_tourney.pack_forget()
            self.bot1_menu.config(state=tk.NORMAL)
            self.bot2_menu.config(state=tk.DISABLED)
            self.lbl_bot1.config(text="Wybierz agenta do analizy:")
            self.lbl_bot2.config(text="(Ignorowane)")
            self.btn_start.config(state=tk.NORMAL, text="GENERUJ MAPĘ")

    def _create_bot(self, bot_name):
        if bot_name.startswith("PPO: "):
            filename = bot_name.replace("PPO: ", "")
            path = os.path.join(self.checkpoint_dir, filename)
            agent = PPOAgent(path, board_size=9)
            return agent if agent.valid else None
        elif bot_name == "Heuristic Agent": 
            return HeuristicAgent2()
        elif bot_name == "Blind Heuristic Agent": return HeuristicAgent()
        elif bot_name == "CMA-ES": return LinearHexBotTest()
        return None

    def start_game(self):
        mode = self.mode_var.get()
        
        if mode == "Human_vs_Bot" or mode == "Bot_vs_Bot":
            if mode == "Human_vs_Bot":
                bot = self._create_bot(self.bot1_var.get())
                if random.choice([True, False]): agent0, agent1 = None, bot
                else: agent0, agent1 = bot, None
            else:
                agent0 = self._create_bot(self.bot1_var.get())
                agent1 = self._create_bot(self.bot2_var.get())

            self.destroy()
            HexGameGUI(self.master, agent0, agent1, board_size=9, delay_ms=2000)
            
        elif mode == "Background_Tournament":
            self.btn_start.config(state=tk.DISABLED, text="⏳ TRWA SYMULACJA...")
            self.info_lbl.config(text="Rozgrywki w tle... (to może chwilę potrwać)...")
            threading.Thread(target=self.run_tournament_thread, daemon=True).start()
            
        elif mode == "Swap_Map":
            agent_name = self.bot1_var.get()
            agent = self._create_bot(agent_name)
            self.destroy()
            SwapMapGUI(self.master, agent, agent_name, board_size=9, samples=100)
            
        elif mode == "First_Move_Map":
            agent_name = self.bot1_var.get()
            agent = self._create_bot(agent_name)
            self.destroy()
            FirstMoveMapGUI(self.master, agent, agent_name, board_size=9, samples=1000)
            
        elif mode == "Value_Map":
            agent_name = self.bot1_var.get()
            agent = self._create_bot(agent_name)
            self.destroy()
            ValueMapGUI(self.master, agent, agent_name, board_size=9)

    def run_tournament_thread(self):
        # Automatycznie ładujemy wszystkie boty z listy rozwijanej
        bot_dict = {}
        for bot_name in self.bot_options:
            agent = self._create_bot(bot_name)
            if agent is not None:
                bot_dict[bot_name] = agent
                
        bot_names = list(bot_dict.keys())
        games_per_pair = self.games_var.get()
        
        # Zmodyfikowana struktura wyników
        results = {
            name: {
                "W": 0, "L": 0, "P": 0,
                "H2H": {opp: {"W": 0, "L": 0} for opp in bot_names if opp != name}
            } for name in bot_names
        }

        # Każdy z każdym
        for i in range(len(bot_names)):
            for j in range(len(bot_names)):
                if i == j: continue
                
                name1, name2 = bot_names[i], bot_names[j]
                bot1, bot2 = bot_dict[name1], bot_dict[name2]

                for _ in range(games_per_pair):
                    state = HexState(9)
                    rot = random.choice([True, False])
                    
                    while not state.is_terminal():
                        cp = state.current_player()
                        if cp == 0: action = bot1.step(state, p=cp, rot=rot)
                        else:       action = bot2.step(state, p=cp, rot=rot)
                        state.apply_action(action)
                        
                    if state.returns()[0] == 1:
                        # Wygrana Bota 1 (name1)
                        results[name1]["W"] += 1
                        results[name2]["L"] += 1
                        results[name1]["H2H"][name2]["W"] += 1
                        results[name2]["H2H"][name1]["L"] += 1
                    else:
                        # Wygrana Bota 2 (name2)
                        results[name2]["W"] += 1
                        results[name1]["L"] += 1
                        results[name2]["H2H"][name1]["W"] += 1
                        results[name1]["H2H"][name2]["L"] += 1
                        
                    results[name1]["P"] += 1
                    results[name2]["P"] += 1

        # Powrót do głównego wątku, by wyświetlić tabelę
        self.master.after(0, lambda: self.show_tournament_results(results))

    def show_tournament_results(self, results):
        self.destroy() 
        
        sorted_results = []
        for name, stats in results.items():
            win_rate = (stats["W"] / stats["P"] * 100) if stats["P"] > 0 else 0
            short_name = name.replace("PPO: ", "").replace(".zip", "")
            # Zapisujemy pełną nazwę, żeby mieć dostęp do statystyk H2H
            sorted_results.append((short_name, name, stats["W"], stats["L"], stats["P"], round(win_rate, 2), stats["H2H"]))
            
        # Sortujemy po liczbie wygranych (indeks 2 to wygrane)
        sorted_results.sort(key=lambda x: x[2], reverse=True) 

        frame = tk.Frame(self.master)
        frame.pack(pady=20, fill=tk.BOTH, expand=True)
        
        tk.Label(frame, text="WYNIKI TURNIEJU", font=("Arial", 20, "bold")).pack(pady=10)

        # Używamy show="tree headings", żeby pokazać kolumnę z rozwijanym drzewkiem
        columns = ("Bot", "Wygrane", "Przegrane", "Rozegrane", "Win Rate (%)")
        tree = ttk.Treeview(frame, columns=columns, show="tree headings", height=15)
        
        # Konfiguracja głównej kolumny drzewa
        tree.heading("#0", text="Miejsce / Przeciwnik")
        tree.column("#0", width=150, anchor=tk.W)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor=tk.CENTER, width=120)

        # Wstawianie danych do tabeli
        for idx, row_data in enumerate(sorted_results):
            short_name, full_name, w, l, p, wr, h2h = row_data
            
            # Wiersz główny (rodzic)
            parent_id = tree.insert("", tk.END, text=f" #{idx + 1}", values=(short_name, w, l, p, f"{wr}%"))
            
            # Podrzędne wiersze H2H (dzieci)
            for opp_full, opp_stats in h2h.items():
                opp_short = opp_full.replace("PPO: ", "").replace(".zip", "")
                h2h_w = opp_stats["W"]
                h2h_l = opp_stats["L"]
                h2h_p = h2h_w + h2h_l
                
                # Zabezpieczenie na wypadek braku gier (choć w tym trybie raczej będą)
                if h2h_p > 0:
                    h2h_wr = round((h2h_w / h2h_p) * 100, 2)
                    
                    # Wstawiamy jako dziecko parent_id. 
                    tree.insert(parent_id, tk.END, text=f"  vs {opp_short}", values=("", h2h_w, h2h_l, h2h_p, f"{h2h_wr}%"))
            
        tree.pack(pady=10, fill=tk.BOTH, expand=True)
        tk.Button(frame, text="Zakończ", font=("Arial", 12), command=self.master.quit).pack(pady=10)


# START
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Hex Engine")
    root.geometry("850x700")

    app = MainMenu(root, checkpoint_dir="league_checkpoints")
    root.mainloop()