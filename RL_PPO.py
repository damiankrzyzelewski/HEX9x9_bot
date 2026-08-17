import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import concurrent.futures

from torch.utils.data import Dataset, DataLoader
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# KONFIGURACJA
BOARD_SIZE = 9
CHECKPOINT_DIR = "./league_checkpoints/"
BC_PATH = "bc_resnet_deep_9x9.pth"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def generate_fair_openings(size):
    openings = []
    center = size // 2
    
    # Przeszukujemy planszę, omijając skrajne krawędzie
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            # Liczymy dystans od środka (dystans Manhattan)
            dist = abs(r - center) + abs(c - center)
            
            # Bierzemy pola, które są oddalone od środka o 2 lub 3 kratki
            if dist == 2 or dist == 3:
                openings.append(r * size + c)
                
    return openings

FAIR_OPENINGS = generate_fair_openings(BOARD_SIZE)
print(f"Wygenerowano {len(FAIR_OPENINGS)} uczciwych otwarć dla planszy {BOARD_SIZE}x{BOARD_SIZE}.")

class LinearSchedule:
    def __init__(self, initial_value, min_lr=1e-6, start_decay_at=0.9):
        self.initial_value = initial_value
        self.min_lr = min_lr
        self.start_decay_at = start_decay_at

    def __call__(self, progress_remaining):
        # Faza stałego LR
        if progress_remaining >= self.start_decay_at:
            return self.initial_value
        
        # Faza spadku
        scale = progress_remaining / self.start_decay_at
        return self.min_lr + scale * (self.initial_value - self.min_lr)

# 1. SILNIK GRY
class HexState:
    def __init__(self, board_size=11):
        self.size = board_size
        self.board = [-1] * (board_size * board_size)
        self._current_player = 0
        self._is_terminal = False
        self.moves_played = 0
        self.first_move_action = None
        self.swap_action = board_size * board_size

    def current_player(self): return self._current_player
    def is_terminal(self): return self._is_terminal

    def legal_actions(self):
        actions = [i for i, s in enumerate(self.board) if s == -1]
        if self.moves_played == 1:
            actions.append(self.swap_action)
        return actions

    def apply_action(self, action):
        if self._is_terminal:
            return

        if action == self.swap_action:
            if self.moves_played != 1:
                return
            # zmiana właściciela pierwszego ruchu
            self.board[self.first_move_action] = 1
            self._current_player = 0
            self.moves_played += 1
            return

        if self.board[action] != -1:
            raise ValueError(f"Illegal move: {action}")

        self.board[action] = self._current_player

        if self.moves_played == 0:
            self.first_move_action = action

        if self._check_win(self._current_player):
            self._is_terminal = True
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

# 2. ARCHITEKTURA (RESNET 128)
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
    def forward(self, x): return self.lin(self.net(x))

class BCModel(nn.Module):
    def __init__(self, size=BOARD_SIZE):
        super().__init__()
        self.ex = HexCNN(spaces.Box(0, 1, (3, size, size)), 256)
        self.head = nn.Linear(256, size * size + 1)
    def forward(self, x): return self.head(self.ex(x))


# 3. AGENCI HEURYSTYCZNI
import collections
import random

class RandomAgent:
    def __init__(self):
        # Powołujemy do życia ukrytego w nim "mądrego" bota
        self.smart_bot = HeuristicAgent2()

    def step(self, state):
        legal = state.legal_actions()
        if not legal: return None
        
        # Raz na 4 ruchy (25% szans) budzi się w nim mistrz
        if random.random() < 0.25:
            return self.smart_bot.step(state)
        
        # W pozostałych 75% przypadków gra całkowicie losowo
        return random.choice(legal)

class HeuristicAgent:
    def step(self, state):
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
            
            # 75% szans na AIR_OPENINGS
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
    def step(self, state):
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
            
            # 75% szans na FAIR_OPENINGS
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
    
class HeuristicAgent3:
    def step(self, state):
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
            
            # 75% szans na FAIR_OPENINGS
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

        for action in legal:
            if action == state.swap_action: continue
            test = self._copy_state(state)
            test.apply_action(action)
            if test.is_terminal(): return action

        for action in legal:
            if action == state.swap_action: continue
            test = self._copy_state(state)
            test._current_player = opponent
            test.apply_action(action)
            if test.is_terminal(): return action

        bot_start = self._get_distances(state, player, True)
        bot_end = self._get_distances(state, player, False)
        opp_start = self._get_distances(state, opponent, True)
        opp_end = self._get_distances(state, opponent, False)

        best_score = float("inf")
        best_actions = []

        for action in legal:
            if action == state.swap_action: continue
            r = action // size
            c = action % size
            bot_path = bot_start[action] + bot_end[action]
            opp_path = opp_start[action] + opp_end[action]
            connect_bonus = 0
            for n in state._get_neighbors(action):
                if state.board[n] == player:
                    connect_bonus -= 0.4
            center_dist = abs(r - size//2) + abs(c - size//2)
            center_bonus = center_dist * 0.05
            score = (min(bot_path, opp_path + 0.2) + 0.5 * bot_path + 0.7 * opp_path + center_bonus + connect_bonus)

            if score < best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        if best_actions: return random.choice(best_actions)
        return random.choice(legal)

    def _copy_state(self, state):
        new = state.__class__(state.size)
        new.board = state.board.copy()
        new._current_player = state._current_player
        new.moves_played = state.moves_played
        new.first_move_action = state.first_move_action
        return new

    def _get_distances(self, state, player, is_start_edge):
        size = state.size
        dist = [float('inf')] * (size * size)
        q = collections.deque()
        for i in range(size):
            if player == 0: node = i if is_start_edge else (size - 1) * size + i
            else: node = i * size if is_start_edge else i * size + (size - 1)
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

class HeuristicAgent4:
    def step(self, state):
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
            
            # 75% szans na FAIR_OPENINGS
            valid_openings = [m for m in FAIR_OPENINGS if m in legal]
            if valid_openings:
                return random.choice(valid_openings)
            
        player = state.current_player()
        opponent = 1 - player
        size = state.size

        if state.moves_played == 1 and state.swap_action in legal:
            r = state.first_move_action // state.size
            c = state.first_move_action % state.size
            center = state.size // 2
            if abs(r - center) <= 2 and abs(c - center) <= 2:
                return state.swap_action

        for action in legal:
            if action == state.swap_action: continue
            test = self._copy_state(state)
            test.apply_action(action)
            if test.is_terminal(): return action

        for action in legal:
            if action == state.swap_action: continue
            test = self._copy_state(state)
            test._current_player = opponent 
            test.apply_action(action)
            if test.is_terminal(): return action

        bot_start = self._get_distances(state, player, True)
        bot_end = self._get_distances(state, player, False)
        opp_start = self._get_distances(state, opponent, True)
        opp_end = self._get_distances(state, opponent, False)

        best_score = None
        best_actions = []

        for action in legal:
            if action == state.swap_action: continue

            bot_path = bot_start[action] + bot_end[action]
            opp_path = opp_start[action] + opp_end[action]
            r = action // size
            c = action % size
            center_dist = abs(r - size//2) + abs(c - size//2)
            critical_score = min(bot_path, opp_path) 
            total_paths = bot_path + opp_path
            
            score_tuple = (critical_score, bot_path, total_paths, center_dist)

            if best_score is None or score_tuple < best_score:
                best_score = score_tuple
                best_actions = [action]
            elif score_tuple == best_score:
                best_actions.append(action)

        if best_actions: return random.choice(best_actions)
        return random.choice(legal)

    def _copy_state(self, state):
        new = state.__class__(state.size)
        new.board = state.board.copy()
        new._current_player = state._current_player
        new.moves_played = state.moves_played
        new.first_move_action = state.first_move_action
        return new

    def _get_distances(self, state, player, is_start_edge):
        size = state.size
        dist = [float('inf')] * (size * size)
        q = collections.deque()
        for i in range(size):
            if player == 0: node = i if is_start_edge else (size - 1) * size + i
            else: node = i * size if is_start_edge else i * size + (size - 1)
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
    
class HeuristicAgent6:
    def step(self, state):
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
            
            # 75% szans na FAIR_OPENINGS
            valid_openings = [m for m in FAIR_OPENINGS if m in legal]
            if valid_openings:
                return random.choice(valid_openings)

        player = state.current_player()
        opponent = 1 - player
        size = state.size

        if state.moves_played == 1 and state.swap_action in legal:
            r = state.first_move_action // size
            c = state.first_move_action % size
            center = size // 2
            if abs(r - center) <= 2 and abs(c - center) <= 2:
                return state.swap_action

        for action in legal:
            if action == state.swap_action: continue
            test = self._copy_state(state)
            test.apply_action(action)
            if test.is_terminal(): return action

        for action in legal:
            if action == state.swap_action: continue
            test = self._copy_state(state)
            test._current_player = opponent 
            test.apply_action(action)
            if test.is_terminal(): return action

        bot_start = self._get_distances(state, player, True)
        bot_end = self._get_distances(state, player, False)
        opp_start = self._get_distances(state, opponent, True)
        opp_end = self._get_distances(state, opponent, False)

        best_score = None
        best_actions = []

        for action in legal:
            if action == state.swap_action: continue
            bot_path = bot_start[action] + bot_end[action]
            opp_path = opp_start[action] + opp_end[action]
            critical_score = min(bot_path, opp_path)
            total_paths = bot_path + opp_path
            
            neighbors = state._get_neighbors(action)
            friendly = [n for n in neighbors if state.board[n] == player]
            opp = [n for n in neighbors if state.board[n] == opponent]
            
            clumps = 0
            for i in range(len(friendly)):
                for j in range(i+1, len(friendly)):
                    if friendly[j] in state._get_neighbors(friendly[i]):
                        clumps += 1
                        
            opp_bridges_blocked = 0
            for i in range(len(opp)):
                for j in range(i+1, len(opp)):
                    if opp[j] not in state._get_neighbors(opp[i]):
                        opp_bridges_blocked += 1

            struct_eval = (clumps * 2) - (opp_bridges_blocked * 3)
            r = action // size
            c = action % size
            center_dist = abs(r - size//2) + abs(c - size//2)

            score_tuple = (critical_score, bot_path, total_paths, struct_eval, center_dist)

            if best_score is None or score_tuple < best_score:
                best_score = score_tuple
                best_actions = [action]
            elif score_tuple == best_score:
                best_actions.append(action)

        if best_actions: return random.choice(best_actions)
        return random.choice(legal)

    def _copy_state(self, state):
        new = state.__class__(state.size)
        new.board = state.board.copy()
        new._current_player = state._current_player
        new.moves_played = state.moves_played
        new.first_move_action = state.first_move_action
        return new

    def _get_distances(self, state, player, is_start_edge):
        size = state.size
        dist = [float('inf')] * (size * size)
        q = collections.deque()
        for i in range(size):
            if player == 0: node = i if is_start_edge else (size - 1) * size + i
            else: node = i * size if is_start_edge else i * size + (size - 1)
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
    
class HeuristicAgent7:
    def step(self, state):
        legal = state.legal_actions()
        if not legal: return None
        
        player = state.current_player()
        opponent = 1 - player
        size = state.size
        
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
            
            # 75% szans na FAIR_OPENINGS
            valid_openings = [m for m in FAIR_OPENINGS if m in legal]
            if valid_openings:
                return random.choice(valid_openings)
        
        # 1. ZASADA SWAP
        if state.moves_played == 1 and state.swap_action in legal:
            r = state.first_move_action // size
            c = state.first_move_action % size
            center = size // 2
            if abs(r - center) <= 2 and abs(c - center) <= 2:
                return state.swap_action

        # 2. BŁYSKAWICZNY ATAK / OBRONA (1-Step Lookahead)
        for action in legal:
            if action == state.swap_action: continue
            test = self._copy_state(state)
            test.apply_action(action)
            if test.is_terminal(): return action

        for action in legal:
            if action == state.swap_action: continue
            test = self._copy_state(state)
            test._current_player = opponent 
            test.apply_action(action)
            if test.is_terminal(): return action

        # 3. BFS - GLOBALNY WYŚCIG
        bot_start = self._get_distances(state, player, True)
        bot_end = self._get_distances(state, player, False)
        opp_start = self._get_distances(state, opponent, True)
        opp_end = self._get_distances(state, opponent, False)
        
        best_score = None
        best_actions = []

        for action in legal:
            if action == state.swap_action: continue
            
            bot_path = bot_start[action] + bot_end[action]
            opp_path = opp_start[action] + opp_end[action]
            
            # Sprawia, że woli atakować niż bronić przy remisie
            critical_score = min(bot_path, opp_path + 0.1)
            
            bridge_bonus = 0
            opp_block_bonus = 0
            action_neighbors = set(state._get_neighbors(action))
            
            # A. Czy mój ruch tworzy idealny most z moim innym klockiem?
            for n1 in action_neighbors:
                for n2 in state._get_neighbors(n1):
                    if n2 == action or n2 in action_neighbors: continue
                    if state.board[n2] == player:
                        # Sprawdzamy, czy mają dokładnie 2 puste pola styku
                        shared = action_neighbors.intersection(state._get_neighbors(n2))
                        if len(shared) == 2 and all(state.board[s] == -1 for s in shared):
                            bridge_bonus += 1

            # B. Czy ten ruch przecina wrogi most idealnie w połowie?
            opp_neighbors = [n for n in action_neighbors if state.board[n] == opponent]
            for i in range(len(opp_neighbors)):
                for j in range(i+1, len(opp_neighbors)):
                    o1, o2 = opp_neighbors[i], opp_neighbors[j]
                    shared = set(state._get_neighbors(o1)).intersection(state._get_neighbors(o2))
                    if len(shared) == 2 and action in shared:
                        other = (shared - {action}).pop()
                        if state.board[other] == -1:
                            opp_block_bonus += 1

            secondary_score = (bot_path + opp_path) - (bridge_bonus * 0.5) - (opp_block_bonus * 1.5)
            
            r, c = action // size, action % size
            center_dist = abs(r - size//2) + abs(c - size//2)
            
            score_tuple = (critical_score, secondary_score, center_dist)
            
            if best_score is None or score_tuple < best_score:
                best_score = score_tuple
                best_actions = [action]
            elif score_tuple == best_score:
                best_actions.append(action)
                
        if best_actions: return random.choice(best_actions)
        return random.choice(legal)

    def _copy_state(self, state):
        new = HexState(state.size)
        new.board = state.board.copy()
        new._current_player = state._current_player
        new.moves_played = state.moves_played
        new.first_move_action = state.first_move_action
        return new

    def _get_distances(self, state, player, is_start_edge):
        size = state.size
        dist = [float('inf')] * (size * size)
        q = collections.deque()
        for i in range(size):
            if player == 0: node = i if is_start_edge else (size - 1) * size + i
            else: node = i * size if is_start_edge else i * size + (size - 1)
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


# 4. IMITATION LEARNING (Faza 0)
def get_canonical_data(state, real_action):
    size = state.size; obs = np.zeros((3, size, size), dtype=np.float32)
    my_p, opp_p = state.current_player(), 1 - state.current_player()
    for r in range(size):
        for c in range(size):
            rr, rc = (c, r) if my_p == 1 else (r, c)
            val = state.board[rr * size + rc]
            if val == my_p: obs[0, r, c] = 1.0
            elif val == opp_p: obs[1, r, c] = 1.0
            else: obs[2, r, c] = 1.0
    mask = np.zeros(size * size + 1, dtype=np.bool_)
    for act in state.legal_actions():
        if act == state.swap_action: mask[act] = True
        else:
            pr, pc = act // size, act % size
            if my_p == 1: pr, pc = pc, pr
            mask[pr * size + pc] = True
    if real_action == state.swap_action: a_idx = real_action
    else:
        r, c = real_action // size, real_action % size
        if my_p == 1: r, c = c, r
        a_idx = r * size + c
    return obs, a_idx, mask

def play_single_game(size):
    expert = HeuristicAgent2(); opponents = [HeuristicAgent7(), HeuristicAgent(), HeuristicAgent3()]
    opponent = random.choice(opponents); s = HexState(size); game_data = []
    if random.choice([True, False]): agent1, agent2 = expert, opponent
    else: agent1, agent2 = opponent, expert
    while not s.is_terminal():
        legal = s.legal_actions(); current = agent1 if s.current_player() == 0 else agent2
        a = current.step(s)
        if current == expert:
            o, idx, m = get_canonical_data(s, a); game_data.append((o, idx, m))
        s.apply_action(a)
    return game_data

def generate_dataset(n_games=8000, size=BOARD_SIZE):
    data = []
    print(f"🚀 Generowanie {n_games} gier na CPU...")
    with concurrent.futures.ProcessPoolExecutor() as ex:
        futures = [ex.submit(play_single_game, size) for _ in range(n_games)]
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            data.extend(f.result())
            if (i+1)%500 == 0: print(f"Postęp: {i+1}/{n_games}")
    return data

class HexDataset(Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return torch.tensor(self.d[i][0]), torch.tensor(self.d[i][1]), torch.tensor(self.d[i][2])


# 5. LIGA I ŚRODOWISKO
def get_canonical_obs(state):
    size = state.size
    obs = np.zeros((3, size, size), dtype=np.float32)

    me = state.current_player()
    opp = 1 - me

    for r in range(size):
        for c in range(size):
            rr, rc = (c, r) if me == 1 else (r, c)
            val = state.board[rr * size + rc]

            if val == me:
                obs[0, r, c] = 1
            elif val == opp:
                obs[1, r, c] = 1
            else:
                obs[2, r, c] = 1

    return obs


def get_action_mask(state):
    size = state.size
    mask = np.zeros(size * size + 1, dtype=np.bool_)

    me = state.current_player()

    for act in state.legal_actions():
        if act == state.swap_action:
            mask[act] = True
        else:
            r, c = act // size, act % size
            if me == 1:
                r, c = c, r
            mask[r * size + c] = True

    return mask


def encode_action(state, action):
    size = state.size
    me = state.current_player()

    if action == state.swap_action:
        return action

    r, c = action // size, action % size
    if me == 1:
        r, c = c, r

    return r * size + c

class LeagueAgent:
    def __init__(self, size=BOARD_SIZE, device="cpu"):
        self.size = size
        self.device = device
        self.current_model = None
        self.past_models = []
        self.active_model = None

    def load_past_models(self):
        files = sorted(
            [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip")],
            reverse=True
        )
        self.past_models = []

        for f in files[:4]:
            try:
                self.past_models.append(
                    MaskablePPO.load(os.path.join(CHECKPOINT_DIR, f), device=self.device)
                )
            except:
                pass

    def select_opponent(self, current_model, heuristics):
        self.current_model = current_model
        r = random.random()

        # 1. HEURYSTYKI: 50%
        if r < 0.5:
            self.active_model = None 
            return random.choice(heuristics)

        # 2. PRZESZŁE MODELE 25%
        elif r < 0.75:
            if self.past_models:
                self.active_model = random.choice(self.past_models)
                return self
            else:
                # Jeśli liga pusta (początek treningu), idź w Self-Play
                self.active_model = self.current_model
                return self

        # 3. TERAŹNIEJSZY JA (SELF-PLAY): 25%
        else:
            self.active_model = self.current_model
            return self
        
    def get_agent_obs(self, state):
        size = self.size
        obs = np.zeros((3, size, size), dtype=np.float32)
        cp = state.current_player()
        opp = 1 - cp
        for r in range(size):
            for c in range(size):
                rr, rc = (c, r) if cp == 1 else (r, c)
                val = state.board[rr * size + rc]
                if val == cp: obs[0, r, c] = 1.0
                elif val == opp: obs[1, r, c] = 1.0
                else: obs[2, r, c] = 1.0
        return obs

    def get_agent_mask(self, state):
        size = self.size
        mask = np.zeros(size * size + 1, dtype=np.bool_)
        me = state.current_player()
        for act in state.legal_actions():
            if act == state.swap_action:
                mask[act] = True
            else:
                r, c = act // size, act % size
                if me == 1: r, c = c, r
                mask[r * size + c] = True
        return mask

    def decode_action(self, state, idx):
        if idx == self.size * self.size:
            return idx
        r, c = idx // self.size, idx % self.size
        if state.current_player() == 1:
            r, c = c, r 
        return r * self.size + c

    def step(self, state):
        legal = state.legal_actions()
        if not legal: return None

        if self.active_model is None:
            return random.choice(legal)

        obs = self.get_agent_obs(state)
        mask = self.get_agent_mask(state)

        action, _ = self.active_model.predict(
            obs, action_masks=mask, deterministic=False 
        )
        return self.decode_action(state, int(action))

class HexEnv(gym.Env):
    def __init__(self, size=BOARD_SIZE):
        super().__init__()
        self.size = size
        self.state = None
        self.model = None
        self.league = LeagueAgent(size)
        self.heuristics = [HeuristicAgent2(), HeuristicAgent7()]
        self.action_space = spaces.Discrete(size * size + 1)
        self.observation_space = spaces.Box(0, 1, (3, size, size), dtype=np.float32)
    
    def set_model(self, model):
        """Żywa referencja do modelu w jednowątkowym środowisku DummyVecEnv"""
        self.model = model

    def load_selfplay_model(self, path):
        """Wczytuje najnowszy model z dysku tylko do inference'u."""
        self.model = MaskablePPO.load(path, device="cpu")

    def set_difficulty(self, phase):
        """Zmienia zestaw oponentów w zależności od fazy treningu."""
        if phase == 1:
            self.heuristics = [HeuristicAgent()] 
        else:
            self.heuristics = [HeuristicAgent6(), HeuristicAgent7(), HeuristicAgent2(), HeuristicAgent4()]

    def refresh_league(self):
        self.league.load_past_models()

    def valid_action_mask(self):
        mask = np.zeros(self.size * self.size + 1, dtype=np.bool_)
        for a in range(self.size * self.size + 1):
            if self._map(a) in self.state.legal_actions():
                mask[a] = True
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = HexState(self.size)
        self.opponent = self.league.select_opponent(self.model, self.heuristics)
        
        self.p = random.choice([0, 1])
        self.rot = random.choice([True, False])

        if self.p == 1:
            opp_action = self.opponent.step(self.state)
            if opp_action is not None:
                self.state.apply_action(opp_action)

        return self._obs(), {}

    def step(self, action):
        real_action = self._map(action)

        if real_action not in self.state.legal_actions():
            return self._obs(), -1.0, True, False, {}

        self.state.apply_action(real_action)
        if self.state.is_terminal():
            return self._obs(), 1.0, True, False, {}

        opp_action = self.opponent.step(self.state)
        if opp_action is not None:
            self.state.apply_action(opp_action)
            if self.state.is_terminal():
                return self._obs(), -1.0, True, False, {}

        return self._obs(), 0.0, False, False, {}

    def _obs(self):
        size = self.size
        obs = np.zeros((3, size, size), dtype=np.float32)
        my_color = self.p
        opp_color = 1 - my_color
        for r in range(size):
            for c in range(size):
                rr, rc = (size-1-r, size-1-c) if self.rot else (r, c)
                if my_color == 1:
                    rr, rc = rc, rr
                val = self.state.board[rr * size + rc]
                if val == my_color: obs[0, r, c] = 1.0
                elif val == opp_color: obs[1, r, c] = 1.0
                else: obs[2, r, c] = 1.0
        return obs

    def _map(self, a):
        if a == self.size * self.size: return a
        r, c = a // self.size, a % self.size
        if self.rot: 
            r, c = self.size - 1 - r, self.size - 1 - c
        if self.p == 1: 
            r, c = c, r
        return r * self.size + c
            
    
# 6. TRENING GŁÓWNY

def get_latest_checkpoint(checkpoint_dir, prefix='hex_league'):
    """Funkcja szukająca najnowszego checkpointu w folderze."""
    if not os.path.exists(checkpoint_dir): return None, 0
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix) and f.endswith(".zip")]
    if not files: return None, 0
    
    max_steps = -1
    latest_file = ""
    for f in files:
        match = re.search(r'_(\d+)_steps', f)
        if match:
            steps = int(match.group(1))
            if steps > max_steps:
                max_steps = steps
                latest_file = f
                
    if max_steps == -1: return None, 0
    return os.path.join(checkpoint_dir, latest_file), max_steps

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOTAL_PHASE2_STEPS = 10_000_000
    
    # 0. IMITATION LEARNING
    if not os.path.exists(BC_PATH):
        print("\n=== FAZA 0: Imitation Learning ===")
        dataset = HexDataset(generate_dataset(8000))
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        bc = BCModel().to(device); opt = optim.Adam(bc.parameters(), lr=1e-3)
        for ep in range(8):
            l_tot = 0
            for o, a, m in loader:
                o, a, m = o.to(device), a.to(device), m.to(device)
                logits = bc(o).masked_fill(~m, -1e9)
                loss = nn.CrossEntropyLoss()(logits, a)
                opt.zero_grad(); loss.backward(); opt.step(); l_tot += loss.item()
            print(f"BC Epoch {ep+1} Loss: {l_tot/len(loader):.4f}")
        torch.save(bc.state_dict(), BC_PATH)

    def make_env(rank, seed=0):
        def _init():
            base_env = HexEnv(BOARD_SIZE)
            monitored_env = Monitor(base_env)
            masked_env = ActionMasker(monitored_env, lambda e: e.unwrapped.valid_action_mask())
            return masked_env
        return _init

    print("🔥 Odpalam środowisko w jednym, super-szybkim procesie (DummyVecEnv)...")
    env = DummyVecEnv([make_env(0)])

    policy_kwargs = dict(
        features_extractor_class=HexCNN, 
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[], vf=[])
    )

    steps_per_save = 250_000
    steps_per_refresh = 100_000

    class LCallback(CheckpointCallback):
        def _on_step(self) -> bool:
            if self.n_calls % steps_per_refresh == 0:
                self.training_env.env_method("set_model", self.model)
                self.training_env.env_method("refresh_league")
            return super()._on_step()

    callback = LCallback(
        save_freq=steps_per_save, 
        save_path=CHECKPOINT_DIR, 
        name_prefix='hex_league'
    )

    # 2. LOGIKA WZNAWIANIA LUB STARTU TRENINGU
    latest_ckpt, steps_done = get_latest_checkpoint(CHECKPOINT_DIR)

    if latest_ckpt:
        print(f"\n🔄 WZNANAWIAM TRENING OD PUNKTU: {latest_ckpt} ({steps_done} kroków)")
        
        steps_remaining = TOTAL_PHASE2_STEPS - steps_done
        
        if steps_remaining <= 0:
            print("🏆 Trening osiągnął już cel 10 milionów kroków!")
        else:
            print(f"🔥 Pozostało do wykonania: {steps_remaining} kroków.")
            
            custom_objects = {
                "learning_rate": LinearSchedule(2e-5, start_decay_at=0.5), 
                "target_kl": 0.05,
                "ent_coef": 0.005,
                "clip_range": 0.2
            }
            
            model_phase2 = MaskablePPO.load(
                latest_ckpt, 
                env=env, 
                device=device,
                custom_objects=custom_objects
            )
            
            env.env_method("set_difficulty", 2)
            env.env_method("set_model", model_phase2)
            env.env_method("refresh_league")
            
            model_phase2.learn(
                total_timesteps=TOTAL_PHASE2_STEPS,
                callback=callback, 
                reset_num_timesteps=False 
            )
            model_phase2.save("ppo_hex_GOD_MODE")

    else:
        print("\n=== FAZA 1 (Critic Warmup) ===")
        env.env_method("set_difficulty", 1)
        
        model = MaskablePPO(
            "CnnPolicy", env, 
            learning_rate=LinearSchedule(5e-5), 
            ent_coef=0.002, 
            clip_range=0.1, 
            target_kl=0.02, 
            n_steps=512, batch_size=512, verbose=1, device=device, 
            policy_kwargs=policy_kwargs
        )

        bc_state = torch.load(BC_PATH, map_location=device, weights_only=True)
        
        model.policy.features_extractor.load_state_dict(
            {k.replace('ex.', ''): v for k, v in bc_state.items() if 'ex.' in k}
        )

        with torch.no_grad():
            model.policy.action_net.weight.copy_(bc_state['head.weight'])
            model.policy.action_net.bias.copy_(bc_state['head.bias'])
            print("Pomyślnie załadowano wagi głowy decyzyjnej z BC do PPO!")

        for p in model.policy.features_extractor.parameters(): p.requires_grad = False
        for p in model.policy.action_net.parameters(): p.requires_grad = False
        
        model.learn(total_timesteps=200_000)

        print("\n=== FAZA 2: Liga Mistrzów (10 MLN) ===", flush=True)
        env.env_method("set_difficulty", 2)
        
        model_phase2 = MaskablePPO(
            "CnnPolicy", env, 
            learning_rate=LinearSchedule(5e-5), 
            ent_coef=0.005, 
            clip_range=0.15, 
            target_kl=0.03, 
            n_steps=512, batch_size=512, verbose=1, device=device, 
            policy_kwargs=policy_kwargs
        )

        model_phase2.policy.load_state_dict(model.policy.state_dict())

        for p in model_phase2.policy.parameters(): 
            p.requires_grad = True

        env.env_method("set_model", model_phase2)
        
        model_phase2.learn(total_timesteps=TOTAL_PHASE2_STEPS, callback=callback, reset_num_timesteps=False)
        model_phase2.save("ppo_hex_GOD_MODE")