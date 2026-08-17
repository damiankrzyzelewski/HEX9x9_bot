import argparse
import collections
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

try:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError:
    print("Brak bibliotek RL. Zainstaluj: pip install torch stable-baselines3 sb3-contrib")
    class BaseFeaturesExtractor(nn.Module):
        def __init__(self, obs_space, features_dim):
            super().__init__()

BOARD_SIZE = 9
N_CELLS = BOARD_SIZE * BOARD_SIZE


# ARCHITEKTURA PPO
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


class PPOAgent:
    def __init__(self, model_path, board_size=9):
        self.size = board_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.valid = False

        if os.path.exists(model_path):
            print(f"Ładowanie modelu: {model_path} na urządzenie {self.device}")
            try:
                self.model = MaskablePPO.load(model_path, device=self.device)
                self.valid = True
                print("Model PPO gotowy")
            except Exception as e:
                print(f"Błąd podczas ładowania modelu PPO: {e}")
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

    def step(self, state, p, rot=False):
        if not self.valid:
            return random.choice(state.legal_actions())

        obs = self._obs(state, p, rot)
        mask = self.valid_action_mask(state, p, rot)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=False)
        return self._map(int(action), p, rot)


PPO_MODEL_PATH = "league_checkpoints/PPO_1500000_steps.zip"
GLOBAL_PPO_BOT = PPOAgent(PPO_MODEL_PATH, BOARD_SIZE)

def ppo_agent_fn(state, player):
    return GLOBAL_PPO_BOT.step(state, player, rot=False)


# HEX ENGINE
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

    def current_player(self):
        return self._current_player

    def is_terminal(self):
        return self._is_terminal

    def returns(self):
        return self._returns

    def legal_actions(self):
        actions = [i for i, s in enumerate(self.board) if s == -1]
        if self.moves_played == 1:
            actions.append(self.swap_action)
        return actions

    def apply_action(self, action):
        if self._is_terminal:
            return

        if action == self.swap_action:
            if self.first_move_action is None:
                return
            self.board[self.first_move_action] = 1
            self._current_player = 0
            self.moves_played += 1
            return

        if action is None or action < 0 or action >= self.size * self.size:
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
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.size and 0 <= cc < self.size:
                out.append(rr * self.size + cc)
        return out

    def _check_win(self, player):
        visited = set()
        stack = []

        if player == 0:
            for c in range(self.size):
                node = c
                if self.board[node] == player:
                    stack.append(node)
        else:
            for r in range(self.size):
                node = r * self.size
                if self.board[node] == player:
                    stack.append(node)

        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            r, c = curr // self.size, curr % self.size

            if player == 0 and r == self.size - 1:
                return True
            if player == 1 and c == self.size - 1:
                return True

            for n in self._get_neighbors(curr):
                if self.board[n] == player and n not in visited:
                    stack.append(n)

        return False


class DistanceHelper:
    def get_distances(self, state, player, is_start_edge):
        size = state.size
        dist = [float("inf")] * (size * size)
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
                if state.board[n] == 1 - player:
                    continue
                cost = 0 if state.board[n] == player else 1
                nd = curr_d + cost
                if nd < dist[n]:
                    dist[n] = nd
                    if cost == 0:
                        q.appendleft(n)
                    else:
                        q.append(n)
        return dist

DIST = DistanceHelper()

# HEURYSTYCZNY OPPONENT
def generate_fair_openings(size):
    openings = []
    center = size // 2
    
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            dist = abs(r - center) + abs(c - center)
            if dist == 2 or dist == 3:
                openings.append(r * size + c)
                
    return openings

FAIR_OPENINGS = generate_fair_openings(9)

class HeuristicAgent2:
    def step(self, state):
        legal = state.legal_actions()
        if not legal:
            return None

        if state.moves_played == 0:
            valid_openings = [m for m in FAIR_OPENINGS if m in legal]
            if valid_openings and random.random() < 0.35:
                return random.choice(valid_openings)

        if state.moves_played == 1 and state.swap_action in legal:
            r = state.first_move_action // state.size
            c = state.first_move_action % state.size
            center = state.size // 2
            if abs(r - center) <= 2 and abs(c - center) <= 2:
                return state.swap_action

        player = state.current_player()
        opponent = 1 - player

        my_ds = DIST.get_distances(state, player, True)
        my_de = DIST.get_distances(state, player, False)
        opp_ds = DIST.get_distances(state, opponent, True)
        opp_de = DIST.get_distances(state, opponent, False)

        best_score = None
        best_actions = []
        for action in legal:
            if action == state.swap_action:
                continue
            my_path = my_ds[action] + my_de[action]
            opp_path = opp_ds[action] + opp_de[action]
            r, c = action // state.size, action % state.size
            center_d = abs(r - state.size // 2) + abs(c - state.size // 2)

            score = (min(my_path, opp_path + 0.1), my_path + opp_path, center_d)
            if best_score is None or score < best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        if best_actions:
            return random.choice(best_actions)
        return random.choice([a for a in legal if a != state.swap_action])

_HEURISTIC = HeuristicAgent2()

# COMPONENTS / BRIDGE FEATURES
def compute_components(state, color):
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


def count_bridge_patterns(state, action, color):
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


def move_feature_cache(state, player):
    opponent = 1 - player
    my_ds = DIST.get_distances(state, player, True)
    my_de = DIST.get_distances(state, player, False)
    opp_ds = DIST.get_distances(state, opponent, True)
    opp_de = DIST.get_distances(state, opponent, False)

    my_comp, my_comp_sizes = compute_components(state, player)
    opp_comp, opp_comp_sizes = compute_components(state, opponent)

    return {
        "my_ds": my_ds,
        "my_de": my_de,
        "opp_ds": opp_ds,
        "opp_de": opp_de,
        "my_comp": my_comp,
        "my_comp_sizes": my_comp_sizes,
        "opp_comp": opp_comp,
        "opp_comp_sizes": opp_comp_sizes,
    }


def features_for_move(state, action, player, cache):
    size = state.size
    max_center = 2 * (size - 1)
    norm_path = 2 * size
    opponent = 1 - player

    if action == state.swap_action:
        if state.first_move_action is None:
            return np.zeros(12, dtype=np.float32)

        first = state.first_move_action
        r, c = first // size, first % size
        center_dist = abs(r - size // 2) + abs(c - size // 2)
        center_score = 1.0 - (center_dist / max_center if max_center > 0 else 0.0)

        first_color = state.board[first]
        first_bridge = count_bridge_patterns(state, first, first_color)
        first_empty_n = sum(1 for n in state._get_neighbors(first) if state.board[n] == -1)
        first_friend_n = sum(1 for n in state._get_neighbors(first) if state.board[n] == first_color)

        return np.array([
            0.0, 0.0, 0.0,
            center_score, 0.0, 0.0,
            0.0, 0.0,
            float(first_bridge) / 6.0,
            0.0,
            float(first_empty_n) / 6.0,
            1.0,
        ], dtype=np.float32)

    r, c = action // size, action % size
    center_dist = abs(r - size // 2) + abs(c - size // 2)
    center_score = 1.0 - (center_dist / max_center if max_center > 0 else 0.0)

    my_path = cache["my_ds"][action] + cache["my_de"][action]
    opp_path = cache["opp_ds"][action] + cache["opp_de"][action]

    neigh = state._get_neighbors(action)
    friend_n = [n for n in neigh if state.board[n] == player]
    enemy_n = [n for n in neigh if state.board[n] == opponent]
    empty_n = [n for n in neigh if state.board[n] == -1]

    touched_components = set()
    comp_sizes = cache["my_comp_sizes"]
    my_comp = cache["my_comp"]
    for n in friend_n:
        cid = my_comp[n]
        if cid != -1:
            touched_components.add(cid)

    component_size_after = 1
    for cid in touched_components:
        component_size_after += comp_sizes[cid]

    bridge_support = count_bridge_patterns(state, action, player)
    bridge_block = count_bridge_patterns(state, action, opponent)

    my_path_score = 1.0 - min(my_path, norm_path * 2) / (norm_path * 2)
    opp_path_score = 1.0 - min(opp_path, norm_path * 2) / (norm_path * 2)
    path_adv = (opp_path - my_path) / norm_path

    return np.array([
        my_path_score,
        opp_path_score,
        path_adv,
        center_score,
        len(friend_n) / 6.0,
        len(enemy_n) / 6.0,
        len(touched_components) / 6.0,
        component_size_after / float(N_CELLS),
        bridge_support / 6.0,
        bridge_block / 6.0,
        len(empty_n) / 6.0,
        0.0,  
    ], dtype=np.float32)


# LINEAR POLICY
FEATURE_NAMES = [
    "my_path_score", "opp_path_score", "path_adv", "center_score",
    "friend_neighbors", "enemy_neighbors", "touched_components",
    "component_size_after", "bridge_support", "bridge_block",
    "liberties", "swap_bias",
]
N_PARAMS = len(FEATURE_NAMES)


def linear_agent(state, player, weights):
    legal = state.legal_actions()
    if not legal:
        return None

    cache = move_feature_cache(state, player)
    best_score = None
    best_action = None

    for action in legal:
        feat = features_for_move(state, action, player, cache)
        score = float(np.dot(weights, feat))
        if best_score is None or score > best_score:
            best_score = score
            best_action = action

    return best_action


def random_agent(state, _player):
    legal = [a for a in state.legal_actions() if a < N_CELLS]
    return random.choice(legal) if legal else None


def heuristic_agent(state, player):
    return _HEURISTIC.step(state)


# GAME SIMULATION
def play_game(fn0, fn1, board_size=BOARD_SIZE):
    state = HexState(board_size)
    max_moves = board_size * board_size + 5

    for _ in range(max_moves):
        if state.is_terminal():
            break
        p = state.current_player()
        action = fn0(state, p) if p == 0 else fn1(state, p)
        legal = state.legal_actions()
        if action not in legal:
            action = random.choice(legal)
        state.apply_action(action)

    if not state.is_terminal():
        return 0 if state._check_win(0) else 1
    return 0 if state.returns()[0] > 0 else 1


# HALL OF FAME
class HallOfFame:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.entries = [] 

    def is_empty(self):
        return len(self.entries) == 0

    def try_add(self, weights, fitness):
        entry = (float(fitness), weights.copy())
        if len(self.entries) < self.max_size:
            self.entries.append(entry)
            self.entries.sort(key=lambda x: x[0])
            return
        if fitness > self.entries[0][0]:
            self.entries[0] = entry
            self.entries.sort(key=lambda x: x[0])

    def sample(self, k=3):
        if not self.entries:
            return []
        k = min(k, len(self.entries))
        return [p for _, p in random.choices(self.entries, k=k)]

    def best(self):
        return self.entries[-1][1] if self.entries else None

    def __len__(self):
        return len(self.entries)


# FITNESS
def winrate(fn_a, fn_b, n_games):
    wins = 0
    half = n_games // 2

    for _ in range(half):
        if play_game(fn_a, fn_b) == 0:
            wins += 1
    for _ in range(half):
        if play_game(fn_b, fn_a) == 1:
            wins += 1

    return wins / max(n_games, 1)


def evaluate(weights, hof, progress, n_games):
    agent = lambda s, p: linear_agent(s, p, weights)

    if progress < 0.30:
        w_rand, w_heur, w_hof, w_ppo = 0.40, 0.40, 0.00, 0.20
    elif progress < 0.70:
        w_rand, w_heur, w_hof, w_ppo = 0.10, 0.30, 0.20, 0.40
    else:
        w_rand, w_heur, w_hof, w_ppo = 0.00, 0.20, 0.30, 0.50

    if not GLOBAL_PPO_BOT.valid:
        w_heur += w_ppo
        w_ppo = 0.0

    if hof.is_empty():
        w_heur += w_hof
        w_rand += 0.15 * w_hof
        w_hof = 0.0

    total = 0
    wins = 0

    def _run(opponent_fn, count):
        nonlocal total, wins
        if count <= 0:
            return
        half = max(1, count // 2)
        if half % 2 == 1 and half > 1:
            half -= 1
        count = half * 2
        if count <= 0:
            return
        for _ in range(count // 2):
            if play_game(agent, opponent_fn) == 0:
                wins += 1
        for _ in range(count // 2):
            if play_game(opponent_fn, agent) == 1:
                wins += 1
        total += count

    n_rand = round(n_games * w_rand)
    n_heur = round(n_games * w_heur)
    n_ppo = round(n_games * w_ppo)
    
    _run(random_agent, n_rand)
    _run(heuristic_agent, n_heur)
    
    if GLOBAL_PPO_BOT.valid and n_ppo > 0:
        _run(ppo_agent_fn, n_ppo)

    if w_hof > 0 and not hof.is_empty():
        n_hof = max(2, round(n_games * w_hof))
        for opp_p in hof.sample(n_hof):
            opp = lambda s, p, op=opp_p: linear_agent(s, p, op)
            side = random.randint(0, 1)
            if side == 0:
                if play_game(agent, opp) == 0:
                    wins += 1
            else:
                if play_game(opp, agent) == 1:
                    wins += 1
            total += 1

    return wins / total if total > 0 else 0.0


# CMA-ES
def optimize_cmaes(n_gen, pop_size, n_games, sigma0, hof_size):
    try:
        import cma
    except ImportError:
        print("Brak cma. Zainstaluj: pip install cma")
        return None, 0.0, HallOfFame(hof_size)

    hof = HallOfFame(max_size=hof_size)
    x0 = np.random.randn(N_PARAMS) * 0.1
    opts = {
        "maxiter": n_gen,
        "popsize": pop_size,
        "verbose": -9,
        "tolx": 1e-5,
        "tolfun": 1e-5,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_params = x0.copy()
    best_fitness = -1.0

    final_phase_best_params = x0.copy()
    final_phase_best_fitness = -1.0

    gen = 0
    t0 = time.time()
    
    current_phase = 0

    while not es.stop():
        progress = min(gen / max(n_gen, 1), 1.0)
        
        if progress < 0.30:
            phase = 0
            stage_name = "rand+heur"
        elif progress < 0.70:
            phase = 1
            stage_name = "heur+HoF+PPO"
        else:
            phase = 2
            stage_name = "PPO+HoF"

        if phase > current_phase:
            print(f"\n[!] Zmiana fazy na: {stage_name}! Czyszczę HoF ze starych modeli...")
            hof.entries.clear()
            current_phase = phase

        solutions = es.ask()
        fitnesses = [evaluate(s, hof, progress, n_games) for s in solutions]
        es.tell(solutions, [1.0 - f for f in fitnesses])

        gen_best_i = int(np.argmax(fitnesses))
        gen_best_fit = fitnesses[gen_best_i]

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_params = solutions[gen_best_i].copy()
            
        if phase == 2:
            if gen_best_fit > final_phase_best_fitness:
                final_phase_best_fitness = gen_best_fit
                final_phase_best_params = solutions[gen_best_i].copy()

        hof.try_add(solutions[gen_best_i], gen_best_fit)
        gen += 1

        eta = (time.time() - t0) / max(gen, 1) * (n_gen - gen)
        
        final_fit_str = f"{final_phase_best_fitness:.3f}" if final_phase_best_fitness >= 0 else "---"
        print(
            f"Gen {gen:4d}/{n_gen} │ gen: {gen_best_fit:.3f} "
            f"│ best(all): {best_fitness:.3f} │ best(final): {final_fit_str} "
            f"│ HoF: {len(hof):2d} │ [{stage_name}] │ ETA: {eta/60:.0f}min"
        )

    if final_phase_best_fitness >= 0.0:
        print(f"\n★ Zakończono! Wybieram najlepszy model z OSTATNIEJ fazy (fitness: {final_phase_best_fitness:.3f})")
        return final_phase_best_params, final_phase_best_fitness, hof
    else:
        print(f"\n⚠ Nie dotarto do ostatniej fazy. Wybieram najlepszy model ogólny (fitness: {best_fitness:.3f})")
        return best_params, best_fitness, hof

# TOURNAMENT
def tournament(weights, hof, n_games=200):
    agent = lambda s, p: linear_agent(s, p, weights)

    print("\n" + "=" * 60)
    print(f"TURNIEJ KOŃCOWY ({n_games} gier na matchup)")
    print("=" * 60)

    wr_rand = winrate(agent, random_agent, n_games)
    wr_heur = winrate(agent, heuristic_agent, n_games)
    print(f"vs RANDOM    : {wr_rand * 100:5.1f}%")
    print(f"vs HEURISTIC : {wr_heur * 100:5.1f}%")

    if not hof.is_empty():
        best_hof = lambda s, p: linear_agent(s, p, hof.best())
        wr_hof = winrate(agent, best_hof, n_games)
        print(f"vs BEST HoF  : {wr_hof * 100:5.1f}%")

    if GLOBAL_PPO_BOT.valid:
        wr_ppo = winrate(agent, ppo_agent_fn, n_games)
        print(f"vs PPO BOT   : {wr_ppo * 100:5.1f}%")

    print()
    if wr_heur >= 0.55:
        print("✓ Bot jest lepszy od heurystyki")
    elif wr_heur >= 0.45:
        print("~ Bot jest zbliżony do heurystyki")
    else:
        print("✗ Bot jest słabszy od heurystyki")



def main():
    parser = argparse.ArgumentParser(
        description="Hex bot: ewolucja heurystyki z dodatkiem sieci PPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", choices=["cmaes"], default="cmaes")
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=24)
    parser.add_argument("--hof", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--save", type=str, default="best_hex.npy")
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    print("\nHex bot – optymalizacja heurystyki z integracją PPO")
    print(f"Wejście: {N_PARAMS} cech")
    print("Cechy:", ", ".join(FEATURE_NAMES))

    if args.load:
        best_params = np.load(args.load)
        hof = HallOfFame(args.hof)
        hof.try_add(best_params, 0.5)
        print(f"\nWczytano wagi: {args.load}")
    else:
        best_params, best_fit, hof = optimize_cmaes(
            n_gen=args.generations,
            pop_size=args.popsize,
            n_games=args.games,
            sigma0=args.sigma,
            hof_size=args.hof,
        )
        if best_params is None:
            raise SystemExit(1)

    np.save(args.save, best_params)
    print(f"\n✓ Zapisano: {args.save}")
    print("Wagi:", np.array2string(best_params, precision=3, suppress_small=True))

    # Turniej weryfikujący
    tournament(best_params, hof, n_games=100)


if __name__ == "__main__":
    main()