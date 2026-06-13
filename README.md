# Geometric Deep Learning in the Context of PPO RL Implementations

Orbit Wars is a real-time strategy game for 2 or 4 players [(Kaggle documentation)](https://www.kaggle.com/competitions/orbit-wars). The goal of this competition is to create and/or train AI bots to play a novel multi-agent 1v1 or 4p FFA game against other submitted agents. 

----
## Overview of the Game
Players start with a single home planet and compete to control the map by sending fleets to capture neutral and enemy planets. The board is a 100x100 continuous space with a sun at the center. Planets orbit the sun, comets fly through on elliptical trajectories, and fleets travel in straight lines. The game lasts 500 turns. The player with the most total ships (on planets + in fleets) at the end wins.

### Layout
* Board: 100x100 continuous space, origin at top-left.
* Sun: Centered at (50, 50) with radius 10. Fleets that cross the sun are destroyed.
* Symmetry: All planets and comets are placed with 4-fold mirror symmetry around the center: `(x, y), (100-x, y), (x, 100-y), (100-x, 100-y)`. This ensures fairness regardless of starting position.

Each planet is represented as `[id, owner, x, y, radius, ships, production]`:
* `owner`: Player ID (0-3), or -1 for neutral.
* `radius`: Determined by production: `1 + ln(production)`. Higher production planets are physically larger.
* `production`: Integer from 1 to 5. Each turn, an owned planet generates this many ships.
* `ships`: Current garrison. Starts between 5 and 99 (skewed toward lower values).

* Orbiting planets: Planets whose `orbital_radius + planet_radius < 50` rotate around the sun at a constant angular velocity (0.025-0.05 radians/turn, randomized per game). Use `initial_planets` and `angular_velocity` from the observation to predict their positions.
* Static planets: Planets further from the center do not rotate.

One symmetric group is randomly chosen as the starting planets. In a 2-player game, players start on diagonally opposite planets (Q1 and Q4). In a 4-player game, each player gets one planet from the group. Home planets start with 10 ships.


Each fleet is represented as `[id, owner, x, y, angle, from_planet_id, ships]`.
* `angle`: Direction of travel in radians.
* `ships`: Number of ships in the fleet (does not change during travel).

Fleet speed scales with size on a logarithmic curve:
```
speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5
```
Fleets travel in a straight line at their computed speed each turn. A fleet is removed if it:
* Goes out of bounds (leaves the 100x100 playing field).
* Crosses the sun (path segment comes within the sun's radius).
* Collides with any planet (path segment comes within the planet's radius). This triggers combat.

Each turn, your agent returns a list of moves: `[from_planet_id, direction_angle, num_ships]`.
* You can only launch from planets you own.
* You cannot launch more ships than the planet currently has.
* The fleet spawns just outside the planet's radius in the given direction.
* You can issue multiple launches from the same or different planets in a single turn.

`Comets` are temporary extra-solar objects that fly through the board on highly elliptical orbits around the sun. They spawn in groups of 4 (one per quadrant) at steps 50, 150, 250, 350, and 450.

### Turn Order

Each turn executes in this order:
1. Comet expiration: Remove comets that have left the board.
2. Comet spawning: Spawn new comet groups at designated steps.
3. Fleet launch: Process all player actions, creating new fleets.
4. Production: All owned planets (including comets) generate ships.
5. Fleet movement: Move all fleets along their headings. Check for out-of-bounds, sun collision, and planet collision. Fleets that hit planets are queued for combat.
6. Planet rotation & comet movement: Orbiting planets rotate, comets advance along their paths. Any fleet caught by a moving planet/comet is swept into combat with it.
7. Combat resolution: Resolve all queued planet combats.

When one or more fleets collide with a planet (either by flying into it or being swept by a moving planet), combat is resolved:
1. All arriving fleets are grouped by owner. Ships from the same owner are summed.
2. The largest attacking force fights the second largest. The difference in ships survives.
3. If there is a surviving attacker:
    * If the attacker is the same owner as the planet, the surviving ships are added to the garrison.
    * If the attacker is a different owner, the surviving ships fight the garrison. If the attackers exceed the garrison, the planet changes ownership and the garrison becomes the surplus.
4. If two attackers tie, all attacking ships are destroyed (no survivors).

The game ends when:
1. Step limit reached: 500 turns.
2. Elimination: Only one player (or zero) remains with any planets or fleets.
`Final score = total ships on owned planets + total ships in owned fleets`. Highest score wins.

### Observation Reference
|Field |	Type	| Description |
| --- | --------| -----------|
| `planets`	| `[[id, owner, x, y, radius, ships, production], ...]`|	All planets including comets|
| `fleets`|	`[[id, owner, x, y, angle, from_planet_id, ships], ...]` |	All active fleets |
| `player` |	`int`	| Your player ID (0-3) |
|`angular_velocity`|	`float`|	Planet rotation speed (radians/turn)|
|`initial_planets`|	`[[id, owner, x, y, radius, ships, production], ...]`|	Planet positions at game start|
|`comets`	|`[{planet_ids, paths, path_index}, ...]`|	Active comet group data|
|`comet_planet_ids`|	`[int, ...]`|	Planet IDs that are comets|
|`remainingOverageTime`|	`float`	|Remaining overage time budget (seconds)|

**Action Format**: Return a list of moves: `[[from_planet_id, direction_angle, num_ships], ...]`
* `from_planet_id`: ID of a planet you own.
* `direction_angle`: Angle in radians (0 = right, $\frac{\pi}{2}$ = down).
* `num_ships`: Integer number of ships to send.

---
## Exploring the Environment with `random_agent`

To gain familiarity with the mechanics of the game, we make `random_agent`:

```
...
    source = random.choice(my_planets)
    target = random.choice(planets)

    # Compute angle from source to target
    angle = math.atan2(target.y - source.y, target.x - source.x)

    # Send a random fraction of ships (between 20% and 80%)
    ships_to_send = int(source.ships * random.uniform(0.2, 0.8))
    if ships_to_send < 1:
        return []
...
```
`random_agent` selects a source and target at random and sends a random fraction of ships to that planet.

The image below captures a step between two `random_agent`s:

<p align="center">
   
<img width="508" height="508" alt="image" src="https://github.com/user-attachments/assets/719b60e2-3ecd-4c42-a7a5-11e55e9b35f3" />

</p>

---
## A Greedy Agent - `nearest_planet_sniper`
The greedy agent uses a simple strategy:
* For each planet we own, find the nearest planet we don't own
* Check if we have enough ships to capture it (need more than the target's garrison)
* Send exactly enough ships to take it (target ships + 1)

```
...
   # Separate our planets from targets
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # Find the nearest planet we don't own
        nearest = None
        min_dist = float('inf')
        for t in targets:
            dist = math.sqrt((mine.x - t.x)**2 + (mine.y - t.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = t

        if nearest is None:
            continue

        # How many ships do we need? Target's garrison + 1
        ships_needed = max(nearest.ships + 1, 20)

        # Only send if we have enough
        if mine.ships >= ships_needed:
            # Calculate angle from our planet to the target
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])
...
```

The image below shows a simulation between a `random_agent` (in orange) and a `nearest_planet_sniper` (in blue):
<p align="center">
   <img width="508" height="508" alt="image" src="https://github.com/user-attachments/assets/b092137d-3a7b-4f13-8408-76376230a37b" />
</p>

The sniper agent has a few problems:

* It doesn't account for travel time - the target planet produces ships while the fleet is in transit
* It sends fleets from every planet, even if multiple are targeting the same planet
* It ignores the sun - fleets aimed through the center get destroyed
* It holds ships on planets that have no nearby targets instead of consolidating

These limitations inform our construction of a physics-informed agent.

---
## Building a Physics-Informed `heuristic_agent`
To address the problems with the sniper agent, we make a more sophisticated agent, informed by physical information about the system:
```
...
    actions = []

    for src in my_planets:
        if src.id not in assignments:
            continue  # idle planet — no assignment found

        tgt, ships_to_send = assignments[src.id]

        # Compute intercept point (aim at where planet WILL be, not where it is)
        ix, iy, eta = intercept_point(
            src.x, src.y,
            tgt.id,
            ships_to_send,
            current_turn,
            initial_planets,
            angular_velocity
        )

        # Get a sun-safe angle toward the intercept point
        angle = safe_angle(src.x, src.y, ix, iy)

        if angle is None:
            # No safe path exists — skip this action rather than fly into the sun
            continue

        # Guard: never send 0 ships
        if ships_to_send <= 0:
            continue

        actions.append([src.id, angle, ships_to_send])
...
```
**Evaluation against `random_agent`:**
```
Results over 200 games:
  Wins:   165
  Losses: 35
  Draws:  0
  Win rate: 82.5%
```

**Evaluation against `nearest_planet_sniper`:**
```
Results over 200 games:
  Wins:   125
  Losses: 75
  Draws:  0
  Win rate: 62.5%
```


---
## Search and Planning with `beam_search_agent`

**Initial Strategy: Monte Carlo Tree Search** (MCTS) *Lite*

*Explanation*:

A regular heuristic agent asks: *"what's the best action right now?"*
MCTS asks: **"if I play this action, and then both sides play reasonably for the rest of the game, who wins?"**

It does this by building a tree of possible futures, one node per game state. The four steps that repeat every iteration:

1. **Selection** — starting from the root (current game state), walk down the tree picking the most promising node at each level. "Most promising" balances exploitation (nodes that have won before) vs exploration (nodes we haven't tried much). This is the UCB1 formula: $UCB1$ = $\frac{w}{n}+C\sqrt{\frac{ln N}{n}}$, where: $\frac{w}{n}$: win rate of the node (exploitation), $N$: visits to the parent, $n$: visits to this node (exploration), $C$: exploration constant (typically $\sqrt{2}$).

2. **Expansion** - at a leaf node (one we haven't explored yet), add its children (one per candidate action).

3. **Simulation** - from the new node, play the game to completion using our heuristic agent for both sides. Record who wins.

4. **Backpropagation** - walk back up the tree, updating each node's win count and visit count.

> Refer arXiv:2103.04931v4 [cs.AI] for more details

> Why this doesn't work? **Long** simulation times, never converged in the length of the game.

**Actual Strategy: Beam Search**

*Explanation*

At each decision point, instead of exploring one action (greedy) or all actions (full tree), you keep the top K candidates — called the "beam width". You expand only those K, evaluate their children, keep the top K again, and repeat for a fixed depth.

At each level you prune ruthlessly — only the top K survive regardless of which parent they came from.

MCTS needs many iterations to build up reliable visit counts before UCB1 means anything. With only 4-10 iterations you're essentially random.
Beam search makes no such assumption — it does one pass, evaluates every node once, and picks the best. With a fixed K and depth D you know exactly how many deepcopy calls you'll make: K × D. 

```
...
   # sync global env to true game state
    sync_env_to_obs(_mcts_env, obs)

    # generate and pre-rank candidates by score_target — no deepcopy yet
    candidate_actions = _get_candidate_actions(obs)
    if not candidate_actions:
        return heuristic_agent(obs, config)

    # pre-score each candidate cheaply using score_target
    planets = [Planet(*p) for p in obs.planets]
    my_planets = {p.id: p for p in planets if p.owner == my_player_id}
    other_planets = {p.id: p for p in planets if p.owner != my_player_id}

    prescored = []
    for action in candidate_actions:
        src_id = action[0][0]
        src = my_planets.get(src_id)
        if src is None:
            continue
        # score against best available target
        best = max(
            score_target(src, tgt, my_player_id, obs.step,
                        obs.initial_planets, obs.angular_velocity)
            for tgt in other_planets.values()
        )
        prescored.append((best, action))

    prescored.sort(reverse=True)
    top_candidates = [action for _, action in prescored[:K]]

    # simulate only top K candidates
    scored = []
    for action in top_candidates:
        sim_env = copy.deepcopy(_mcts_env)
        opp_obs = to_obs(sim_env.state[opp_player_id].observation)
        opp_action = heuristic_agent(opp_obs, None)

        if my_player_id == 0:
            sim_env.step([action, opp_action])
        else:
            sim_env.step([opp_action, action])

        result_obs = to_obs(sim_env.state[my_player_id].observation)
        score = evaluate_position(result_obs, my_player_id)
        scored.append((score, action))

    scored.sort(reverse=True)
    best_score, best_action = scored[0]
...
```

> Beam search can miss the best action if it gets pruned early — it's not guaranteed optimal like full tree search. But in practice for fast games with tight time budgets it tends to outperform broken MCTS significantly.

> No significant difference between D = 1 or D = 2; to keep computation costs low, we set D = 1

**Evaluation against `heuristic_agent`:**
```
Results over 200 games:
  Wins:   188
  Losses: 12
  Draws:  0
  Win rate: 94.0%
```

**Evaluation against `nearest_planet_sniper`:**
```
Results over 200 games:
  Wins:   189
  Losses: 11
  Draws:  0
  Win rate: 94.5%
```


---
## Imitation Learning using Multi-Layer Perceptrons 

With a strong `beam_search_agent` established (~94% win rate against the `heuristic_agent` baseline), the natural next step before full RL was imitation learning — training a neural network to mimic `BeamSearch` decisions. The motivation was twofold: introduce neural policies without RL instability, and establish a performance floor that PPO would need to beat.

### Data Collection
Demonstrations were collected by running `BeamSearch` vs `BeamSearch` self-play games and recording (observation, action) pairs at every turn. Both players' decisions were logged as separate training samples, effectively doubling data collection speed. Turns where either player had no valid moves (e.g. lost all planets near end of game) were skipped since they carry no useful signal.
Data collection was parallelised using Python's `multiprocessing.Pool` with 2 workers, each running independent game instances with their own copy of the global `_mcts_env state`. A key lesson learned here was to save incrementally after every game rather than at the end of the run — Colab session disconnects are frequent, and end-of-run saving risks losing everything.

```
...
def collect_imitation_data(n_games=200, verbose=True, save_path=None, worker_id=0):
    global _mcts_env
    _mcts_env = make("orbit_wars", debug=False)
    _mcts_env.reset()

    dataset = []

    for game_idx in range(n_games):
        env = make("orbit_wars", debug=False)
        env.reset()
        _mcts_env = env

        done = False
        while not done:
            obs0 = to_obs(env.state[0].observation)
            act0 = beam_search_agent(obs0, None)

            _mcts_env = env
            obs1 = to_obs(env.state[1].observation)
            act1 = beam_search_agent(obs1, None)

            if not act0 or not act1:
                env.step([act0, act1])
                done = (env.state[0].status != "ACTIVE")
                continue

            feat0 = obs_to_features(obs0, my_player_id=0)
            src0, angle0, ships0 = act0[0]

            feat1 = obs_to_features(obs1, my_player_id=1)
            src1, angle1, ships1 = act1[0]

            dataset.append({"features": feat0, "source_id": int(src0),
                            "angle": float(angle0), "ship_count_norm": float(ships0) / MAX_SHIPS})
            dataset.append({"features": feat1, "source_id": int(src1),
                            "angle": float(angle1), "ship_count_norm": float(ships1) / MAX_SHIPS})

            env.step([act0, act1])
            done = (env.state[0].status != "ACTIVE")

        # save after every game so a disconnect only loses current game
        if save_path is not None:
            with open(save_path, "wb") as f:
                pickle.dump(dataset, f)

        if verbose:
            print(f"[Worker {worker_id}] Game {game_idx + 1}/{n_games} done — {len(dataset)} samples", flush=True)

    return dataset
...
```
```
...
def run_worker(args):
    worker_id, n_games = args
    save_path = f"/content/drive/MyDrive/OrbitWars/imitation_data_worker{worker_id}.pkl"
    print(f"[Worker {worker_id}] starting {n_games} games", flush=True)
    data = collect_imitation_data(n_games=n_games, verbose=True,
                                  save_path=save_path, worker_id=worker_id)
    print(f"[Worker {worker_id}] done — {len(data)} samples", flush=True)
    return data

if __name__ == '__main__':
    N_GAMES   = 200
    N_WORKERS = 2
    games_per_worker = N_GAMES // N_WORKERS

    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        chunks = pool.map(run_worker, [(i, games_per_worker) for i in range(N_WORKERS)])
...
```
A complication emerged during preprocessing: planet count is not fixed across games, varying from 20 to 44, producing feature vectors of different lengths (103 to 223 dimensions). Since a standard MLP requires fixed-size inputs, we filtered to 32-planet games (163-dimensional feature vectors), the most common configuration in the dataset. This reduced the usable dataset from ~59,000 to ~15,000 samples.

### Feature Representation
Each observation was encoded as a flat vector of planet features concatenated with global game state features. Per planet (sorted by `id` for consistent ordering): normalised x/y coordinates (divided by 100, matching the map's coordinate range), ownership encoded relative to the current player (`0 = neutral, 1 = mine, 2 = opponent`), normalised ship count (divided by `MAX_SHIPS=500`), and normalised production (divided by the maximum production value in that game). Global features appended at the end: normalised turn number, total my ships, total opponent ships. Encoding ownership relative to the current player rather than using absolute player IDs means both players' observations look identical from their own perspective, which doubles effective training data and makes the policy transferable between player slots.

### Architecture
The network uses `Keras`'s functional API with a shared trunk and three separate output heads, since the action has mixed output types. The trunk is four dense layers `(256→128→128→64)` with batch normalisation and dropout `(0.3, 0.2, 0.2)` for regularisation. The three heads are:
* Source planet (which planet to dispatch from): 32-way softmax over planet indices, trained with categorical cross-entropy
* Angle (direction to send the fleet): rather than regressing the angle directly — which has a wrapping discontinuity at ±π — we predict (sin θ, cos θ) separately with tanh activations, recovering the angle at inference via `atan2(sin, cos)`
* Ship count: single sigmoid output producing a normalised fraction in [0, 1], multiplied by `MAX_SHIPS` at inference and clipped to the source planet's available ships

The three loss components operate on different scales: categorical cross-entropy for 32 classes has an expected random baseline of log(32) ≈ 3.58, while MSE losses for sin/cos and ships have expected baselines of ~0.5. Without correction, cross-entropy would dominate training. Each loss was therefore weighted by the inverse of its expected random baseline, normalising all components to start at approximately 1.0:

```
total_loss = (CE / 3.58) + (sin_MSE / 0.5) + (cos_MSE / 0.5) + (ships_MSE / 0.5)
```

Training converged at epoch 46 with a source planet accuracy of ~26.9% on the validation set — 8.6x better than the random baseline of 3.1% (1/32). Train and validation losses were virtually identical (1.94 vs 1.94), indicating clean generalisation with no overfitting, which suggests the bottleneck was data quantity rather than model capacity.

<center>
<img width="1990" height="2457" alt="image" src="https://github.com/user-attachments/assets/9534c571-826f-45c7-8feb-3e64d6377007" />
</center>

In practice, the agent was not competitive against the heuristic baseline. Two fundamental limitations made this expected: the 15,000 sample dataset was too small to learn reliable angle prediction (sin/cos MSE plateaued at ~0.34), and the fixed 32-planet filter discarded ~75% of collected data. The agent also falls back to the heuristic on non-32-planet games, making evaluation results difficult to interpret cleanly.

Imitation learning was therefore treated as a proof-of-concept and pipeline validation exercise rather than a serious performance milestone. The full data pipeline, multi-head architecture, and loss normalisation scheme are all sound — the limiting factor was purely data volume, which a variable-input architecture handles naturally by construction.


---
## Vanilla PPO Implementation

With imitation learning establishing a proof of concept for neural decision-making, the next phase introduced reinforcement learning via Proximal Policy Optimization (PPO), implemented through `Stable-Baselines3`. The goal was to move beyond mimicking the beam search agent and learn directly from game outcomes.



**Architecture of PPO Network**

```
Input: 163-dim feature vector
         │
         ▼
┌─────────────────────────────┐
│  FlattenExtractor           │  ← just passes the vector through unchanged
└─────────────────────────────┘
         │
         ├──────────────────────────┐
         ▼                          ▼
  Policy network              Value network
  Linear(163 → 64)            Linear(163 → 64)
  Tanh                        Tanh
  Linear(64 → 64)             Linear(64 → 64)
  Tanh                        Tanh
         │                          │
         ▼                          ▼
  action_net                  value_net
  Linear(64 → 34)             Linear(64 → 1)
         │                          │
         ▼                          ▼
  34 action values            1 scalar
  (32 logits + angle          "how good is
   + ship fraction)            this state")

```
### Action-Space Design

The most consequential design decision in this phase was the action space. The naive approach — a continuous `Box` action space with Gaussian policy heads predicting a source planet logit vector, a firing angle, and a ship fraction — failed completely. The `std` of the action distribution never moved from its initialisation value of 1.0 across tens of thousands of training steps, meaning the policy never committed to any action. The root cause is a mismatch between the problem structure and the policy architecture: selecting a target planet is fundamentally a discrete choice between at most 32 options, and a Gaussian head cannot efficiently learn what is essentially a lookup. The correct angle to fire is almost always "aim at planet X" — there is no meaningful sense in which angles near the correct value are better than angles far from it in a smooth, differentiable way.
The fix was to replace the continuous action space entirely with `MultiDiscrete([32, 32, 10])` — three independent discrete choices representing the source planet index, the target planet index, and a ship fraction bin (10% increments from 10% to 100%). Critically, the angle is never predicted by the network at all. Once the policy selects a target planet, the existing physics code (`intercept_point + safe_angle`) computes the correct interception angle deterministically. This offloads the hardest part of the geometric reasoning to the analytical toolkit built before, and lets the policy focus entirely on the strategically meaningful question of which planet to send ships to.

### Observation Space
The observation is a fixed-length `float32` vector of shape (183,), constructed by concatenating five features per planet — normalised x/y coordinates, an owner encoding relative to the current player (`0 = neutral, 1 = mine, 2 = opponent`), normalised ship count, and normalised production — across all planets, padded or truncated to a fixed count of 36, plus three global features (normalised step, my total ships, opponent total ships). The owner encoding being player-relative is essential: it ensures the same feature vector structure regardless of whether the agent is player 0 or player 1.

### Reward Structure
Two reward schemes were tested. Dense reward shaping — giving a signal proportional to the change in ship share each turn — failed despite appearing theoretically sound. Against a strong opponent, a randomly initialised policy loses ship share on almost every turn, producing a stream of small negative rewards with no variance. PPO has no contrast between good and bad actions and cannot find a gradient to follow. Sparse terminal reward (+1 for a win, -1 for a loss) was strictly better: it occasionally produces a positive signal when the agent gets lucky, and that contrast is enough to bootstrap learning.

### Training Curriculum
A two-stage curriculum was used. Round 1 trained against a random opponent for 100,000 steps, reaching approximately 60–70% win rate. Round 2 continued training against the heuristic agent for a further 100,000 steps. The transition was intentional — starting against the heuristic directly produces uniformly negative gradients early in training because the untrained policy has no chance of winning, starving it of positive signal. The random opponent provides enough winnable games to establish a baseline policy before the difficulty increases.
One additional fix proved essential: randomly assigning the agent as player 0 or player 1 at the start of each episode. Without this, the policy learns to output fixed planet IDs associated with the starting configuration it always sees as player 0, rather than learning to read the ownership encoding from the feature vector. This is a subtle form of overfitting that produces a policy that appears to train well but generalises to zero in evaluation.

<center>
   <img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/dfe301da-8e99-40ea-b3ca-7b401cb46dec" />
</center>

The PPO agent plateaued at roughly 20% win rate against the heuristic and effectively 0% against beam search. This ceiling is architectural rather than a training failure. The heuristic dispatches fleets from every owned planet simultaneously on each turn; the PPO agent dispatches from one. As the game progresses and planet counts diverge, no policy quality can compensate for a structural 5–10x disadvantage in action throughput. This single-dispatch limitation, combined with the MLP's inability to generalise across permutations of planet assignments, motivates the GNN architecture — where multi-dispatch falls out naturally from per-node scoring, and permutation invariance is a built-in property of the graph representation rather than something the network must learn.


### What PPO run taught us:

* `MultiDiscrete` is the right action space design: Discrete target selection dramatically outperformed the original continuous `Box` action space. PPO learns categorical decisions naturally; forcing a Gaussian policy to emulate discrete planet selection through argmax decoding produced weak learning signals and persistent exploration.
* Random player assignment is essential: Without randomisation the policy learns correlations with fixed planet indices and spawn positions. Training from both player perspectives forces the network to learn ownership patterns and game structure rather than memorising specific IDs.
* Sparse rewards work, dense rewards don't: Discrete target selection dramatically outperformed the original continuous Box action space. PPO learns categorical decisions naturally; forcing a Gaussian policy to emulate discrete planet selection through argmax decoding produced weak learning signals and persistent exploration.
* Single dispatch is a hard ceiling on performance. MLP can't generalise planet positions — it memorises them. As the empire grows, the gap between available strategic options and executable actions grows larger. The policy may understand advantageous positions but lacks the ability to exploit them fully.

The PPO implementation itself appears healthy.

Throughout debugging:

* KL divergence remained reasonable.
* Clip fractions stayed in a healthy range.
* Explained variance improved substantially.
* Policies successfully learned against random opponents.

The bottlenecks were not PPO hyperparameters but environment and representation design.

A flat MLP can learn useful behaviours but is an inefficient representation for Orbit Wars.

The game is naturally relational:

* planets influence neighbouring planets
* threats depend on travel distance
* reinforcements depend on connectivity

The MLP must infer these relationships from a flattened vector. This likely contributes to poor generalisation across map configurations and opponents.

### What went wrong against `heuristic_agent`?

The policy that learned to beat random never learned why it was winning — it learned surface patterns like "when I see this feature configuration, pick planet X." Against heuristic, those patterns don't transfer because heuristic responds intelligently and punishes the same moves that worked against random. The policy has no deeper strategic understanding to fall back on.

This is actually the expected result given the single dispatch limitation. The heuristic sends from every planet every turn. Our agent sends from one. Even if it picks perfectly, it's operating at a fraction of the heuristic's throughput. No amount of training fixes a 5x action rate disadvantage.

Further improvements are unlikely to come from PPO hyperparameter tuning alone. Future gains will likely require changes to the action space, state representation, or model architecture.

---
## PPO Implementation via Graph Neural Networks

### A Lightning Introduction to Graph Neural Networks

### Why GNNs?
* Multi-dispatch falls out naturally from per-node scoring
* Permutation invariance means planet IDs don't matter
* Variable planet counts handled natively


### What GNN fixes directly:

|**Problem**|**GNN solution**|
|-------|------------|
|Fixed planet ID memorisation | Nodes carry own features — permutation invariant|
Single dispatch| Per-node scoring → threshold → dispatch from all owned planets|
Variable planet counts| Graph handles variable node counts natively|
| No planet relationship modelling| Message passing propagates neighbour context|

### What ``obs_to_graph`` needs to do, and why:

The raw observation gives us a list of planet tuples. We need to turn that into a `torch_geometric.data.Data` object, which has three mandatory fields:
* `x` — node feature matrix, shape `[num_nodes, num_features]`
* `edge_index` — which nodes are connected, shape `[2, num_edges]` in COO format (two rows: source indices, target indices)
* `edge_attr` — edge feature matrix, shape `[num_edges, num_edge_features]`

**Node features — what to include and why each one**

The key design principle: use relative positions (`Δx, Δy` from the graph centroid or from the sun), not absolute coordinates. This makes the representation approximately translation-invariant — the network sees the same pattern regardless of where on the board things happen to be.
For each planet node we'll encode:

* `dx, dy` — position relative to the sun (50, 50). We already know sun is the fixed reference point in the game, so it's the natural origin. Normalised by 50 (half the board).
* `owner_enc` — 0 neutral, 1 mine, 2 opponent. Perspective-relative: the GNN learns "mine vs not-mine", not "player 0 vs player 1".
* `ships_norm` — `ships` / `MAX_SHIPS`
* `production_norm` — `production` / `max_production` across all planets
* `is_orbiting` — boolean flag; orbiting planets move, static ones don't. The physics toolkit already computes this. Worth including so the network can learn different strategies for moving vs static targets.

**Edge construction — k-nearest neighbours**

We connect each planet to its k=4 nearest neighbours (by current Euclidean distance). Always connect to the sun node — but for now let's keep it simple: planet nodes only, k=4 edges. We can add the sun node later when we do the full GNN policy.

**Edge features — what flows along edges**

* `delta_x, delta_y` — relative position from source node to target node, normalised. Encodes direction and rough distance.
* `dist_norm` — Euclidean distance normalised by board diagonal (~141 units). Redundant with `Δx/Δy` but useful for attention mechanisms to weight by proximity.
* `same_owner` — 1 if both planets have the same owner, 0 otherwise. Helps the network reason about "friendly territory".

```
   planets = [Planet(*p) for p in obs.planets]
    planets.sort(key=lambda p: p.id)   # consistent node ordering
    N = len(planets)

    max_prod = max(p.production for p in planets) or 1.0

    # ── node features ────────────────────────────────────────────────────────
    node_feats = []
    for p in planets:
        # position relative to sun, normalised to roughly [-1, 1]
        dx = (p.x - SUN_X) / (MAP_SIZE / 2)
        dy = (p.y - SUN_Y) / (MAP_SIZE / 2)

        # owner from this player's perspective
        if p.owner == -1:
            owner_enc = 0.0        # neutral
        elif p.owner == my_player_id:
            owner_enc = 1.0        # mine
        else:
            owner_enc = 2.0        # opponent

        ships_norm = p.ships / MAX_SHIPS
        prod_norm  = p.production / max_prod

        # is this planet orbiting? (same check as planet_position in physics.py)
        orbital_radius = math.sqrt((p.x - SUN_X)**2 + (p.y - SUN_Y)**2)
        is_orbiting    = 1.0 if (orbital_radius + p.radius) < 50 else 0.0

        node_feats.append([dx, dy, owner_enc, ships_norm, prod_norm, is_orbiting])

    x = torch.tensor(node_feats, dtype=torch.float)   # [N, 6]

    # ── build k-NN edges ─────────────────────────────────────────────────────
    # compute pairwise distances between all planets
    coords = [(p.x, p.y) for p in planets]

    src_list, dst_list = [], []
    edge_feats = []

    BOARD_DIAG = math.sqrt(MAP_SIZE**2 + MAP_SIZE**2)   # ≈ 141.4

    for i in range(N):
        # distance from planet i to all others
        dists = []
        for j in range(N):
            if i == j:
                continue
            d = math.sqrt((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2)
            dists.append((d, j))

        dists.sort()
        neighbours = [j for _, j in dists[:K_NEIGHBOURS]]   # K closest

        for j in neighbours:
            # add directed edge i → j
            src_list.append(i)
            dst_list.append(j)

            # edge features: relative position + distance + same_owner flag
            rel_x = (coords[j][0] - coords[i][0]) / (MAP_SIZE / 2)
            rel_y = (coords[j][1] - coords[i][1]) / (MAP_SIZE / 2)
            dist_norm   = math.sqrt(rel_x**2 + rel_y**2)   # already normalised
            same_owner  = 1.0 if (planets[i].owner == planets[j].owner
                                  and planets[i].owner != -1) else 0.0

            edge_feats.append([rel_x, rel_y, dist_norm, same_owner])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)   # [2, E]
    edge_attr  = torch.tensor(edge_feats, dtype=torch.float)             # [E, 4]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
```

Three things to note for the actual function:

* `torch_geometric` uses COO (coordinate) format for edges: `edge_index` is a `[2, E]` int64 tensor where `edge_index[0]` is all the source node indices and `edge_index[1]` is all the destination node indices. So edge i goes from node `edge_index[0][i]` to node `edge_index[1][i]`.
* We're building a directed graph — each k-NN edge goes both ways. So if planet A is a neighbour of planet B, we add both A→B and B→A. This doubles the edge count `(2 × N × K edges total)` but gives the message passing something to work with in both directions.
* `is_orbiting` is just (`orbital_radius + planet_radius) < 50` — the exact same check from `planet_position` in the physics toolkit.

### Why ``GATv2Conv``:

The GNN encoder takes a graph, outputs a per-node embedding vector. This is the "understanding" part of the network. We'll use ``GATv2Conv`` (Graph Attention Network v2) from ``PyTorch Geometric``.

The original GAT computes attention weights as:

<center>
  $\alpha_{ij}$ = softmax( $a^T$ · LeakyReLU( $W·h_i$ || $W·h_j$ ) )
</center>

The problem is that `W·h_i` and `W·h_j` are computed before they interact — so the attention is actually a function of `W·h_i` alone when you factor it. GATv2 fixes this by concatenating before the linear transform:

<center>
  $\alpha_{ij}$ = softmax( $a^T$ · LeakyReLU($W$ · $[h_i$ || $h_j$] ) )
</center>

Now the attention weight genuinely depends on the pair `(i, j)`, not just `i`. For OrbitWars this matters: whether planet A should "pay attention" to planet B depends on both of them — their relative ownership, ship counts, and proximity all interact.

After the encoder runs, we have `h`: a matrix of shape `[N, 64]` — one 64-dimensional embedding per planet. The network now needs to answer two questions:

* **Policy head (actor)** : *"which action should I take?"* In our `MultiDiscrete([32, 32, 10])` space this means producing three independent probability distributions — one over source planets, one over target planets, one over ship fraction bins.
* **Value head (critic)**: *"how good is the current state overall?"* This needs to be a single scalar — not per-planet, but a global assessment. We get there by global mean pooling: average all node embeddings into one vector, then pass through an MLP. This is the standard approach and it's permutation-invariant by construction — the mean of a set doesn't depend on ordering.

The policy head is more interesting. We have N planet embeddings but a fixed 40-slot action space. The way to handle this cleanly:

* For source and target: learn a linear projection `[64 → 1]` that scores each node. This gives `[N, 1]` scores. Pad to [40, 1] (zeros for missing slots) and apply softmax. The network learns "what makes a planet a good source / good target" as a scoring function over embeddings.
* For ship fraction: global mean pool → small MLP → 10 logits. Ship fraction is a global decision (how aggressive to be right now), not a per-planet one.

### Masking

**Rationale**

With 40 source slots but only 6 valid ones, roughly 85% of source samples are invalid at any given turn. PPO wastes most of its exploration budget on actions it will never execute. Worse, the entropy bonus encourages exploring all 40 slots equally — so the policy is actively rewarded for spreading probability mass over illegal actions.
Action masking fixes this by zeroing out illegal logits before the softmax, so illegal actions get exactly zero probability. SB3 supports this natively through `MaskablePPO` from the `sb3-contrib package`.
The mask itself is simple — a boolean vector of length 40 for source (`True` where the planet is owned by us), another for target (`True` where the planet exists and isn't our source), and `all-True` for ship bins. We compute this from `obs` at each step.

`MaskablePPO` expects the environment to implement a method called `action_masks()` that returns a flat boolean numpy array. For a `MultiDiscrete([40, 40, 10])` space, this array has length `40 + 40 + 10 = 90`. The first 40 entries mask the source distribution, the next 40 mask the target distribution, the last 10 mask the ship bins. `True` means *"this action is legal"*, `False` means *"zero probability, never sample this"*.

The masking happens inside the distribution layer before sampling — so illegal actions are never chosen, and the entropy calculation only counts legal actions. This is exactly what we want.

**What the masks should be**:

* Source mask: True only for planets we own that have at least 1 ship.
* Target mask: True for any planet that exists in this game and isn't the source. We can't know the source at mask-construction time (it hasn't been sampled yet), so we mask out planets that don't exist in the game. The `source == target` case is handled in `_decode_action` by returning `[]` — it's rare enough that letting it fall through to do-nothing is fine.
* Ship bins: all True — all fractions are always legal.

To establish a baseline and check network connections, we implement a baseline PPO with masking, the results of which are shown below:

<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/a13cbd1f-4e86-4a9e-a041-f05dc583f142" />

**What's the damage?**

The critic learning well (explained variance 0.93) while the actor stays flat is a known failure mode in PPO with sparse rewards and hard-to-represent observations. The critic only needs to predict "is this state winning or losing" — a signal it can extract even from a mediocre representation. The actor needs to identify which specific action caused the win, which requires meaningful features. The flat padded MLP can't provide those.

The baseline failure mode is now well-characterized: **the actor's action distributions remain almost uniform despite healthy training metrics.**


---
## References
- [Deep Learning Drizzle](https://github.com/kmario23/deep-learning-drizzle#tada-graph-neural-networks-geometric-dl-confetti_ball-balloon): a very nice collection of references
- Deep Reinforcement Learning, Y. Li (2018). arXiv: [1810.06339](https://arxiv.org/abs/1810.06339)
- GNN: Graph Neural Network and Large Language Model Based for Data Discovery, T. Hoang (2024). arXiv: [2408.13609v1](https://arxiv.org/abs/2408.13609v1)
- [A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/): a very nice blogpost
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/): the holy grail of debugging PPO networks
- [Theoretical Foundations of Graph Neural Networks](https://www.youtube.com/watch?v=uF53xsT7mjc) and the references therein
- [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
- Introduction to Graph Neural Networks for Machine Learning Engineers, J. H. Tanis, C. Giannella, A. V. Mariano, D. Meerzaman (2024). arXiv: [2412.19419](https://arxiv.org/abs/2412.19419)


