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


