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

* Orbiting planets: Planets whose `orbital_radius + planet_radius < 50` rotate around the sun at a constant angular velocity (0.025-0.05 radians/turn, randomized per game). Use initial_planets and angular_velocity from the observation to predict their positions.
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
## Proof of Concept: Imitation Learning via FFNs (courtesy `heuristic_agent`)

<center>
<img width="1990" height="2457" alt="image" src="https://github.com/user-attachments/assets/9534c571-826f-45c7-8feb-3e64d6377007" />
</center>

---
## A First Stab at Reinforcement Learning: Vanilla PPO Implementation

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


<center>
   <img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/dfe301da-8e99-40ea-b3ca-7b401cb46dec" />
</center>


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
