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

---
## Search and Planning with `beam_search_agent`

---
## Proof of Concept: Imitation Learning

---
## A First Stab at Reinforcement Learning: Vanilla PPO Implementation

---
## PPO Implementation via Graph Neural Networks

---
## References
- [Deep Learning Drizzle](https://github.com/kmario23/deep-learning-drizzle#tada-graph-neural-networks-geometric-dl-confetti_ball-balloon): a very nice collection of references
- Deep Reinforcement Learning, Y. Li (2018). arXiv: [1810.06339](https://arxiv.org/abs/1810.06339)
- GNN: Graph Neural Network and Large Language Model Based for Data Discovery, T. Hoang (2024). arXiv: [2408.13609v1](https://arxiv.org/abs/2408.13609v1)
- [A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/): a very nice blogpost
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/): the holy grail of debugging PPO networks
- [Theoretical Foundations of Graph Neural Networks](https://www.youtube.com/watch?v=uF53xsT7mjc) and the references therein
