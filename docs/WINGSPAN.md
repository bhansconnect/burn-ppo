# WINGSPAN.md - Wingspan RL Agent Design

This document describes the design for implementing Wingspan as a PPO training environment in this framework.

## Philosophy

**Single-player framing**: The agent maximizes its own score. Other players are part of the "environment" - they take actions according to fixed policies (random, trained agents, or human). This simplifies credit assignment: the agent only cares about optimizing its own score delta.

**Reward: Delta score per turn**: `reward = current_score - previous_score`. No sparse end-game bonus. This provides dense signal for learning.

**Generalization**: Cards are encoded with semantic features, not learned embeddings. New cards from expansions can be added without retraining.

---

## Game Complexity Analysis

| Aspect | Value | Implication |
|--------|-------|-------------|
| Rounds | 4 | Long-horizon planning required |
| Total turns | 26 (8+7+6+5) | ~50-100 network calls per game |
| Bird cards | 170+ base, 400+ with expansions | Large card space, need semantic encoding |
| Powers | 50+ unique types | Modular power system required |
| Decision types | 4 main + sub-decisions | Hybrid action space |
| Stochastic elements | Dice, deck draws | Can't pre-plan everything |

---

## State Space Design

### Own State (~500-700 dims)

| Component | Dimensions | Notes |
|-----------|------------|-------|
| Hand | 16 × 20 = 320 | 16 card slots × semantic features |
| Board | 15 × 24 = 360 | 3 habitats × 5 columns × (card + eggs + cache) |
| Resources | 6 | 5 food types + egg supply |
| Game progress | 5 | round, turn, actions remaining, score |

### End-of-Round Goals (~60 dims)

The agent must see the current goals to strategize toward them.

| Per Goal | Dims | Description |
|----------|------|-------------|
| Goal type | 15 | One-hot encoding of goal type |
| My progress | 1 | Current count toward this goal |
| Leading progress | 1 | Highest count among all players |
| Am leading | 1 | Binary: am I currently winning? |
| **Per goal total** | ~18 | |
| **4 rounds total** | ~60-70 | |

**Goal Types:**
- `most_eggs_in_forest`, `most_eggs_in_grassland`, `most_eggs_in_wetland`
- `most_birds_in_forest`, `most_birds_in_grassland`, `most_birds_in_wetland`
- `most_cavity_nests`, `most_platform_nests`, `most_bowl_nests`, `most_ground_nests`
- `sets_of_eggs`, `birds_with_no_eggs`, `food_in_personal_supply`
- ~15 total goal types

### Bonus Cards (~60 dims, Phase 3+)

- Variable number (start 1-2, gain more via powers)
- Fixed 4 slots for encoding, zero-padded
- Per card: type one-hot + progress + max_possible

### Opponent Visibility (Configurable)

| Level | Description | Additional Dims |
|-------|-------------|-----------------|
| 0 (None) | Default - single player optimization | 0 |
| 1 (Scores) | Track if ahead/behind | 4 (one per opponent) |
| 2 (Summary) | End-of-round goal awareness | ~48 (12 per opponent) |

**Level 2 includes per opponent:**
- Score
- Bird count per habitat (3)
- Total eggs
- Nest type counts (5)

### Card Semantic Encoding

Fixed features per card (not learned, allows expansion generalization):

```
Numeric (4 dims):
  - cost / 4.0         # Normalized food cost
  - vp / 9.0           # Normalized victory points
  - egg_capacity / 6.0 # Normalized
  - wingspan / 200.0   # Normalized

Habitat compatibility (3 dims):
  - Multi-hot: [1,0,0] = forest only
  -            [1,1,0] = forest or grassland
  -            [1,1,1] = any habitat

Nest type (5 dims, one-hot):
  - Platform, Bowl, Cavity, Ground, Wild

Food cost types (5 dims):
  - Invertebrate, Seed, Fish, Fruit, Rodent counts

Power semantics (6 dims):
  - Effect category (draw, tuck, food, eggs, cache, other)
  - Trigger timing (when_played, when_activated, between_turns)
  - Magnitude (scaled 0-1)
```

**Total: ~23 dims per card**

---

## Action Space Design

### Why Hybrid (Not Flat)

A flat action space won't work for Wingspan because:

1. **Stochastic elements**: Can't specify "gain worm" before seeing dice
2. **Combinatorial explosion**: Egg distribution over 15 birds = massive space
3. **Sequential dependencies**: Power choices depend on what was drawn

### Main Action (Flat, Masked, ~20 actions)

```
PLAY_BIRD_0 through PLAY_BIRD_15  (16 actions)
ACTIVATE_FOREST                    (1 action)
ACTIVATE_GRASSLAND                 (1 action)
ACTIVATE_WETLAND                   (1 action)
END_ROUND                          (1 action)
```

Total: ~20 main actions with masking for illegal moves.

### Sequential Sub-Decisions

After main action, additional decisions may be required:

| Trigger | Sub-Decision | Action Space |
|---------|--------------|--------------|
| PLAY_BIRD | Habitat slot | 0-14 (valid columns) |
| PLAY_BIRD | Food payment | If multiple valid combos |
| ACTIVATE_* | Power choices | Auto-resolved initially |
| Row end | Egg distribution | Iterative assignment |
| Row end | Card selection | From tray/deck options |

### Action Masking

- **Hand positions**: Only cards actually in hand
- **Habitat slots**: Only valid habitats for the bird + non-full columns
- **Food payment**: Only affordable combinations
- **Birds**: Only if player has required resources

---

## Power System

### Declarative TOML (~90% of powers)

Powers defined in TOML, executed by generic engine:

```toml
# Simple: Draw 2 cards
[[powers]]
id = "draw_2"
trigger = "when_activated"
effects = [{ type = "draw_cards", count = 2 }]

# Conditional: If eggs, gain food
[[powers]]
id = "food_if_eggs"
trigger = "when_activated"
condition = { type = "min_eggs", target = "self", value = 1 }
effects = [{ type = "gain_food", food = "any", count = 1 }]

# Choice: Draw OR gain food
[[powers]]
id = "draw_or_food"
trigger = "when_activated"
choice_of = [
  [{ type = "draw_cards", count = 1 }],
  [{ type = "gain_food", food = "any", count = 1 }]
]

# Multiplier: Draw cards equal to eggs
[[powers]]
id = "draw_per_egg"
trigger = "when_activated"
effects = [{ type = "draw_cards", count = { per = "eggs_on_self" } }]

# All players (pink power)
[[powers]]
id = "all_may_tuck"
trigger = "between_turns"
all_players = true
optional = true
effects = [{ type = "tuck", source = "hand", count = 1 }]
```

### Effect Types

| Type | Parameters | Description |
|------|------------|-------------|
| `draw_cards` | count, source | Draw from deck or tray |
| `tuck` | count, source | Tuck cards behind bird |
| `gain_food` | food, count | Gain from birdfeeder |
| `lay_eggs` | target, count | Lay eggs on bird(s) |
| `cache_food` | food, count | Cache food on bird |
| `discard` | source, count | Discard cards/food |

### Triggers

- `when_played` - Once when bird enters play
- `when_activated` - Each time row is activated
- `between_turns` - On other players' turns (pink)
- `end_of_round` - At round end
- `end_of_game` - Final scoring

### Custom Handlers (~10% of powers)

Complex powers implemented in Rust:

```toml
[[powers]]
id = "look_top_keep"
trigger = "when_activated"
custom_handler = "look_top_keep"
handler_args = { look = 3, keep = 1 }
```

---

## Card Data Format

### Bird Card Definition

```toml
# data/cards/base/forest.toml

[[cards]]
id = "eastern_bluebird"
name = "Eastern Bluebird"
habitat = ["forest", "grassland"]    # Multi-habitat
cost = ["invertebrate"]
nest = "cavity"
egg_capacity = 5
victory_points = 4
wingspan = 32
power = { trigger = "when_activated", effects = [{ type = "draw_cards", count = 1 }] }
set = "base"

[[cards]]
id = "common_raven"
name = "Common Raven"
habitat = ["any"]                     # Any habitat bird
cost = ["wild", "wild"]
nest = "platform"
egg_capacity = 4
victory_points = 2
wingspan = 120
power = { trigger = "when_played", custom_handler = "steal_from_supply" }
set = "base"
```

### Card Set Organization

```
data/cards/
├── minimal/          # 20-30 simple cards for Phase 1
│   ├── forest.toml
│   ├── grassland.toml
│   └── wetland.toml
├── simple/           # 80-100 cards for Phase 2
│   └── ...
├── base/             # Full base game (~170 cards)
│   └── ...
├── european/         # European expansion
│   └── all.toml
├── oceania/          # Oceania expansion
│   └── all.toml
└── powers.toml       # Shared power definitions
```

### Configuration

```toml
[wingspan]
card_sets = ["minimal"]                # Phase 1
# card_sets = ["minimal", "simple"]    # Phase 2
# card_sets = ["base"]                 # Phase 3
# card_sets = ["base", "european"]     # With expansion

num_rounds = 1                         # 1-4
opponent_visibility = "none"           # none, scores, summary
power_resolution = "auto"              # auto, network
```

---

## Data Structures

```rust
pub struct WingspanConfig {
    pub card_sets: Vec<String>,
    pub num_rounds: usize,
    pub opponent_visibility: OpponentVisibility,
    pub power_resolution: PowerResolution,
}

pub enum OpponentVisibility {
    None,
    ScoresOnly,
    BoardSummary,
}

pub enum PowerResolution {
    AutoResolve,
    NetworkChoice,
}

pub struct WingspanState {
    // Own state
    pub hand: [Option<CardId>; 16],
    pub board: [[Option<BirdSlot>; 5]; 3],  // [habitat][column]
    pub food: [u8; 5],                       // Per food type
    pub eggs_supply: u8,
    pub round: u8,
    pub turn: u8,
    pub actions_remaining: u8,
    pub score: i32,

    // Goals and bonuses
    pub round_goals: [RoundGoal; 4],
    pub bonus_cards: Vec<BonusCard>,

    // Opponent state (if visible)
    pub opponent_scores: Option<[i32; 4]>,
    pub opponent_summaries: Option<[BoardSummary; 4]>,
}

pub struct BirdSlot {
    pub card_id: CardId,
    pub eggs: u8,
    pub cached_food: Vec<FoodType>,
    pub tucked_cards: u8,
}

pub struct RoundGoal {
    pub goal_type: GoalType,
    pub my_progress: u8,
    pub leading_progress: u8,
    pub am_leading: bool,
}

pub struct BoardSummary {
    pub score: i32,
    pub birds_per_habitat: [u8; 3],
    pub total_eggs: u8,
    pub nest_counts: [u8; 5],
}
```

---

## Reward Design

### Primary: Score Delta

```rust
reward = current_score - previous_score
```

Scores come from:
- Victory points on birds played
- Eggs laid (1 VP each)
- Cards tucked (1 VP each)
- Food cached (1 VP each, some birds)
- End-of-round goal points
- Bonus card points

### Optional Shaping (Phase 2+)

```rust
reward = score_delta
       + 0.1 * birds_played_delta
       + 0.05 * eggs_laid_delta
```

Small shaping rewards encourage board building and egg accumulation.

### No Sparse End-Game Bonus

All reward comes from turn-by-turn deltas for better credit assignment.

---

## Training Curriculum

### Phase 1: Micro-Wingspan (Learn Mechanics)

| Aspect | Setting |
|--------|---------|
| Rounds | 1 |
| Turns | 8 |
| Cards | 20-30 simple (single-habitat) |
| Powers | draw, tuck, gain_food only |
| Goals | None (zeros in state) |
| Bonus cards | None |
| Opponents | Random policy |
| Focus | Learn basic mechanics |

**Validation target**: Average score > 15 in 1M steps

### Phase 2: Mini-Wingspan (Learn Strategy)

| Aspect | Setting |
|--------|---------|
| Rounds | 2 |
| Turns | 8, 6 |
| Cards | 50-80 varied (includes multi-habitat) |
| Powers | Full variety |
| Goals | 2 simple types (birds/eggs in habitat) |
| Bonus cards | None |
| Opponents | Random or simple trained |
| Focus | Engine building, goal awareness |

**Validation target**: Average score > 40 in 5M steps

### Phase 3: Standard (Full Game)

| Aspect | Setting |
|--------|---------|
| Rounds | 4 |
| Turns | 8, 7, 6, 5 |
| Cards | Full base game (~170) |
| Powers | All including custom handlers |
| Goals | Full 4 goals (all types) |
| Bonus cards | 2 per player |
| Opponents | Trained agents or self-play |
| Focus | Complete strategy |

**Validation target**: Average score > 70 in 20M steps

### Phase 4: Expansions

- Add European/Oceania cards
- Test zero-shot generalization
- Fine-tune if needed
- Add nectars (Oceania food type)

---

## Implementation Phases

### Phase 1: Core Loop

- Basic `WingspanState` struct
- Main action selection (PLAY_BIRD, ACTIVATE_*, END_ROUND)
- `Environment` trait implementation
- TOML card loading
- Single round scoring
- No powers (or trivial only)

### Phase 2: Full Rules

- All 4 rounds with turn reduction
- End-of-round goal tracking and scoring
- Complete power system (auto-resolved)
- Full food/egg/tuck mechanics
- Birdfeeder dice simulation

### Phase 3: Powers and Choices

- Power triggers (when_played, when_activated, between_turns)
- Auto-resolution heuristics for power choices
- Optional: network-driven power choices (sub-policy head)

### Phase 4: Opponent Modeling

- Configurable visibility levels
- Multiple opponent policies (random, trained, human)
- Score tracking and board summaries
- Self-play training option

---

## Testing Strategy

### Unit Tests

- Card TOML parsing and validation
- Power effect resolution correctness
- Action masking (no illegal actions ever valid)
- State encoding dimension checks
- Score calculation accuracy

### Integration Tests

- Full game completes without crash
- Random agent plays valid games
- Episode statistics tracking works
- Checkpoint save/load preserves state

### Validation Benchmarks

| Phase | Target Score | Timesteps | Baseline |
|-------|--------------|-----------|----------|
| Micro | > 15 | 1M | Random: ~8 |
| Mini | > 40 | 5M | Random: ~20 |
| Standard | > 70 | 20M | Random: ~25 |

---

## Future Extensions

### Network Architecture

- **Attention over hand**: Handle variable-length hands without truncation
- **Attention over board**: Focus on relevant birds for decisions
- **Separate actor/critic**: If shared backbone limits performance

### Training Methods

- **Self-play pool**: Train against historical versions
- **Population-based training**: Evolve hyperparameters
- **Inverse RL**: Learn reward function from human game logs

### Game Extensions

- Asian expansion (new birds)
- Nectars (Oceania food type)
- Automa (official solo opponent)
- Goal/bonus card variants

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Player framing | Single-player vs environment | Simpler credit assignment |
| Reward | Delta score | Dense signal, no end-game cliff |
| Card encoding | Semantic (static) | Generalize to expansions |
| Habitat | Multi-hot array | Handles any-habitat birds |
| Action space | Hybrid flat + sequential | Balance simplicity and expressiveness |
| Powers | Declarative TOML | Extensible, no recompile |
| Goals | Encoded in state | Agent can strategize toward them |
| Opponent visibility | Configurable levels | Progressive complexity |

---

## Wingspan Rules Reference

### Turn Structure

1. Choose action: Play Bird, Activate Row, or End Round
2. Execute action (triggers powers)
3. Score delta calculated
4. Next player's turn (in environment)

### Actions

**Play Bird:**
1. Choose bird from hand
2. Choose habitat (if bird allows multiple)
3. Pay food cost
4. Place bird in leftmost empty column
5. Trigger "when played" power

**Activate Forest (Gain Food):**
1. Place action cube on rightmost empty forest space
2. Resolve each bird's power left-to-right
3. Gain food from birdfeeder (based on column position)

**Activate Grassland (Lay Eggs):**
1. Place action cube on rightmost empty grassland space
2. Resolve each bird's power left-to-right
3. Lay eggs (based on column position)
4. Distribute eggs to birds (player choice)

**Activate Wetland (Draw Cards):**
1. Place action cube on rightmost empty wetland space
2. Resolve each bird's power left-to-right
3. Draw cards from tray or deck (based on column position)

### Scoring

- Bird victory points (printed on card)
- Eggs on birds (1 VP each)
- Tucked cards (1 VP each)
- Cached food (1 VP each, some birds)
- End-of-round goal points
- Bonus card points

---

## Appendix: Insights from Settlers of Catan RL

> **Source**: [Learning to Play Settlers of Catan with Deep RL](https://settlers-rl.github.io/)
>
> These are additional architectural ideas drawn from a similar board game RL project. The Settlers implementation faced comparable challenges: complex action spaces, multi-player dynamics, and long-horizon planning.

### Multi-Head Action Architecture

Settlers' most important insight: **decompose complex actions into specialized heads**.

The Settlers implementation uses **13 specialized action heads** for different action types (placement, trading, development cards, robber movement, etc.). Each head has its own masking logic.

**Relevance to Wingspan**: The hybrid action space design in this document aligns with this approach. A potential extension:

| Head | Actions | Masking |
|------|---------|---------|
| Action Type | PLAY_BIRD, ACTIVATE_*, END_ROUND | Always valid subset |
| Habitat | Forest, Grassland, Wetland | Bird's valid habitats |
| Card Selection | Hand indices 0..15 | Cards actually in hand |
| Food Payment | Combinations | Affordable combos only |
| Egg Distribution | Bird indices | Birds with capacity |

This decomposition prevents the combinatorial explosion of a flat action space while maintaining expressiveness.

### Observation Encoding with Attention

Settlers uses:
- **Multi-head attention** for processing 19 board tiles
- **Embedding → Attention → Pooling** for variable-length development card sequences
- **Layer normalization** after attention and FC layers

**Relevance to Wingspan**: Critical for handling variable structures:

| Component | Encoding Strategy |
|-----------|-------------------|
| Hand cards (0-16) | Card embedding → Self-attention → Pool |
| Birds on mat (variable per habitat) | Per-habitat attention blocks |
| Tray cards (3 visible) | Fixed-size embedding |
| Opponent public info | Separate FC module |

This is more expressive than fixed-size encoding and handles the variable hand/board sizes naturally.

### Historical Opponent Sampling

Settlers addresses multi-agent non-stationarity by sampling opponents from earlier policy checkpoints:

> "Only one of the four players would be actively controlled by the policy being trained, whilst the other three would be controlled by earlier versions chosen at random."

**Relevance to Wingspan**: When moving to self-play (Phase 4+), maintain a checkpoint pool:

```
runs/<run>/checkpoints/
  opponent_pool/  ← Symlinks to ~10-20 diverse checkpoints
```

Sample opponents uniformly from the pool during training. This prevents overfitting to the current policy and improves robustness.

### Action Masking Validation

The Settlers paper confirms action masking is essential: they add negative infinity to logits before softmax for invalid actions. This matches the design already in this document.

**Key lesson from Settlers**: They regret not masking invalid trades as unavailable actions, leading to wasted computation proposing impossible trades. For Wingspan, ensure all sub-decisions (food payment, egg distribution) are properly masked.

### What Doesn't Apply

| Settlers Approach | Applies to Wingspan? | Rationale |
|-------------------|---------------------|-----------|
| MAPPO (as originally designed) | Partially | Original MAPPO is cooperative; see "Global Critic" section below |
| Value function normalization (fixed mean/std) | No | Standard advantage normalization is sufficient |
| Recurrent trading heads | No | Wingspan trading is simpler (no negotiation) |
| Semi-async distributed PPO | No | Vectorized sync PPO is simpler and sufficient |
| Forward search (MCTS) | Maybe later | Nice-to-have for stronger play, not essential initially |

### Global Critic for Competitive Games (CTDE)

> **Update**: Recent research (2022-2024) proves that centralized critics work for competitive zero-sum games,
> contradicting the earlier Settlers-based assessment. Key papers:
> - [Cen et al. 2022](https://arxiv.org/abs/2210.01050): Entropy-regularized OMWU with **simultaneous symmetric updates**
> - [Alacaoglu et al. 2022](https://proceedings.mlr.press/v162/alacaoglu22a.html): Natural Actor-Critic for zero-sum Markov games

**The CTDE Paradigm (Centralized Training, Decentralized Execution):**

| Component | Training | Execution |
|-----------|----------|-----------|
| Critic (Value) | Sees global state (all players' info) | Not used |
| Policy (Actor) | Sees only local observation | Sees only local observation |

**Why This Works for Zero-Sum:**
- For two-player zero-sum: V₂*(s) = -V₁*(s) (values are exact opposites)
- A single omniscient critic can compute both players' values
- Simultaneous updates provably converge to Nash equilibrium (Cen 2022, Alacaoglu 2022)
- No need to freeze one player while training the other

**Theoretical Guarantees:**
- Cen et al.: Õ(|S|/(1-γ)⁵·ε) complexity, last-iterate linear convergence
- Alacaoglu et al.: Sample complexity matches single-agent RL

**Caveats for Wingspan:**
1. **N-player complexity**: Proofs are for 2-player; Wingspan has 1-5 players
2. **Not strictly zero-sum**: Wingspan scores aren't exact opposites (your gain ≠ others' loss)
3. **Implementation complexity**: Global critic needs all players' hidden info during training

**Recommendation**: Consider CTDE for 2-player Wingspan variant first. For N-player,
historical opponent sampling (Settlers approach) may be simpler with comparable results.

### Network Architecture Sketch

Based on Settlers' approach, a Wingspan-specific architecture could look like:

```
Observation Inputs:
  ├─ Hand Cards ──────┬─ Card Embedding ─┬─ Self-Attention ─┬─ Pool ─┐
  ├─ Board Birds ─────┤                  │                  │        │
  ├─ Food/Eggs ───────┴─ MLP ────────────┴──────────────────┴─ Concat ─► Backbone
  ├─ Tray Cards ───────────────────────────────────────────────────────┘
  ├─ Round Goals ─────┐
  └─ Opponent Info ───┘

Backbone (shared) → [Action Type Head]   → Softmax × Mask
                 ├─ [Habitat Head]       → Softmax × Mask
                 ├─ [Card Selection]     → Softmax × Mask
                 ├─ [Food Payment]       → Multi-head × Mask
                 └─ [Egg Distribution]   → Iterative assignment

Value Head → [batch, num_players] or omniscient [batch, 1] with CTDE
```

This modular design allows:
1. Attention-based encoding for variable structures
2. Per-head action masking
3. Extensibility for power choices (additional heads)

### Summary Table

| Settlers Learning | Priority for Wingspan | Notes |
|-------------------|----------------------|-------|
| Multi-head action architecture | High | Essential for action space complexity |
| Action masking | Done | Already in design |
| Attention for variable structures | High | Cards, birds on mat |
| Historical opponent sampling | Medium | For Phase 4+ self-play |
| Dense reward shaping | Medium | Score delta approach already planned |
| Global Critic (CTDE) | Medium | Proven for 2-player zero-sum; N-player Wingspan needs more research |
