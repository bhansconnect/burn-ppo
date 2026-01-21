/// Skull (Skull & Roses) environment with 2-6 player support
///
/// A bluffing game where players place coasters (roses/skulls) face-down,
/// bid on how many they can reveal without hitting a skull, then reveal.
/// Features hidden information, bidder's choice reveal, and elimination-based gameplay.
use crate::env::{Environment, GameOutcome};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

// Game constants
const MAX_PLAYERS: usize = 6;
const CARDS_PER_PLAYER: usize = 4; // 3 roses + 1 skull
const ROSES_PER_PLAYER: usize = 3;
const MAX_BID: usize = MAX_PLAYERS * CARDS_PER_PLAYER; // 24
const WINS_TO_WIN: usize = 2;

// Action space layout
const PLACE_SKULL: usize = 0;
const PLACE_ROSE: usize = 1;
const BID_BASE: usize = 2; // bids 1-24 at indices 2-25
const PASS_ACTION: usize = BID_BASE + MAX_BID; // 26
const REVEAL_PLAYER_BASE: usize = PASS_ACTION + 1; // 27-32

const ACTION_COUNT: usize = REVEAL_PLAYER_BASE + MAX_PLAYERS; // 33

// Observation space layout
// NOTE: Per-player arrays use RELATIVE indexing (0 = current player, 1 = next clockwise, etc.)
const OBS_OWN_HAND: usize = CARDS_PER_PLAYER; // 4: which cards still in hand [skull, rose1, rose2, rose3]
const OBS_OWN_STACK: usize = CARDS_PER_PLAYER; // 4: what's in own stack (known to self)
const OBS_STACK_SIZES: usize = MAX_PLAYERS; // 6: stack sizes per player (relative, normalized)
const OBS_COASTERS: usize = MAX_PLAYERS; // 6: remaining coasters (relative, normalized 0-1)
const OBS_ALIVE_FLAGS: usize = MAX_PLAYERS; // 6: player alive flags (relative)
const OBS_PLAYER_EXISTS: usize = MAX_PLAYERS; // 6: which player slots are active (relative)
const OBS_SEAT_POSITION: usize = MAX_PLAYERS; // 6: absolute seat position one-hot
const OBS_PHASE: usize = 3; // 3: phase one-hot (placing/bidding/revealing)
const OBS_CURRENT_BID: usize = 1; // 1: current bid normalized
const OBS_CURRENT_BIDDER: usize = MAX_PLAYERS; // 6: who is bidder one-hot (relative)
const OBS_PASSED: usize = MAX_PLAYERS; // 6: who has passed in bidding (relative)
const OBS_WIN_COUNT: usize = MAX_PLAYERS; // 6: successful challenges per player (relative)
const OBS_REVEALED_COUNT: usize = MAX_PLAYERS; // 6: cards revealed per stack (relative)
const OBS_NUM_PLAYERS: usize = MAX_PLAYERS - 1; // 5: one-hot for 2-6 players

// Bid history: last 8 bids, each encoded as bidder (6) + bid normalized (1) + is_pass (1) = 8
const BID_HISTORY_SIZE: usize = 8;
const BID_HISTORY_ENTRY_SIZE: usize = MAX_PLAYERS + 2; // 8
const OBS_BID_HISTORY: usize = BID_HISTORY_SIZE * BID_HISTORY_ENTRY_SIZE; // 64

const OBSERVATION_DIM: usize = OBS_OWN_HAND
    + OBS_OWN_STACK
    + OBS_STACK_SIZES
    + OBS_COASTERS
    + OBS_ALIVE_FLAGS
    + OBS_PLAYER_EXISTS
    + OBS_SEAT_POSITION
    + OBS_PHASE
    + OBS_CURRENT_BID
    + OBS_CURRENT_BIDDER
    + OBS_PASSED
    + OBS_WIN_COUNT
    + OBS_REVEALED_COUNT
    + OBS_NUM_PLAYERS
    + OBS_BID_HISTORY; // 4+4+6+6+6+6+6+3+1+6+6+6+6+5+64 = 135

/// Card types in Skull
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Card {
    Skull,
    Rose,
}

/// Game phases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Placing,
    Bidding,
    Revealing,
}

/// Entry in bid history
#[derive(Debug, Clone, Copy)]
struct BidEntry {
    player: usize,
    bid: usize, // 0 = pass, 1-24 = bid amount
}

/// Skull game state
#[derive(Debug, Clone)]
pub struct Skull {
    // Configuration
    num_players: usize,
    reward_shaping_coef: f32,

    // Per-player persistent state (across rounds)
    has_trap: [bool; MAX_PLAYERS],    // still has skull coaster
    rose_count: [usize; MAX_PLAYERS], // roses remaining (0-3)
    wins: [usize; MAX_PLAYERS],       // successful challenges (2 = game win)

    // Per-player round state
    stack: [Vec<Card>; MAX_PLAYERS], // cards played this round
    passed: [bool; MAX_PLAYERS],     // who has passed in current bidding/placing

    // Round state
    phase: Phase,
    current_player: usize,
    round_starter: usize,

    // Bidding state
    current_bid: usize, // 0 = no bid yet
    current_bidder: Option<usize>,
    bid_history: VecDeque<BidEntry>,

    // Reveal state
    revealed: [usize; MAX_PLAYERS], // how many cards revealed per stack
    roses_found: usize,
    must_reveal_own: bool,           // bidder must reveal own stack first
    last_skull_owner: Option<usize>, // player whose skull was last revealed (for next round starter)

    // Game state
    elimination_order: Vec<usize>,
    game_over: bool,
    winner: Option<usize>,
    rng: StdRng,
}

impl Skull {
    /// Create a new Skull game with specified player count
    pub fn new_with_players(num_players: usize, reward_shaping_coef: f32, seed: u64) -> Self {
        assert!(
            (2..=MAX_PLAYERS).contains(&num_players),
            "Player count must be 2-6"
        );

        let mut game = Self {
            num_players,
            reward_shaping_coef,
            has_trap: [true; MAX_PLAYERS],
            rose_count: [ROSES_PER_PLAYER; MAX_PLAYERS],
            wins: [0; MAX_PLAYERS],
            stack: Default::default(),
            passed: [false; MAX_PLAYERS],
            phase: Phase::Placing,
            current_player: 0,
            round_starter: 0,
            current_bid: 0,
            current_bidder: None,
            bid_history: VecDeque::with_capacity(BID_HISTORY_SIZE),
            revealed: [0; MAX_PLAYERS],
            roses_found: 0,
            must_reveal_own: false,
            last_skull_owner: None,
            elimination_order: Vec::new(),
            game_over: false,
            winner: None,
            rng: StdRng::seed_from_u64(seed),
        };

        // Initialize stacks
        for i in 0..MAX_PLAYERS {
            game.stack[i] = Vec::with_capacity(CARDS_PER_PLAYER);
        }

        // Mark non-existent players as eliminated
        for i in num_players..MAX_PLAYERS {
            game.has_trap[i] = false;
            game.rose_count[i] = 0;
        }

        game
    }

    /// Check if a player is alive (has coasters and exists)
    fn is_alive(&self, player: usize) -> bool {
        player < self.num_players && (self.has_trap[player] || self.rose_count[player] > 0)
    }

    /// Get total coaster count for a player
    fn coaster_count(&self, player: usize) -> usize {
        if player >= self.num_players {
            return 0;
        }
        let skull = usize::from(self.has_trap[player]);
        skull + self.rose_count[player]
    }

    /// Count alive players
    fn alive_count(&self) -> usize {
        (0..self.num_players).filter(|&p| self.is_alive(p)).count()
    }

    /// Get next alive player after given player
    fn next_alive_player(&self, from: usize) -> usize {
        let mut next = (from + 1) % self.num_players;
        let start = next;
        loop {
            if self.is_alive(next) {
                return next;
            }
            next = (next + 1) % self.num_players;
            if next == start {
                // Should not happen if game not over
                return from;
            }
        }
    }

    /// Get next player who hasn't passed (for bidding phase)
    fn next_non_passed_player(&self, from: usize) -> Option<usize> {
        let mut next = (from + 1) % self.num_players;
        let start = next;
        loop {
            if self.is_alive(next) && !self.passed[next] {
                return Some(next);
            }
            next = (next + 1) % self.num_players;
            if next == start {
                return None;
            }
        }
    }

    /// Count how many alive players haven't passed
    fn non_passed_count(&self) -> usize {
        (0..self.num_players)
            .filter(|&p| self.is_alive(p) && !self.passed[p])
            .count()
    }

    /// Get total cards currently in all stacks
    fn total_cards_in_stacks(&self) -> usize {
        (0..self.num_players).map(|p| self.stack[p].len()).sum()
    }

    /// Check if player has skull in hand (not in stack)
    fn has_trap_in_hand(&self, player: usize) -> bool {
        self.has_trap[player] && !self.stack[player].contains(&Card::Skull)
    }

    /// Count roses in hand (not in stack)
    fn roses_in_hand(&self, player: usize) -> usize {
        let roses_in_stack = self.stack[player]
            .iter()
            .filter(|&&c| c == Card::Rose)
            .count();
        self.rose_count[player].saturating_sub(roses_in_stack)
    }

    /// Get unrevealed card count for a player's stack
    fn unrevealed_count(&self, player: usize) -> usize {
        self.stack[player]
            .len()
            .saturating_sub(self.revealed[player])
    }

    /// Place a card from hand to stack
    fn place_card(&mut self, player: usize, card: Card) {
        self.stack[player].push(card);
    }

    /// Reveal top unrevealed card from a player's stack
    /// Returns the card and whether it was a skull
    fn reveal_card(&mut self, player: usize) -> (Card, bool) {
        let idx = self.stack[player].len() - 1 - self.revealed[player];
        let card = self.stack[player][idx];
        self.revealed[player] += 1;
        let is_skull = card == Card::Skull;
        if !is_skull {
            self.roses_found += 1;
        }
        (card, is_skull)
    }

    /// Bidder loses a coaster (randomly chosen)
    fn lose_coaster(&mut self, player: usize) {
        let total = self.coaster_count(player);
        if total == 0 {
            return;
        }

        // Randomly choose which coaster to lose
        let choice = self.rng.gen_range(0..total);
        if self.has_trap[player] && choice == 0 {
            self.has_trap[player] = false;
        } else {
            self.rose_count[player] -= 1;
        }

        // Check for elimination
        if self.coaster_count(player) == 0 {
            self.elimination_order.push(player);
        }
    }

    /// Start a new round (after reveal resolution)
    fn start_new_round(&mut self, starter: usize) {
        // Clear round state
        for i in 0..MAX_PLAYERS {
            self.stack[i].clear();
            self.passed[i] = false;
            self.revealed[i] = 0;
        }
        self.phase = Phase::Placing;
        self.current_bid = 0;
        self.current_bidder = None;
        self.bid_history.clear();
        self.roses_found = 0;
        self.must_reveal_own = false;
        self.last_skull_owner = None;

        // Find next alive player from starter
        if self.is_alive(starter) {
            self.current_player = starter;
        } else {
            self.current_player = self.next_alive_player(starter);
        }
        self.round_starter = self.current_player;
    }

    /// Calculate terminal rewards based on placement.
    /// Uses `compute_placements()` for proper ordering by wins/coasters/elimination.
    /// Handles ties by averaging rewards for tied positions.
    fn calculate_final_rewards(&self) -> Vec<f32> {
        use std::collections::HashMap;

        let n = self.num_players;
        let placements = self.compute_placements();
        let mut rewards = vec![0.0; n];

        // Group players by placement for tie handling
        let mut placement_groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (player, &placement) in placements.iter().enumerate() {
            placement_groups.entry(placement).or_default().push(player);
        }

        // Calculate reward for each placement, averaging for ties
        // reward(p) = 1.0 - 2.0 * (p - 1) / (n - 1)
        for (&placement, players) in &placement_groups {
            let group_size = players.len();

            // Average the rewards for positions [placement, placement + group_size - 1]
            let total_reward: f32 = (0..group_size)
                .map(|offset| {
                    let effective_placement = placement + offset;
                    if n > 1 {
                        1.0 - 2.0 * (effective_placement as f32 - 1.0) / (n as f32 - 1.0)
                    } else {
                        0.0
                    }
                })
                .sum();
            let avg_reward = total_reward / group_size as f32;

            for &player in players {
                rewards[player] = avg_reward;
            }
        }

        rewards
    }

    /// Calculate reward shaping for round end (if enabled)
    fn calculate_round_rewards(&self, success: bool, bidder: usize) -> Vec<f32> {
        let mut rewards = vec![0.0; self.num_players];

        if self.reward_shaping_coef > 0.0 {
            // Small survival bonus for alive players
            for (p, reward) in rewards.iter_mut().enumerate().take(self.num_players) {
                if self.is_alive(p) {
                    *reward += 0.25 * self.reward_shaping_coef;
                }
            }
            // Extra bonus/penalty for bidder
            if success {
                rewards[bidder] += self.reward_shaping_coef;
            } else {
                rewards[bidder] -= self.reward_shaping_coef;
            }
        }

        rewards
    }

    /// Compute placements for all players using competition ranking (1224).
    /// Returns Vec<usize> where index = player, value = placement (1-indexed).
    ///
    /// Ordering criteria (best to worst):
    /// 1. Winner (always 1st)
    /// 2. More wins
    /// 3. More coasters remaining
    /// 4. Later in elimination order (non-eliminated > eliminated)
    fn compute_placements(&self) -> Vec<usize> {
        let n = self.num_players;
        let elim_len = self.elimination_order.len();

        // Build sortable entries: (player_idx, is_winner, wins, coasters, elimination_rank)
        // elimination_rank: higher = better
        //   - Not eliminated: elim_len (highest possible)
        //   - Eliminated: position in elimination_order (0 = first out = worst)
        let mut entries: Vec<(usize, bool, usize, usize, usize)> = (0..n)
            .map(|p| {
                let is_winner = self.winner == Some(p);
                let elim_rank =
                    if let Some(pos) = self.elimination_order.iter().position(|&x| x == p) {
                        pos // First eliminated (pos=0) = worst, later = better
                    } else {
                        elim_len // Not eliminated = best
                    };
                (p, is_winner, self.wins[p], self.coaster_count(p), elim_rank)
            })
            .collect();

        // Sort by criteria (descending for all - higher is better)
        entries.sort_by(|a, b| {
            b.1.cmp(&a.1) // is_winner (true > false)
                .then(b.2.cmp(&a.2)) // wins
                .then(b.3.cmp(&a.3)) // coasters
                .then(b.4.cmp(&a.4)) // elimination_rank
        });

        // Assign placements with competition ranking (1224)
        let mut placements = vec![0; n];
        let mut current_placement = 1;
        let mut i = 0;

        while i < n {
            // Find all players tied with entries[i]
            let mut j = i + 1;
            while j < n
                && entries[j].1 == entries[i].1
                && entries[j].2 == entries[i].2
                && entries[j].3 == entries[i].3
                && entries[j].4 == entries[i].4
            {
                j += 1;
            }

            // All players from i to j-1 are tied, give them the same placement
            for k in i..j {
                placements[entries[k].0] = current_placement;
            }

            // Next placement skips the tied positions (competition ranking)
            current_placement += j - i;
            i = j;
        }

        placements
    }

    /// Encode current state as observation vector
    /// Uses RELATIVE player indexing: 0 = current player, 1 = next clockwise, etc.
    fn get_observation(&self) -> Vec<f32> {
        let mut obs = vec![0.0; OBSERVATION_DIM];
        let mut idx = 0;

        let player = self.current_player;
        let n = self.num_players;

        // Own hand: [has_trap_in_hand, rose1, rose2, rose3]
        obs[idx] = if self.has_trap_in_hand(player) {
            1.0
        } else {
            0.0
        };
        let roses_in_hand = self.roses_in_hand(player);
        for i in 0..ROSES_PER_PLAYER {
            obs[idx + 1 + i] = if i < roses_in_hand { 1.0 } else { 0.0 };
        }
        idx += OBS_OWN_HAND;

        // Own stack contents (known to self): encode as [skull_in_stack, rose_count_in_stack/3]
        // Actually encode each position as skull=1, rose=0
        for i in 0..CARDS_PER_PLAYER {
            if i < self.stack[player].len() {
                obs[idx + i] = if self.stack[player][i] == Card::Skull {
                    1.0
                } else {
                    0.0
                };
            }
        }
        idx += OBS_OWN_STACK;

        // Stack sizes per player (normalized by max cards) - RELATIVE indexing
        for rel_idx in 0..MAX_PLAYERS {
            if rel_idx < n {
                let abs_idx = (rel_idx + player) % n;
                obs[idx + rel_idx] = self.stack[abs_idx].len() as f32 / CARDS_PER_PLAYER as f32;
            }
        }
        idx += OBS_STACK_SIZES;

        // Coasters remaining (normalized) - RELATIVE indexing
        for rel_idx in 0..MAX_PLAYERS {
            if rel_idx < n {
                let abs_idx = (rel_idx + player) % n;
                obs[idx + rel_idx] = self.coaster_count(abs_idx) as f32 / CARDS_PER_PLAYER as f32;
            }
        }
        idx += OBS_COASTERS;

        // Alive flags - RELATIVE indexing
        for rel_idx in 0..MAX_PLAYERS {
            if rel_idx < n {
                let abs_idx = (rel_idx + player) % n;
                obs[idx + rel_idx] = if self.is_alive(abs_idx) { 1.0 } else { 0.0 };
            }
        }
        idx += OBS_ALIVE_FLAGS;

        // Player exists flags - RELATIVE indexing
        for rel_idx in 0..MAX_PLAYERS {
            if rel_idx < n {
                obs[idx + rel_idx] = 1.0; // All relative indices < num_players exist
            }
        }
        idx += OBS_PLAYER_EXISTS;

        // Seat position one-hot - ABSOLUTE (for position awareness)
        obs[idx + player] = 1.0;
        idx += OBS_SEAT_POSITION;

        // Phase one-hot
        let phase_idx = match self.phase {
            Phase::Placing => 0,
            Phase::Bidding => 1,
            Phase::Revealing => 2,
        };
        obs[idx + phase_idx] = 1.0;
        idx += OBS_PHASE;

        // Current bid (normalized)
        obs[idx] = self.current_bid as f32 / MAX_BID as f32;
        idx += OBS_CURRENT_BID;

        // Current bidder one-hot - RELATIVE indexing
        if let Some(bidder) = self.current_bidder {
            let rel_bidder = (bidder + n - player) % n;
            obs[idx + rel_bidder] = 1.0;
        }
        idx += OBS_CURRENT_BIDDER;

        // Passed flags - RELATIVE indexing
        for rel_idx in 0..MAX_PLAYERS {
            if rel_idx < n {
                let abs_idx = (rel_idx + player) % n;
                obs[idx + rel_idx] = if self.passed[abs_idx] { 1.0 } else { 0.0 };
            }
        }
        idx += OBS_PASSED;

        // Win count per player - RELATIVE indexing
        for rel_idx in 0..MAX_PLAYERS {
            if rel_idx < n {
                let abs_idx = (rel_idx + player) % n;
                obs[idx + rel_idx] = self.wins[abs_idx] as f32 / WINS_TO_WIN as f32;
            }
        }
        idx += OBS_WIN_COUNT;

        // Revealed count per stack (normalized) - RELATIVE indexing
        for rel_idx in 0..MAX_PLAYERS {
            if rel_idx < n {
                let abs_idx = (rel_idx + player) % n;
                obs[idx + rel_idx] = self.revealed[abs_idx] as f32 / CARDS_PER_PLAYER as f32;
            }
        }
        idx += OBS_REVEALED_COUNT;

        // Number of players one-hot (2-6 maps to indices 0-4)
        if (2..=MAX_PLAYERS).contains(&n) {
            obs[idx + n - 2] = 1.0;
        }
        idx += OBS_NUM_PLAYERS;

        // Bid history - RELATIVE player indexing
        for (i, entry) in self.bid_history.iter().enumerate() {
            let base = idx + i * BID_HISTORY_ENTRY_SIZE;
            let rel_player = (entry.player + n - player) % n;
            obs[base + rel_player] = 1.0; // Player one-hot - RELATIVE
            if entry.bid == 0 {
                obs[base + MAX_PLAYERS + 1] = 1.0; // Is pass flag
            } else {
                obs[base + MAX_PLAYERS] = entry.bid as f32 / MAX_BID as f32; // Bid value
            }
        }

        obs
    }

    /// Transition to bidding phase
    fn transition_to_bidding(&mut self, bidder: usize, bid: usize) {
        self.phase = Phase::Bidding;
        self.current_bid = bid;
        self.current_bidder = Some(bidder);
        self.bid_history.push_back(BidEntry {
            player: bidder,
            bid,
        });

        // Check if bid equals total cards (immediate reveal)
        if bid == self.total_cards_in_stacks() {
            self.transition_to_revealing();
        } else {
            // Move to next non-passed player
            if let Some(next) = self.next_non_passed_player(bidder) {
                self.current_player = next;
            }
        }
    }

    /// Transition to revealing phase
    fn transition_to_revealing(&mut self) {
        self.phase = Phase::Revealing;
        self.current_player = self
            .current_bidder
            .expect("transition_to_revealing called without bidder");
        self.must_reveal_own = true;
        self.roses_found = 0;
        for i in 0..MAX_PLAYERS {
            self.revealed[i] = 0;
        }
    }

    /// Check if only one player hasn't passed and trigger reveal
    fn check_bidding_end(&mut self) {
        let non_passed = self.non_passed_count();
        if non_passed == 1 {
            // Last non-passed player becomes bidder and reveals
            let bidder = (0..self.num_players)
                .find(|&p| self.is_alive(p) && !self.passed[p])
                .expect("non_passed_count is 1 but no non-passed player found");
            self.current_bidder = Some(bidder);
            self.transition_to_revealing();
        } else if let Some(next) = self.next_non_passed_player(self.current_player) {
            self.current_player = next;
        }
    }
}

impl Environment for Skull {
    const OBSERVATION_DIM: usize = OBSERVATION_DIM;
    const ACTION_COUNT: usize = ACTION_COUNT;
    const NAME: &'static str = "skull";
    const NUM_PLAYERS: usize = MAX_PLAYERS;

    // Bluffing game needs stochasticity for exploration
    const EVAL_TEMP: f32 = 1.0;

    // Skull supports 2-6 players at runtime
    const VARIABLE_PLAYER_COUNT: bool = true;

    fn new(seed: u64) -> Self {
        // Default to 4 players
        Self::new_with_players(4, 0.0, seed)
    }

    fn reset(&mut self) -> Vec<f32> {
        // Reset all state
        for i in 0..MAX_PLAYERS {
            if i < self.num_players {
                self.has_trap[i] = true;
                self.rose_count[i] = ROSES_PER_PLAYER;
            } else {
                self.has_trap[i] = false;
                self.rose_count[i] = 0;
            }
            self.wins[i] = 0;
            self.stack[i].clear();
            self.passed[i] = false;
            self.revealed[i] = 0;
        }

        self.phase = Phase::Placing;
        self.current_player = self.rng.gen_range(0..self.num_players);
        self.round_starter = self.current_player;
        self.current_bid = 0;
        self.current_bidder = None;
        self.bid_history.clear();
        self.roses_found = 0;
        self.must_reveal_own = false;
        self.last_skull_owner = None;
        self.elimination_order.clear();
        self.game_over = false;
        self.winner = None;

        self.get_observation()
    }

    fn current_player(&self) -> usize {
        self.current_player
    }

    fn step(&mut self, action: usize) -> (Vec<f32>, Vec<f32>, bool) {
        let mut rewards = vec![0.0; self.num_players];

        if self.game_over {
            return (self.get_observation(), rewards, true);
        }

        let player = self.current_player;

        // Validate action using mask
        let mask = self
            .action_mask()
            .expect("action_mask should always return Some");
        if action >= ACTION_COUNT || !mask[action] {
            // Invalid action - game over with no rewards
            self.game_over = true;
            return (self.get_observation(), rewards, true);
        }

        match self.phase {
            Phase::Placing => {
                if action == PLACE_SKULL {
                    self.place_card(player, Card::Skull);
                    self.current_player = self.next_alive_player(player);
                } else if action == PLACE_ROSE {
                    self.place_card(player, Card::Rose);
                    self.current_player = self.next_alive_player(player);
                } else if (BID_BASE..PASS_ACTION).contains(&action) {
                    // Start bidding
                    let bid = action - BID_BASE + 1;
                    self.transition_to_bidding(player, bid);
                } else if action == PASS_ACTION {
                    self.passed[player] = true;
                    // Check if all but one have passed (that one must bid)
                    let non_passed = self.non_passed_count();
                    if non_passed == 1 {
                        // Force the remaining player to start bidding phase
                        // They'll need to place a card or bid on their turn
                        let remaining = (0..self.num_players)
                            .find(|&p| self.is_alive(p) && !self.passed[p])
                            .expect("non_passed is 1 but no non-passed player found");
                        self.current_player = remaining;
                    } else if non_passed == 0 {
                        // Everyone passed without bidding - shouldn't happen with proper masking
                        // Start new round
                        self.start_new_round(self.round_starter);
                    } else {
                        self.current_player = self.next_alive_player(player);
                        // Skip passed players
                        while self.passed[self.current_player] && self.current_player != player {
                            self.current_player = self.next_alive_player(self.current_player);
                        }
                    }
                }
            }

            Phase::Bidding => {
                if (BID_BASE..PASS_ACTION).contains(&action) {
                    // Raise bid
                    let bid = action - BID_BASE + 1;
                    self.current_bid = bid;
                    self.current_bidder = Some(player);

                    if self.bid_history.len() >= BID_HISTORY_SIZE {
                        self.bid_history.pop_front();
                    }
                    self.bid_history.push_back(BidEntry { player, bid });

                    // Check if bid matches total cards
                    if bid == self.total_cards_in_stacks() {
                        self.transition_to_revealing();
                    } else if let Some(next) = self.next_non_passed_player(player) {
                        self.current_player = next;
                    }
                } else if action == PASS_ACTION {
                    self.passed[player] = true;

                    if self.bid_history.len() >= BID_HISTORY_SIZE {
                        self.bid_history.pop_front();
                    }
                    self.bid_history.push_back(BidEntry { player, bid: 0 });

                    self.check_bidding_end();
                }
            }

            Phase::Revealing => {
                let bidder = self
                    .current_bidder
                    .expect("Revealing phase requires a bidder");
                let target = if action >= REVEAL_PLAYER_BASE {
                    action - REVEAL_PLAYER_BASE
                } else {
                    // Should not happen with proper masking
                    return (self.get_observation(), rewards, false);
                };

                // Reveal a card
                let (_, is_skull) = self.reveal_card(target);

                // Check if own stack is now fully revealed
                if target == bidder && self.unrevealed_count(bidder) == 0 {
                    self.must_reveal_own = false;
                }

                if is_skull {
                    // Track whose skull was revealed (for determining next round starter if bidder is eliminated)
                    self.last_skull_owner = Some(target);

                    // Bidder loses a coaster
                    self.lose_coaster(bidder);
                    rewards = self.calculate_round_rewards(false, bidder);

                    // Check for game end (elimination or last player)
                    if self.alive_count() <= 1 {
                        self.game_over = true;
                        // Find winner (last alive player)
                        self.winner = (0..self.num_players).find(|&p| self.is_alive(p));
                        rewards = self.calculate_final_rewards();
                    } else {
                        // Determine next round starter based on rules:
                        // - If bidder is still alive, they start next round
                        // - If bidder is eliminated, the person whose skull was revealed becomes first player
                        let next_starter = if self.is_alive(bidder) {
                            bidder
                        } else {
                            // Bidder was eliminated - skull owner becomes first player
                            // (if skull owner is also eliminated somehow, fall back to next alive)
                            if self.is_alive(target) {
                                target
                            } else {
                                self.next_alive_player(target)
                            }
                        };
                        self.start_new_round(next_starter);
                    }
                } else if self.roses_found >= self.current_bid {
                    // Success! Bidder gains a win
                    self.wins[bidder] += 1;
                    rewards = self.calculate_round_rewards(true, bidder);

                    // Check for game win conditions
                    let is_last_player = self.alive_count() == 1;
                    let has_enough_wins = self.wins[bidder] >= WINS_TO_WIN;

                    if has_enough_wins || is_last_player {
                        // Game won!
                        self.game_over = true;
                        self.winner = Some(bidder);
                        rewards = self.calculate_final_rewards();
                    } else {
                        // Winner starts next round
                        self.start_new_round(bidder);
                    }
                }
                // If neither skull nor success, bidder continues revealing
            }
        }

        (self.get_observation(), rewards, self.game_over)
    }

    fn action_mask(&self) -> Option<Vec<bool>> {
        let mut mask = vec![false; ACTION_COUNT];

        if self.game_over {
            return Some(mask);
        }

        let player = self.current_player;

        match self.phase {
            Phase::Placing => {
                // Can place skull if has it in hand
                if self.has_trap_in_hand(player) {
                    mask[PLACE_SKULL] = true;
                }

                // Can place rose if has roses in hand
                if self.roses_in_hand(player) > 0 {
                    mask[PLACE_ROSE] = true;
                }

                // Can start bidding if has placed at least one card
                if !self.stack[player].is_empty() {
                    let total_cards = self.total_cards_in_stacks();
                    let min_bid = (self.current_bid + 1).max(1);
                    for bid in min_bid..=total_cards {
                        mask[BID_BASE + bid - 1] = true;
                    }

                    // Can pass if other players still need to act
                    // (but not if we're the only non-passed player)
                    if self.non_passed_count() > 1 {
                        mask[PASS_ACTION] = true;
                    }
                }
            }

            Phase::Bidding => {
                // Can raise bid
                let total_cards = self.total_cards_in_stacks();
                for bid in (self.current_bid + 1)..=total_cards {
                    mask[BID_BASE + bid - 1] = true;
                }

                // Can pass if not already passed and not the only non-passed player
                if !self.passed[player] && self.non_passed_count() > 1 {
                    mask[PASS_ACTION] = true;
                }
            }

            Phase::Revealing => {
                let bidder = self
                    .current_bidder
                    .expect("Revealing phase requires a bidder");

                // Only bidder can reveal
                if player == bidder {
                    if self.must_reveal_own && self.unrevealed_count(bidder) > 0 {
                        // Must reveal own stack first
                        mask[REVEAL_PLAYER_BASE + bidder] = true;
                    } else {
                        // Can reveal own or any opponent with unrevealed cards
                        if self.unrevealed_count(bidder) > 0 {
                            mask[REVEAL_PLAYER_BASE + bidder] = true;
                        }
                        for p in 0..self.num_players {
                            if p != bidder && self.unrevealed_count(p) > 0 {
                                mask[REVEAL_PLAYER_BASE + p] = true;
                            }
                        }
                    }
                }
            }
        }

        Some(mask)
    }

    fn game_outcome(&self) -> Option<GameOutcome> {
        if !self.game_over {
            return None;
        }
        Some(GameOutcome(self.compute_placements()))
    }

    fn render(&self) -> Option<String> {
        use std::fmt::Write;
        let mut s = String::new();

        // Game info
        let _ = writeln!(s, "=== Skull ({} players) ===", self.num_players);
        let _ = writeln!(
            s,
            "Phase: {:?} | Current Player: P{}",
            self.phase, self.current_player
        );

        if let Some(bidder) = self.current_bidder {
            let _ = writeln!(s, "Current Bid: {} by P{}", self.current_bid, bidder);
        }

        s.push('\n');

        // Player states
        for p in 0..self.num_players {
            let alive = if self.is_alive(p) { " " } else { "X" };
            let current = if p == self.current_player { ">" } else { " " };
            let wins = self.wins[p];
            let coasters = self.coaster_count(p);
            let stack_size = self.stack[p].len();
            let revealed = self.revealed[p];
            let passed = if self.passed[p] { " (passed)" } else { "" };

            let _ = writeln!(
                s,
                "{current}{alive} P{p}: {wins}W {coasters}C | Stack: {revealed}/{stack_size} revealed{passed}"
            );

            // Show own stack contents for current player
            if p == self.current_player && !self.stack[p].is_empty() {
                let stack_str: String = self.stack[p]
                    .iter()
                    .map(|c| match c {
                        Card::Skull => 'S',
                        Card::Rose => 'R',
                    })
                    .collect();
                let _ = writeln!(s, "   Stack contents: [{stack_str}]");
            }
        }

        if self.game_over {
            if let Some(winner) = self.winner {
                let _ = writeln!(s, "\nGame Over! Winner: P{winner}");
            }
        }

        Some(s)
    }

    fn describe_action(&self, action: usize) -> String {
        if action == PLACE_SKULL {
            "Place Skull".to_string()
        } else if action == PLACE_ROSE {
            "Place Rose".to_string()
        } else if (BID_BASE..PASS_ACTION).contains(&action) {
            let bid = action - BID_BASE + 1;
            format!("Bid {bid}")
        } else if action == PASS_ACTION {
            "Pass".to_string()
        } else if (REVEAL_PLAYER_BASE..ACTION_COUNT).contains(&action) {
            let target = action - REVEAL_PLAYER_BASE;
            format!("Reveal P{target}")
        } else {
            format!("Unknown action {action}")
        }
    }

    fn parse_action(&self, input: &str) -> Result<usize, String> {
        let input = input.trim().to_lowercase();

        if input == "skull" || input == "s" || input == "place skull" {
            return Ok(PLACE_SKULL);
        }
        if input == "rose" || input == "r" || input == "place rose" {
            return Ok(PLACE_ROSE);
        }
        if input == "pass" || input == "p" {
            return Ok(PASS_ACTION);
        }

        // Bid N
        if let Some(rest) = input.strip_prefix("bid ") {
            if let Ok(bid) = rest.trim().parse::<usize>() {
                if (1..=MAX_BID).contains(&bid) {
                    return Ok(BID_BASE + bid - 1);
                }
            }
            return Err(format!("Invalid bid: {rest}"));
        }

        // Just a number = bid
        if let Ok(bid) = input.parse::<usize>() {
            if (1..=MAX_BID).contains(&bid) {
                return Ok(BID_BASE + bid - 1);
            }
        }

        // Reveal PN
        if let Some(rest) = input.strip_prefix("reveal ") {
            let rest = rest.trim();
            if let Some(rest) = rest.strip_prefix('p') {
                if let Ok(p) = rest.parse::<usize>() {
                    if p < MAX_PLAYERS {
                        return Ok(REVEAL_PLAYER_BASE + p);
                    }
                }
            }
            return Err(format!("Invalid reveal target: {rest}"));
        }

        Err(format!("Unknown action: {input}"))
    }

    fn set_num_players(&mut self, num_players: usize) {
        assert!(
            (2..=MAX_PLAYERS).contains(&num_players),
            "Skull supports 2-{MAX_PLAYERS} players, got {num_players}"
        );
        self.num_players = num_players;
    }

    fn active_player_count(&self) -> usize {
        self.num_players
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_dim_matches_constant() {
        let mut env = Skull::new(42);
        env.reset();
        let obs = env.get_observation();
        assert_eq!(
            obs.len(),
            OBSERVATION_DIM,
            "Observation length should match OBSERVATION_DIM"
        );
    }

    #[test]
    fn test_action_count_matches_constant() {
        // Verify action indices are within bounds (compile-time checks)
        const _: () = {
            assert!(PLACE_SKULL < ACTION_COUNT);
            assert!(PLACE_ROSE < ACTION_COUNT);
            assert!(BID_BASE + MAX_BID - 1 < PASS_ACTION);
            assert!(PASS_ACTION < REVEAL_PLAYER_BASE);
            assert!(REVEAL_PLAYER_BASE + MAX_PLAYERS - 1 < ACTION_COUNT);
        };
        assert_eq!(REVEAL_PLAYER_BASE + MAX_PLAYERS, ACTION_COUNT);
    }

    #[test]
    fn test_new_game_initial_state() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        // All 4 players should have 4 coasters
        for p in 0..4 {
            assert!(env.is_alive(p));
            assert_eq!(env.coaster_count(p), 4);
            assert!(env.has_trap[p]);
            assert_eq!(env.rose_count[p], 3);
            assert!(env.stack[p].is_empty());
        }

        // Players 4-5 should not exist
        for p in 4..MAX_PLAYERS {
            assert!(!env.is_alive(p));
            assert_eq!(env.coaster_count(p), 0);
        }

        assert_eq!(env.phase, Phase::Placing);
        assert!(!env.game_over);
        assert_eq!(env.current_bid, 0);
        assert!(env.current_bidder.is_none());
    }

    #[test]
    fn test_relative_observation_symmetry() {
        // Verify that relative index 0 always contains current player's data
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        // Set up known state with different data for each player
        for p in 0..4 {
            env.stack[p].push(Card::Rose);
            env.wins[p] = p; // Different win counts
        }

        // Test from each player's perspective
        for current in 0..4 {
            env.current_player = current;
            let obs = env.get_observation();

            // Stack sizes start after OBS_OWN_HAND + OBS_OWN_STACK
            let stack_offset = OBS_OWN_HAND + OBS_OWN_STACK;

            // Relative index 0 should be current player's stack size
            let expected_stack = env.stack[current].len() as f32 / CARDS_PER_PLAYER as f32;
            assert_eq!(
                obs[stack_offset], expected_stack,
                "P{current}: Relative index 0 should be current player's stack"
            );

            // Seat position should be at the correct offset and encode absolute seat
            let seat_offset = OBS_OWN_HAND
                + OBS_OWN_STACK
                + OBS_STACK_SIZES
                + OBS_COASTERS
                + OBS_ALIVE_FLAGS
                + OBS_PLAYER_EXISTS;
            assert_eq!(
                obs[seat_offset + current],
                1.0,
                "P{current}: Seat position should encode absolute seat"
            );
        }
    }

    #[test]
    fn test_placing_phase_action_mask_full_hand() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        let mask = env.action_mask().unwrap();

        // Can place skull and rose
        assert!(mask[PLACE_SKULL], "Should be able to place skull");
        assert!(mask[PLACE_ROSE], "Should be able to place rose");

        // Cannot bid or pass with empty stack
        assert!(!mask[PASS_ACTION], "Cannot pass with empty stack");
        for bid in 1..=MAX_BID {
            assert!(!mask[BID_BASE + bid - 1], "Cannot bid with empty stack");
        }
    }

    #[test]
    fn test_placing_phase_action_mask_no_skull() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        // Place skull
        env.step(PLACE_SKULL);

        // Come back to same player (in 4 player game, after 3 others place)
        for _ in 0..3 {
            env.step(PLACE_ROSE);
        }

        let mask = env.action_mask().unwrap();

        // Cannot place skull again (already in stack)
        assert!(!mask[PLACE_SKULL], "Cannot place skull twice");
        assert!(mask[PLACE_ROSE], "Should be able to place rose");
    }

    #[test]
    fn test_placing_phase_action_mask_no_roses() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        let player = env.current_player;

        // Place all 3 roses (need to cycle through turns)
        for _ in 0..3 {
            // Place rose for current player
            while env.current_player != player {
                env.step(PLACE_ROSE);
            }
            env.step(PLACE_ROSE);
        }

        // Cycle back to original player
        while env.current_player != player {
            env.step(PLACE_ROSE);
        }

        let mask = env.action_mask().unwrap();

        // Can only place skull now (roses exhausted from hand)
        assert!(mask[PLACE_SKULL], "Should be able to place skull");
        assert!(!mask[PLACE_ROSE], "Cannot place rose when all in stack");
    }

    #[test]
    fn test_placing_phase_can_bid_with_card() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        // Place a card
        env.step(PLACE_ROSE);

        // Other players place
        for _ in 0..3 {
            env.step(PLACE_ROSE);
        }

        let mask = env.action_mask().unwrap();

        // Should be able to bid now (have cards in stack)
        assert!(mask[BID_BASE], "Should be able to bid 1");
        assert!(mask[BID_BASE + 3], "Should be able to bid 4 (total cards)");
        assert!(!mask[BID_BASE + 4], "Cannot bid more than total cards");
    }

    #[test]
    fn test_placing_advances_to_next_player() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        let first_player = env.current_player;
        env.step(PLACE_ROSE);

        assert_ne!(
            env.current_player, first_player,
            "Should advance to next player"
        );
    }

    #[test]
    fn test_bidding_phase_action_mask() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        // All players place a card
        for _ in 0..4 {
            env.step(PLACE_ROSE);
        }

        // Start bidding
        env.step(BID_BASE + 1 - 1); // Bid 1
        assert_eq!(env.phase, Phase::Bidding);

        let mask = env.action_mask().unwrap();

        // Can bid higher
        assert!(mask[BID_BASE + 2 - 1], "Should be able to bid 2");
        assert!(mask[BID_BASE + 4 - 1], "Should be able to bid 4");
        assert!(!mask[BID_BASE + 1 - 1], "Cannot bid same or lower");
        assert!(!mask[BID_BASE + 5 - 1], "Cannot bid more than total");

        // Can pass
        assert!(mask[PASS_ACTION], "Should be able to pass");
    }

    #[test]
    fn test_bidding_raises_bid() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        for _ in 0..4 {
            env.step(PLACE_ROSE);
        }

        let first_bidder = env.current_player;
        env.step(BID_BASE + 1 - 1); // Bid 1

        assert_eq!(env.current_bid, 1);
        assert_eq!(env.current_bidder, Some(first_bidder));

        env.step(BID_BASE + 2 - 1); // Bid 2
        assert_eq!(env.current_bid, 2);
        assert_ne!(env.current_bidder, Some(first_bidder));
    }

    #[test]
    fn test_bidding_pass_marks_player() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        for _ in 0..4 {
            env.step(PLACE_ROSE);
        }

        env.step(BID_BASE); // Bid 1
        let passing_player = env.current_player;
        env.step(PASS_ACTION);

        assert!(
            env.passed[passing_player],
            "Player should be marked as passed"
        );
    }

    #[test]
    fn test_bidding_max_bid_triggers_reveal() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        // Each player places 1 card = 4 total
        for _ in 0..4 {
            env.step(PLACE_ROSE);
        }

        // Bid the maximum (4)
        env.step(BID_BASE + 4 - 1);

        assert_eq!(
            env.phase,
            Phase::Revealing,
            "Max bid should trigger reveal phase"
        );
    }

    #[test]
    fn test_bidding_all_pass_triggers_reveal() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        for _ in 0..4 {
            env.step(PLACE_ROSE);
        }

        let bidder = env.current_player;
        env.step(BID_BASE); // Bid 1

        // Others pass
        env.step(PASS_ACTION);
        env.step(PASS_ACTION);
        env.step(PASS_ACTION);

        assert_eq!(env.phase, Phase::Revealing);
        assert_eq!(env.current_bidder, Some(bidder));
    }

    #[test]
    fn test_revealing_must_reveal_own_first() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        for _ in 0..4 {
            env.step(PLACE_ROSE);
        }

        let bidder = env.current_player;
        env.step(BID_BASE + 4 - 1); // Max bid -> reveal

        let mask = env.action_mask().unwrap();

        // Only own stack should be valid
        assert!(
            mask[REVEAL_PLAYER_BASE + bidder],
            "Must be able to reveal own"
        );
        for p in 0..4 {
            if p != bidder {
                assert!(
                    !mask[REVEAL_PLAYER_BASE + p],
                    "Cannot reveal others before own"
                );
            }
        }
    }

    #[test]
    fn test_revealing_can_choose_opponent_after_own() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        for _ in 0..4 {
            env.step(PLACE_ROSE);
        }

        let bidder = env.current_player;
        env.step(BID_BASE + 4 - 1); // Max bid

        // Reveal own stack
        env.step(REVEAL_PLAYER_BASE + bidder);

        let mask = env.action_mask().unwrap();

        // Should now be able to reveal any opponent
        for p in 0..4 {
            if p != bidder {
                assert!(
                    mask[REVEAL_PLAYER_BASE + p],
                    "Should be able to reveal P{p}"
                );
            }
        }
    }

    #[test]
    fn test_revealing_skull_ends_round() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        // Player 0 places skull
        let p0 = env.current_player;
        env.step(PLACE_SKULL);

        // Others place roses
        for _ in 0..3 {
            env.step(PLACE_ROSE);
        }

        // P0 bids max
        env.step(BID_BASE + 4 - 1);

        // P0 reveals own skull
        env.step(REVEAL_PLAYER_BASE + p0);

        // Should start new round, P0 loses a coaster
        assert_eq!(env.phase, Phase::Placing);
        assert!(
            env.coaster_count(p0) < 4,
            "Bidder should lose a coaster on skull"
        );
    }

    #[test]
    fn test_revealing_success_grants_win() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        // All place roses
        for _ in 0..4 {
            env.step(PLACE_ROSE);
        }

        let bidder = env.current_player;
        env.step(BID_BASE + 4 - 1); // Bid 4

        // Reveal all 4 roses (own first, then 3 opponents)
        env.step(REVEAL_PLAYER_BASE + bidder);
        let remaining: Vec<_> = (0..4).filter(|&p| p != bidder).collect();
        for p in remaining {
            env.step(REVEAL_PLAYER_BASE + p);
        }

        // Bidder should have 1 win
        assert_eq!(env.wins[bidder], 1, "Bidder should have 1 win");
        assert_eq!(env.phase, Phase::Placing, "Should start new round");
    }

    #[test]
    fn test_two_wins_ends_game() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();
        env.current_player = 0;
        env.wins[0] = 1;

        for p in 0..4 {
            env.stack[p].push(Card::Rose);
        }
        env.phase = Phase::Revealing;
        env.current_bid = 4;
        env.current_bidder = Some(0);
        env.must_reveal_own = true;

        // Reveal all roses
        env.step(REVEAL_PLAYER_BASE);
        env.step(REVEAL_PLAYER_BASE + 1);
        env.step(REVEAL_PLAYER_BASE + 2);
        env.step(REVEAL_PLAYER_BASE + 3);

        assert!(env.game_over, "Game should be over after 2 wins");
        assert_eq!(env.winner, Some(0), "P0 should be winner");
    }

    #[test]
    fn test_last_player_standing_wins() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        // Eliminate players 1, 2, 3
        for p in 1..4 {
            env.has_trap[p] = false;
            env.rose_count[p] = 0;
            env.elimination_order.push(p);
        }

        // P0 places and bids
        env.stack[0].push(Card::Rose);
        env.phase = Phase::Revealing;
        env.current_player = 0;
        env.current_bid = 1;
        env.current_bidder = Some(0);
        env.must_reveal_own = true;

        env.step(REVEAL_PLAYER_BASE);

        assert!(env.game_over, "Game should be over - last player standing");
        assert_eq!(env.winner, Some(0), "P0 should be winner");
    }

    #[test]
    fn test_elimination_when_no_coasters() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        // Give P0 only 1 coaster
        env.has_trap[0] = true;
        env.rose_count[0] = 0;

        // P0 places skull
        env.stack[0].push(Card::Skull);
        env.phase = Phase::Revealing;
        env.current_player = 0;
        env.current_bid = 1;
        env.current_bidder = Some(0);
        env.must_reveal_own = true;

        env.step(REVEAL_PLAYER_BASE);

        assert!(!env.is_alive(0), "P0 should be eliminated");
        assert!(
            env.elimination_order.contains(&0),
            "P0 should be in elimination order"
        );
    }

    #[test]
    fn test_two_player_game() {
        let mut env = Skull::new_with_players(2, 0.0, 42);
        env.reset();

        assert_eq!(env.num_players, 2);
        assert!(env.is_alive(0));
        assert!(env.is_alive(1));
        assert!(!env.is_alive(2));
        assert!(!env.is_alive(3));

        // Verify action mask excludes non-existent players
        for _ in 0..2 {
            env.step(PLACE_ROSE);
        }
        env.step(BID_BASE + 2 - 1); // Max bid

        let mask = env.action_mask().unwrap();
        assert!(!mask[REVEAL_PLAYER_BASE + 2], "P2 doesn't exist");
        assert!(!mask[REVEAL_PLAYER_BASE + 3], "P3 doesn't exist");
    }

    #[test]
    fn test_six_player_game() {
        let mut env = Skull::new_with_players(6, 0.0, 42);
        env.reset();

        for p in 0..6 {
            assert!(env.is_alive(p), "P{p} should be alive");
            assert_eq!(env.coaster_count(p), 4);
        }
    }

    #[test]
    fn test_deterministic_seeding() {
        let mut env1 = Skull::new_with_players(4, 0.0, 12345);
        let mut env2 = Skull::new_with_players(4, 0.0, 12345);

        env1.reset();
        env2.reset();

        assert_eq!(
            env1.current_player, env2.current_player,
            "Same seed should give same starting player"
        );

        // Play same sequence
        for _ in 0..4 {
            env1.step(PLACE_ROSE);
            env2.step(PLACE_ROSE);
        }

        assert_eq!(env1.current_player, env2.current_player);
        assert_eq!(env1.phase, env2.phase);
    }

    #[test]
    fn test_invalid_action_handling() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        // Invalid action index
        let (_, _, done) = env.step(ACTION_COUNT + 10);
        assert!(done, "Invalid action should end game");
    }

    #[test]
    fn test_action_after_game_over() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();
        env.game_over = true;

        let (_, rewards, done) = env.step(PLACE_ROSE);
        assert!(done, "Should return done=true");
        assert_eq!(rewards, vec![0.0; 4], "No rewards when game over");
    }

    #[test]
    fn test_reward_shaping_coefficient() {
        let mut env = Skull::new_with_players(4, 0.1, 42);
        env.reset();

        // Set up and complete a round with skull reveal
        env.stack[0].push(Card::Skull);
        for p in 1..4 {
            env.stack[p].push(Card::Rose);
        }
        env.phase = Phase::Revealing;
        env.current_player = 0;
        env.current_bid = 4;
        env.current_bidder = Some(0);
        env.must_reveal_own = true;

        let (_, rewards, _) = env.step(REVEAL_PLAYER_BASE);

        // With reward_shaping_coef = 0.1:
        // - All surviving players get survival bonus (0.1)
        // - Bidder (P0) also gets failure penalty (-0.1)
        // - So P0 gets 0.1 - 0.1 = 0.0, others get 0.1
        let non_zero_count = rewards.iter().filter(|&&r| r != 0.0).count();
        assert!(
            non_zero_count > 0,
            "Reward shaping should produce non-zero rewards: {rewards:?}"
        );
    }

    #[test]
    fn test_zero_sum_rewards() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        // Play a complete game
        let mut done = false;
        let mut total_rewards = [0.0f32; MAX_PLAYERS];
        let mut steps = 0;

        while !done && steps < 10000 {
            let mask = env.action_mask().unwrap();
            let valid: Vec<_> = mask
                .iter()
                .enumerate()
                .filter(|(_, &v)| v)
                .map(|(i, _)| i)
                .collect();

            if valid.is_empty() {
                break;
            }

            let action = valid[steps % valid.len()];
            let (_, rewards, d) = env.step(action);
            done = d;
            steps += 1;

            for (i, &r) in rewards.iter().enumerate() {
                total_rewards[i] += r;
            }
        }

        if done && env.reward_shaping_coef == 0.0 {
            let sum: f32 = total_rewards[..env.num_players].iter().sum();
            assert!(
                sum.abs() < 0.01,
                "Terminal rewards should sum to ~0, got {sum}"
            );
        }
    }

    #[test]
    fn test_player_exists_flags_in_observation() {
        let mut env = Skull::new_with_players(3, 0.0, 42);
        env.reset();
        let obs = env.get_observation();

        // Player exists flags are now RELATIVE
        // So for any current player, relative indices 0, 1, 2 should exist (= 1.0)
        // and relative indices 3, 4, 5 should not exist (= 0.0)
        let offset =
            OBS_OWN_HAND + OBS_OWN_STACK + OBS_STACK_SIZES + OBS_COASTERS + OBS_ALIVE_FLAGS;

        assert_eq!(obs[offset], 1.0, "Relative 0 exists (current player)");
        assert_eq!(obs[offset + 1], 1.0, "Relative 1 exists");
        assert_eq!(obs[offset + 2], 1.0, "Relative 2 exists");
        assert_eq!(obs[offset + 3], 0.0, "Relative 3 doesn't exist");
        assert_eq!(obs[offset + 4], 0.0, "Relative 4 doesn't exist");
        assert_eq!(obs[offset + 5], 0.0, "Relative 5 doesn't exist");
    }

    #[test]
    fn test_bid_history_encoding() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        for _ in 0..4 {
            env.step(PLACE_ROSE);
        }

        env.step(BID_BASE + 1 - 1); // Bid 1
        env.step(BID_BASE + 2 - 1); // Bid 2

        assert_eq!(env.bid_history.len(), 2);

        let obs = env.get_observation();
        // Bid history is at the end of observation
        let history_offset = OBSERVATION_DIM - OBS_BID_HISTORY;

        // First entry should have player one-hot and bid value
        let first_entry = &obs[history_offset..history_offset + BID_HISTORY_ENTRY_SIZE];
        let player_sum: f32 = first_entry[..MAX_PLAYERS].iter().sum();
        assert_eq!(player_sum, 1.0, "Player one-hot should sum to 1");
    }

    #[test]
    fn test_phase_one_hot_encoding() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        let obs = env.get_observation();
        let phase_offset = OBS_OWN_HAND
            + OBS_OWN_STACK
            + OBS_STACK_SIZES
            + OBS_COASTERS
            + OBS_ALIVE_FLAGS
            + OBS_PLAYER_EXISTS
            + OBS_SEAT_POSITION;

        // Should be placing phase
        assert_eq!(obs[phase_offset], 1.0, "Placing phase");
        assert_eq!(obs[phase_offset + 1], 0.0, "Not bidding");
        assert_eq!(obs[phase_offset + 2], 0.0, "Not revealing");
    }

    #[test]
    fn test_random_game_completes() {
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);

        for seed in 0..100 {
            let mut env = Skull::new_with_players(4, 0.0, seed);
            env.reset();

            let mut steps = 0;
            let max_steps = 10000;

            while !env.game_over && steps < max_steps {
                let mask = env.action_mask().unwrap();
                let valid: Vec<_> = mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v)
                    .map(|(i, _)| i)
                    .collect();

                assert!(
                    !valid.is_empty(),
                    "No valid actions at step {steps} for seed {seed}"
                );

                let action = valid[rng.gen_range(0..valid.len())];
                env.step(action);
                steps += 1;
            }

            assert!(
                env.game_over,
                "Game should complete within {max_steps} steps for seed {seed}"
            );
        }
    }

    #[test]
    fn test_parse_action_roundtrip() {
        let env = Skull::new(42);

        for action in 0..ACTION_COUNT {
            let desc = env.describe_action(action);
            let parsed = env.parse_action(&desc);
            assert_eq!(
                parsed,
                Ok(action),
                "parse_action(describe_action({action})) should be {action}"
            );
        }
    }

    #[test]
    fn test_render_output_valid() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        let render = env.render();
        assert!(render.is_some());
        let s = render.unwrap();
        assert!(!s.is_empty());
        assert!(s.contains("Skull"));
        assert!(s.contains("Phase"));
    }

    #[test]
    fn test_game_outcome_placements_valid() {
        let mut env = Skull::new_with_players(4, 0.0, 42);
        env.reset();

        // Play until game over
        let mut steps = 0;
        while !env.game_over && steps < 10000 {
            let mask = env.action_mask().unwrap();
            let valid: Vec<_> = mask
                .iter()
                .enumerate()
                .filter(|(_, &v)| v)
                .map(|(i, _)| i)
                .collect();
            if valid.is_empty() {
                break;
            }
            env.step(valid[steps % valid.len()]);
            steps += 1;
        }

        if env.game_over {
            let outcome = env.game_outcome();
            assert!(outcome.is_some());
            let placements = outcome.unwrap().0;

            // Winner should have placement 1
            if let Some(winner) = env.winner {
                assert_eq!(placements[winner], 1);
            }

            // All active players should have placements 1-num_players
            for (p, &placement) in placements.iter().enumerate().take(env.num_players) {
                assert!(
                    placement >= 1 && placement <= env.num_players,
                    "P{p} placement {placement} out of range"
                );
            }
        }
    }

    /// Diagnostic test: Play many games with random agents to check for positional bias
    /// This helps determine if seat bias is inherent to game implementation vs trained behavior
    #[test]
    fn test_positional_returns_with_random_agents() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let num_games = 10_000;
        let mut returns_by_seat = [0.0f64; 4];
        let mut wins_by_seat = [0u32; 4];
        let mut first_player_counts = [0u32; 4];

        for game_seed in 0..num_games {
            let mut env = Skull::new_with_players(4, 0.0, game_seed);
            env.reset();
            let mut rng = StdRng::seed_from_u64(game_seed + 1_000_000);
            let mut total_rewards = [0.0f32; 4];

            // Track who starts the game
            first_player_counts[env.current_player] += 1;

            let mut steps = 0;
            while !env.game_over && steps < 10000 {
                let mask = env.action_mask().unwrap();
                let valid: Vec<usize> = mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v)
                    .map(|(i, _)| i)
                    .collect();

                if valid.is_empty() {
                    break;
                }

                let action = valid[rng.gen_range(0..valid.len())];
                let (_, rewards, _) = env.step(action);

                for (i, &r) in rewards.iter().enumerate().take(4) {
                    total_rewards[i] += r;
                }
                steps += 1;
            }

            // Record results
            for i in 0..4 {
                returns_by_seat[i] += f64::from(total_rewards[i]);
            }
            if let Some(winner) = env.winner {
                wins_by_seat[winner] += 1;
            }
        }

        // Report results
        println!("\n=== Positional Baseline Test ({num_games} games) ===");
        println!("First player distribution: {first_player_counts:?}");
        println!("\nResults by seat:");
        for i in 0..4 {
            let avg_return = returns_by_seat[i] / num_games as f64;
            let win_rate = f64::from(wins_by_seat[i]) / num_games as f64 * 100.0;
            println!(
                "  Seat {}: avg_return = {:.4}, wins = {} ({:.1}%)",
                i, avg_return, wins_by_seat[i], win_rate
            );
        }

        // Check for severe bias (warn but don't fail - this is diagnostic)
        let max_return = returns_by_seat.iter().copied().fold(f64::MIN, f64::max);
        let min_return = returns_by_seat.iter().copied().fold(f64::MAX, f64::min);
        let spread = (max_return - min_return) / num_games as f64;

        println!("\nSpread (max - min avg return): {spread:.4}");
        if spread > 0.1 {
            println!(
                "WARNING: Large positional spread detected - may indicate game implementation bias"
            );
        }
    }

    /// Diagnostic test: Evaluate trained agent from each seat position against random opponents
    /// This tests whether the agent learned position-specific strategies
    ///
    /// Run with: cargo test `test_trained_agent_by_position` --release -- --ignored --nocapture
    #[test]
    #[ignore = "Requires trained checkpoint - run manually"]
    fn test_trained_agent_by_position() {
        use burn::backend::NdArray;
        use burn::prelude::*;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use std::path::Path;

        use crate::eval::load_model_from_checkpoint;
        use crate::utils::{apply_action_mask, sample_categorical};

        type B = NdArray<f32>;
        let device = Default::default();

        // Try to load checkpoint - skip test if not available
        let checkpoint_path = Path::new("runs/skull_009/checkpoints/latest");
        if !checkpoint_path.exists() {
            println!("Checkpoint not found at {checkpoint_path:?}, skipping test");
            return;
        }

        let (model, metadata, normalizer) =
            match load_model_from_checkpoint::<B>(checkpoint_path, &device) {
                Ok(result) => result,
                Err(e) => {
                    println!("Failed to load checkpoint: {e}, skipping test");
                    return;
                }
            };

        println!("\n=== Trained Agent Position Test ===");
        println!("Loaded checkpoint: {checkpoint_path:?}");
        println!(
            "Model: {} hidden, {} layers",
            metadata.hidden_size, metadata.num_hidden
        );

        let num_games = 1000;

        for agent_seat in 0..4 {
            let mut wins = 0u32;
            let mut total_return = 0.0f64;

            for game_seed in 0..num_games {
                let mut env = Skull::new_with_players(4, 0.0, game_seed);
                env.reset();
                let mut rng = StdRng::seed_from_u64(game_seed + 1_000_000);
                let mut agent_return = 0.0f32;

                let mut steps = 0;
                while !env.game_over && steps < 10000 {
                    let current = env.current_player;

                    let action = if current == agent_seat {
                        // Use trained model
                        let mut obs = env.get_observation();
                        if let Some(ref norm) = normalizer {
                            let obs_len = obs.len();
                            norm.normalize_batch(&mut obs, obs_len);
                        }
                        let obs_tensor: Tensor<B, 2> =
                            Tensor::<B, 1>::from_floats(obs.as_slice(), &device)
                                .reshape([1, Skull::OBSERVATION_DIM]);

                        let (logits, _) = model.forward(obs_tensor);

                        // Apply action mask
                        let mask = env.action_mask();
                        let masked_logits = apply_action_mask(logits, mask);

                        // Sample action
                        let action_tensor = sample_categorical(masked_logits, &mut rng, &device);
                        let action: i64 = action_tensor.into_scalar();
                        usize::try_from(action).expect("action should be non-negative")
                    } else {
                        // Random action for other players
                        let mask = env.action_mask().unwrap();
                        let valid: Vec<usize> = mask
                            .iter()
                            .enumerate()
                            .filter(|(_, &v)| v)
                            .map(|(i, _)| i)
                            .collect();
                        if valid.is_empty() {
                            break;
                        }
                        valid[rng.gen_range(0..valid.len())]
                    };

                    let (_, rewards, _) = env.step(action);
                    agent_return += rewards[agent_seat];
                    steps += 1;
                }

                if env.winner == Some(agent_seat) {
                    wins += 1;
                }
                total_return += f64::from(agent_return);
            }

            let win_rate = f64::from(wins) / num_games as f64 * 100.0;
            let avg_return = total_return / num_games as f64;
            println!(
                "Agent at seat {agent_seat}: wins = {wins} ({win_rate:.1}%), avg_return = {avg_return:.4}"
            );
        }

        println!("\nInterpretation:");
        println!(
            "- If wins are similar across seats: bias is in training metrics, not learned policy"
        );
        println!("- If agent wins more from seat 3: agent learned position-specific strategies");
    }

    // ============================================================================
    // PLACEMENT AND TIE HANDLING TESTS
    // ============================================================================

    /// Helper to create a game state for testing placements
    fn setup_test_game(
        num_players: usize,
        wins: &[usize],
        coasters: &[(bool, usize)], // (has_trap, rose_count) per player
        elimination_order: &[usize],
        winner: Option<usize>,
    ) -> Skull {
        let mut env = Skull::new_with_players(num_players, 0.0, 42);
        env.reset();

        for (p, &w) in wins.iter().enumerate().take(num_players) {
            env.wins[p] = w;
        }
        for (p, &(has_trap, roses)) in coasters.iter().enumerate().take(num_players) {
            env.has_trap[p] = has_trap;
            env.rose_count[p] = roses;
        }
        env.elimination_order = elimination_order.to_vec();
        env.winner = winner;
        env.game_over = true;

        env
    }

    #[test]
    fn test_placement_ordering_by_wins() {
        // P0: winner (2 wins), 3 coasters
        // P1: 1 win, 4 coasters (more coasters but fewer wins)
        // P2: 0 wins, 4 coasters
        // P3: 0 wins, 3 coasters
        let env = setup_test_game(
            4,
            &[2, 1, 0, 0],
            &[(true, 2), (true, 3), (true, 3), (true, 2)], // 3, 4, 4, 3 coasters
            &[],
            Some(0),
        );

        let placements = env.compute_placements();
        assert_eq!(placements[0], 1, "Winner should be 1st");
        assert_eq!(placements[1], 2, "1 win should be 2nd");
        assert_eq!(placements[2], 3, "0 wins, 4 coasters should be 3rd");
        assert_eq!(placements[3], 4, "0 wins, 3 coasters should be 4th");
    }

    #[test]
    fn test_placement_ordering_by_coasters() {
        // P0: winner (2 wins), 4 coasters
        // P1: 0 wins, 4 coasters
        // P2: 0 wins, 3 coasters
        // P3: 0 wins, 2 coasters
        let env = setup_test_game(
            4,
            &[2, 0, 0, 0],
            &[(true, 3), (true, 3), (true, 2), (true, 1)], // 4, 4, 3, 2 coasters
            &[],
            Some(0),
        );

        let placements = env.compute_placements();
        assert_eq!(placements[0], 1, "Winner should be 1st");
        assert_eq!(placements[1], 2, "4 coasters should be 2nd");
        assert_eq!(placements[2], 3, "3 coasters should be 3rd");
        assert_eq!(placements[3], 4, "2 coasters should be 4th");
    }

    #[test]
    fn test_placement_ordering_by_elimination() {
        // P0: winner, 4 coasters
        // P1: 0 wins, 0 coasters (eliminated 2nd)
        // P2: 0 wins, 0 coasters (eliminated 1st - worst)
        // P3: 0 wins, 0 coasters (not eliminated - best among non-winners)
        let env = setup_test_game(
            4,
            &[2, 0, 0, 0],
            &[(true, 3), (false, 0), (false, 0), (false, 0)],
            &[2, 1], // P2 first out, P1 second out
            Some(0),
        );

        let placements = env.compute_placements();
        assert_eq!(placements[0], 1, "Winner should be 1st");
        assert_eq!(placements[3], 2, "Not eliminated should be 2nd");
        assert_eq!(placements[1], 3, "Eliminated 2nd should be 3rd");
        assert_eq!(placements[2], 4, "Eliminated 1st should be 4th");
    }

    #[test]
    fn test_true_tie_same_placement() {
        // P0: winner (2 wins), 4 coasters
        // P1: 0 wins, 4 coasters, not eliminated
        // P2: 0 wins, 4 coasters, not eliminated (true tie with P1)
        // P3: 0 wins, 3 coasters
        let env = setup_test_game(
            4,
            &[2, 0, 0, 0],
            &[(true, 3), (true, 3), (true, 3), (true, 2)], // 4, 4, 4, 3 coasters
            &[],
            Some(0),
        );

        let placements = env.compute_placements();
        assert_eq!(placements[0], 1, "Winner should be 1st");
        assert_eq!(placements[1], 2, "P1 should tie for 2nd");
        assert_eq!(placements[2], 2, "P2 should tie for 2nd");
        assert_eq!(placements[3], 4, "P3 should be 4th (skip 3)");
    }

    #[test]
    fn test_true_tie_rewards_split() {
        // Same setup as test_true_tie_same_placement
        let env = setup_test_game(
            4,
            &[2, 0, 0, 0],
            &[(true, 3), (true, 3), (true, 3), (true, 2)],
            &[],
            Some(0),
        );

        let rewards = env.calculate_final_rewards();

        // P0: +1.0 (winner)
        assert!((rewards[0] - 1.0).abs() < 0.001, "Winner should get +1.0");

        // P1 and P2 should have same reward (avg of 2nd and 3rd place)
        // 2nd place: 1 - 2*(2-1)/(4-1) = 1 - 2/3 = 0.333
        // 3rd place: 1 - 2*(3-1)/(4-1) = 1 - 4/3 = -0.333
        // Average: 0.0
        assert!(
            (rewards[1] - rewards[2]).abs() < 0.001,
            "Tied players should have equal rewards"
        );
        assert!(
            rewards[1].abs() < 0.001,
            "P1 should get ~0.0 (avg of +0.33 and -0.33)"
        );

        // P3: -1.0 (4th place)
        assert!(
            (rewards[3] - (-1.0)).abs() < 0.001,
            "P3 should get -1.0 (4th place)"
        );

        // Verify zero-sum
        let sum: f32 = rewards.iter().sum();
        assert!(sum.abs() < 0.001, "Rewards should sum to 0, got {sum}");
    }

    #[test]
    fn test_three_way_tie() {
        // P0: winner
        // P1, P2, P3: all 0 wins, 4 coasters, not eliminated
        let env = setup_test_game(
            4,
            &[2, 0, 0, 0],
            &[(true, 3), (true, 3), (true, 3), (true, 3)], // all 4 coasters
            &[],
            Some(0),
        );

        let placements = env.compute_placements();
        assert_eq!(placements[0], 1, "Winner should be 1st");
        assert_eq!(placements[1], 2, "P1 should tie for 2nd");
        assert_eq!(placements[2], 2, "P2 should tie for 2nd");
        assert_eq!(placements[3], 2, "P3 should tie for 2nd");

        let rewards = env.calculate_final_rewards();
        // P0: +1.0
        assert!((rewards[0] - 1.0).abs() < 0.001, "Winner should get +1.0");

        // P1/P2/P3: avg(+0.33, -0.33, -1.0) = -0.33 each
        let expected_tied_reward = (1.0 / 3.0 - 1.0 / 3.0 - 1.0) / 3.0; // -0.333...
        for (p, reward) in rewards.iter().enumerate().skip(1).take(3) {
            assert!(
                (reward - expected_tied_reward).abs() < 0.001,
                "P{p} should get {expected_tied_reward}, got {reward}"
            );
        }

        // Verify zero-sum
        let sum: f32 = rewards.iter().sum();
        assert!(sum.abs() < 0.001, "Rewards should sum to 0, got {sum}");
    }

    #[test]
    fn test_elimination_beats_non_elimination() {
        // P0: winner
        // P1: 0 wins, 2 coasters, not eliminated
        // P2: 0 wins, 0 coasters, eliminated (even though had coasters before)
        // P3: 0 wins, 4 coasters, not eliminated
        let env = setup_test_game(
            4,
            &[2, 0, 0, 0],
            &[(true, 3), (true, 1), (false, 0), (true, 3)], // 4, 2, 0, 4 coasters
            &[2],                                           // Only P2 eliminated
            Some(0),
        );

        let placements = env.compute_placements();
        assert_eq!(placements[0], 1, "Winner should be 1st");
        assert_eq!(placements[3], 2, "P3 (4 coasters, not elim) should be 2nd");
        assert_eq!(placements[1], 3, "P1 (2 coasters, not elim) should be 3rd");
        assert_eq!(placements[2], 4, "P2 (eliminated) should be 4th");
    }

    #[test]
    fn test_game_outcome_matches_rewards() {
        // Test that game_outcome placements align with reward ordering
        let env = setup_test_game(
            4,
            &[2, 1, 0, 0],
            &[(true, 3), (true, 2), (true, 3), (true, 1)],
            &[3], // P3 eliminated
            Some(0),
        );

        let outcome = env.game_outcome().unwrap();
        let rewards = env.calculate_final_rewards();

        // Lower placement = higher reward
        for i in 0..4 {
            for j in 0..4 {
                if outcome.0[i] < outcome.0[j] {
                    assert!(
                        rewards[i] >= rewards[j],
                        "P{i} (placement {}) should have >= reward than P{j} (placement {})",
                        outcome.0[i],
                        outcome.0[j]
                    );
                } else if outcome.0[i] == outcome.0[j] {
                    assert!(
                        (rewards[i] - rewards[j]).abs() < 0.001,
                        "Tied players should have equal rewards"
                    );
                }
            }
        }
    }

    #[test]
    fn test_two_player_game_placements() {
        let env = setup_test_game(2, &[2, 0], &[(true, 3), (true, 3)], &[], Some(0));

        let placements = env.compute_placements();
        assert_eq!(placements[0], 1, "Winner should be 1st");
        assert_eq!(placements[1], 2, "Loser should be 2nd");

        let rewards = env.calculate_final_rewards();
        assert!((rewards[0] - 1.0).abs() < 0.001, "Winner gets +1.0");
        assert!((rewards[1] - (-1.0)).abs() < 0.001, "Loser gets -1.0");

        let sum: f32 = rewards.iter().sum();
        assert!(sum.abs() < 0.001, "Rewards should sum to 0");
    }

    #[test]
    fn test_six_player_game_placements() {
        // 6 players with various states
        let env = setup_test_game(
            6,
            &[2, 1, 1, 0, 0, 0],
            &[
                (true, 3),  // P0: 4 coasters
                (true, 2),  // P1: 3 coasters
                (true, 3),  // P2: 4 coasters (ties with P1 on wins but more coasters)
                (true, 3),  // P3: 4 coasters
                (true, 2),  // P4: 3 coasters
                (false, 0), // P5: 0 coasters (eliminated)
            ],
            &[5],
            Some(0),
        );

        let placements = env.compute_placements();
        assert_eq!(placements[0], 1, "Winner should be 1st");
        assert_eq!(placements[2], 2, "P2 (1 win, 4 coasters) should be 2nd");
        assert_eq!(placements[1], 3, "P1 (1 win, 3 coasters) should be 3rd");
        // P3, P4 have 0 wins but different coasters
        assert_eq!(placements[3], 4, "P3 (0 wins, 4 coasters) should be 4th");
        assert_eq!(placements[4], 5, "P4 (0 wins, 3 coasters) should be 5th");
        assert_eq!(placements[5], 6, "P5 (eliminated) should be 6th");

        let rewards = env.calculate_final_rewards();
        let sum: f32 = rewards.iter().sum();
        assert!(sum.abs() < 0.001, "Rewards should sum to 0, got {sum}");
    }

    #[test]
    fn test_zero_sum_with_various_ties() {
        // Test multiple tie configurations
        let test_cases = [
            // (wins, coasters, elimination_order, winner)
            (
                vec![2, 0, 0, 0],
                vec![(true, 3), (true, 3), (true, 3), (true, 3)],
                vec![],
                Some(0),
            ), // 3-way tie for 2nd
            (
                vec![2, 1, 1, 0],
                vec![(true, 3), (true, 2), (true, 2), (true, 1)],
                vec![],
                Some(0),
            ), // 2-way tie for 2nd
            (
                vec![2, 0, 0, 0],
                vec![(true, 3), (true, 2), (true, 2), (true, 2)],
                vec![],
                Some(0),
            ), // 3-way tie for 2nd (same coasters)
            (
                vec![2, 0, 0, 0],
                vec![(true, 3), (false, 0), (false, 0), (false, 0)],
                vec![1, 2],
                Some(0),
            ), // eliminations with tie
        ];

        for (i, (wins, coasters, elim, winner)) in test_cases.iter().enumerate() {
            let env = setup_test_game(4, wins, coasters, elim, *winner);
            let rewards = env.calculate_final_rewards();
            let sum: f32 = rewards.iter().sum();
            assert!(
                sum.abs() < 0.001,
                "Test case {i}: Rewards should sum to 0, got {sum}"
            );
        }
    }

    #[test]
    fn test_random_games_zero_sum_and_valid_placements() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let num_games = 1000;

        for game_seed in 0..num_games {
            let mut env = Skull::new_with_players(4, 0.0, game_seed);
            env.reset();
            let mut rng = StdRng::seed_from_u64(game_seed + 1_000_000);
            let mut total_rewards = [0.0f32; 4];

            let mut steps = 0;
            while !env.game_over && steps < 10000 {
                let mask = env.action_mask().unwrap();
                let valid: Vec<usize> = mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v)
                    .map(|(i, _)| i)
                    .collect();

                if valid.is_empty() {
                    break;
                }

                let action = valid[rng.gen_range(0..valid.len())];
                let (_, rewards, _) = env.step(action);

                for (i, &r) in rewards.iter().enumerate().take(4) {
                    total_rewards[i] += r;
                }
                steps += 1;
            }

            if env.game_over {
                // Verify zero-sum
                let sum: f32 = total_rewards.iter().sum();
                assert!(
                    sum.abs() < 0.01,
                    "Game {game_seed}: Rewards should sum to ~0, got {sum}"
                );

                // Verify valid placements
                let outcome = env.game_outcome().unwrap();
                let placements = &outcome.0;

                // All placements should be 1-4
                for (p, &placement) in placements.iter().enumerate() {
                    assert!(
                        (1..=4).contains(&placement),
                        "Game {game_seed}: P{p} has invalid placement {placement}"
                    );
                }

                // Competition ranking: if placement is k, then k-1 players have better placement
                for (p, &placement) in placements.iter().enumerate() {
                    let better_count = placements.iter().filter(|&&pl| pl < placement).count();
                    assert!(
                        better_count < placement,
                        "Game {game_seed}: P{p} placement {placement} inconsistent with competition ranking"
                    );
                }

                // Winner should have placement 1
                if let Some(winner) = env.winner {
                    assert_eq!(
                        placements[winner], 1,
                        "Game {game_seed}: Winner P{winner} should have placement 1"
                    );
                }

                // Verify game_outcome matches reward ordering
                let rewards = env.calculate_final_rewards();
                for i in 0..4 {
                    for j in 0..4 {
                        if placements[i] < placements[j] {
                            assert!(
                                rewards[i] >= rewards[j] - 0.001,
                                "Game {game_seed}: P{i} (pl {}) should have >= reward than P{j} (pl {})",
                                placements[i], placements[j]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_wins_beats_coasters_beats_elimination() {
        // Comprehensive test: P1 has 1 win but only 2 coasters
        // P2 has 0 wins but 4 coasters
        // P3 has 0 wins, 3 coasters but is eliminated
        // Ordering: P0 (winner) > P1 (wins) > P2 (coasters) > P3 (eliminated)
        let env = setup_test_game(
            4,
            &[2, 1, 0, 0],
            &[(true, 3), (true, 1), (true, 3), (false, 0)],
            &[3], // P3 eliminated
            Some(0),
        );

        let placements = env.compute_placements();
        assert_eq!(placements[0], 1, "Winner beats all");
        assert_eq!(
            placements[1], 2,
            "1 win beats 0 wins even with fewer coasters"
        );
        assert_eq!(placements[2], 3, "Not eliminated beats eliminated");
        assert_eq!(placements[3], 4, "Eliminated is last");

        let rewards = env.calculate_final_rewards();
        assert!(rewards[0] > rewards[1], "Winner reward > P1 reward");
        assert!(rewards[1] > rewards[2], "P1 reward > P2 reward");
        assert!(rewards[2] > rewards[3], "P2 reward > P3 reward");

        let sum: f32 = rewards.iter().sum();
        assert!(sum.abs() < 0.001, "Zero-sum: got {sum}");
    }
}
