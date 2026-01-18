/// Liar's Dice environment with 4-player self-play
///
/// Classic bluffing dice game where players bid on total dice showing a face value.
/// Features hidden information, wild 1s, and elimination-based gameplay.
use crate::env::{Environment, GameOutcome};
use crate::profile::profile_function;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

// Game constants
const NUM_PLAYERS: usize = 4;
const DICE_PER_PLAYER: usize = 2;
const DICE_FACES: usize = 6;
const MAX_TOTAL_DICE: usize = NUM_PLAYERS * DICE_PER_PLAYER; // 8

// Action space: bids (quantity 1-8 × face 1-6) + call liar
const ACTION_COUNT: usize = MAX_TOTAL_DICE * DICE_FACES + 1; // 49
const CALL_LIAR_ACTION: usize = ACTION_COUNT - 1; // 48

// Observation space layout
const OBS_OWN_DICE: usize = DICE_PER_PLAYER * DICE_FACES; // 12: own dice one-hot
const OBS_DICE_COUNTS: usize = NUM_PLAYERS; // 4: dice remaining per player
const OBS_ALIVE_FLAGS: usize = NUM_PLAYERS; // 4: player alive flags
const OBS_CURRENT_PLAYER: usize = NUM_PLAYERS; // 4: current player one-hot
const OBS_CURRENT_BID: usize = MAX_TOTAL_DICE * DICE_FACES; // 48: bid one-hot (qty × face)
const OBS_HAS_BID: usize = 1; // 1: has active bid flag
const OBS_BID_COUNT: usize = 1; // 1: bid count this round
const OBS_LAST_BIDDER: usize = NUM_PLAYERS; // 4: last bidder one-hot

// Bid history: last N bids, each encoded as bidder (4) + quantity (1) + face (6) + valid flag (1) = 12
const BID_HISTORY_SIZE: usize = 16; // Store last 16 bids
const BID_HISTORY_ENTRY_SIZE: usize = NUM_PLAYERS + 1 + DICE_FACES + 1; // 12 floats per entry
const OBS_BID_HISTORY: usize = BID_HISTORY_SIZE * BID_HISTORY_ENTRY_SIZE; // 192

const OBSERVATION_DIM: usize = OBS_OWN_DICE
    + OBS_DICE_COUNTS
    + OBS_ALIVE_FLAGS
    + OBS_CURRENT_PLAYER
    + OBS_CURRENT_BID
    + OBS_HAS_BID
    + OBS_BID_COUNT
    + OBS_LAST_BIDDER
    + OBS_BID_HISTORY; // 78 + 192 = 270

/// Action types in Liar's Dice
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Bid { quantity: usize, face: usize },
    CallLiar,
}

/// Decode action index to Action enum
fn decode_action(action: usize) -> Action {
    if action == CALL_LIAR_ACTION {
        Action::CallLiar
    } else {
        let quantity = action / DICE_FACES + 1; // 1-8
        let face = action % DICE_FACES + 1; // 1-6
        Action::Bid { quantity, face }
    }
}

/// Encode a bid to action index
fn encode_bid(quantity: usize, face: usize) -> usize {
    (quantity - 1) * DICE_FACES + (face - 1)
}

/// Bid history for tracking past bids in the current round
#[derive(Debug, Clone)]
struct BidHistory {
    /// Ring buffer of bids: (bidder, quantity, face)
    history: VecDeque<(usize, usize, usize)>,
}

impl BidHistory {
    fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(BID_HISTORY_SIZE),
        }
    }

    fn push(&mut self, bidder: usize, quantity: usize, face: usize) {
        if self.history.len() >= BID_HISTORY_SIZE {
            self.history.pop_front();
        }
        self.history.push_back((bidder, quantity, face));
    }

    fn clear(&mut self) {
        self.history.clear();
    }

    /// Convert to observation vector (192 floats: 16 bids × 12 floats each)
    fn to_observation(&self) -> Vec<f32> {
        let mut obs = vec![0.0; OBS_BID_HISTORY];

        for (i, &(bidder, quantity, face)) in self.history.iter().enumerate() {
            let base = i * BID_HISTORY_ENTRY_SIZE;

            // Bidder one-hot (4 floats)
            obs[base + bidder] = 1.0;

            // Quantity normalized (1 float): qty / 8
            obs[base + NUM_PLAYERS] = quantity as f32 / MAX_TOTAL_DICE as f32;

            // Face one-hot (6 floats)
            obs[base + NUM_PLAYERS + 1 + (face - 1)] = 1.0;

            // Valid flag (1 float)
            obs[base + BID_HISTORY_ENTRY_SIZE - 1] = 1.0;
        }

        obs
    }
}

/// Liar's Dice game state
#[derive(Debug, Clone)]
pub struct LiarsDice {
    /// Each player's dice values (1-6), indexed as [player][die_index]
    dice: [[u8; DICE_PER_PLAYER]; NUM_PLAYERS],
    /// Dice remaining per player (0 = eliminated)
    dice_count: [usize; NUM_PLAYERS],
    /// Current player index (0-3)
    current_player: usize,
    /// Current bid: (quantity, face) or None if no bid yet
    current_bid: Option<(usize, usize)>,
    /// Who made the current bid
    last_bidder: Option<usize>,
    /// Number of bids in current round (resets each round)
    bid_count: usize,
    /// Bid history for the current round (resets when round ends)
    bid_history: BidHistory,
    /// Track elimination order for `GameOutcome::Placements`
    elimination_order: Vec<usize>,
    /// Is the game over?
    game_over: bool,
    /// Random number generator
    rng: StdRng,
    /// Reward shaping coefficient (per-round survival bonus)
    reward_shaping_coef: f32,
}

impl LiarsDice {
    /// Create with custom reward shaping coefficient
    pub fn new_with_config(seed: u64, reward_shaping_coef: f32) -> Self {
        let mut env = Self {
            dice: [[0; DICE_PER_PLAYER]; NUM_PLAYERS],
            dice_count: [DICE_PER_PLAYER; NUM_PLAYERS],
            current_player: 0,
            current_bid: None,
            last_bidder: None,
            bid_count: 0,
            bid_history: BidHistory::new(),
            elimination_order: Vec::with_capacity(NUM_PLAYERS),
            game_over: false,
            rng: StdRng::seed_from_u64(seed),
            reward_shaping_coef,
        };
        env.roll_all_dice();
        env
    }

    /// Roll all dice for all players
    fn roll_all_dice(&mut self) {
        for player in 0..NUM_PLAYERS {
            for die_idx in 0..self.dice_count[player] {
                self.dice[player][die_idx] = self.rng.gen_range(1_u8..=6);
            }
        }
    }

    /// Count total dice remaining in game
    fn total_dice_remaining(&self) -> usize {
        self.dice_count.iter().sum()
    }

    /// Count how many alive players remain
    fn alive_players(&self) -> usize {
        self.dice_count.iter().filter(|&&c| c > 0).count()
    }

    /// Count dice showing a face value across all players
    /// Wild 1s: For faces 2-6, 1s count as wild. For face 1, only actual 1s count.
    fn count_dice(&self, face: usize) -> usize {
        let mut count = 0;
        for player in 0..NUM_PLAYERS {
            for die_idx in 0..self.dice_count[player] {
                let die_value = self.dice[player][die_idx] as usize;
                if face == 1 {
                    // Bidding on 1s: only count actual 1s
                    if die_value == 1 {
                        count += 1;
                    }
                } else {
                    // Bidding on 2-6: count matches + wild 1s
                    if die_value == face || die_value == 1 {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    /// Check if a bid is valid (higher than current bid)
    fn is_valid_bid(&self, quantity: usize, face: usize) -> bool {
        // Quantity must be at least 1 and not exceed total dice remaining
        if quantity == 0 || quantity > self.total_dice_remaining() {
            return false;
        }
        // Face must be valid (1-6)
        if face == 0 || face > DICE_FACES {
            return false;
        }

        match self.current_bid {
            None => true, // First bid, anything valid
            Some((cur_q, cur_f)) => {
                // Higher quantity OR same quantity with higher face
                quantity > cur_q || (quantity == cur_q && face > cur_f)
            }
        }
    }

    /// Find the next alive player after the given player
    fn next_alive_player(&self, from: usize) -> usize {
        let mut next = (from + 1) % NUM_PLAYERS;
        while self.dice_count[next] == 0 {
            next = (next + 1) % NUM_PLAYERS;
            if next == from {
                // Shouldn't happen if game is not over
                break;
            }
        }
        next
    }

    /// Start a new round after a call resolution
    fn start_new_round(&mut self, loser: usize) {
        // Remove a die from loser
        if self.dice_count[loser] > 0 {
            self.dice_count[loser] -= 1;
        }

        // Check for elimination
        if self.dice_count[loser] == 0 {
            self.elimination_order.push(loser);
        }

        // Check for game end
        if self.alive_players() <= 1 {
            self.game_over = true;
            // Add the winner to elimination order (they're last)
            for p in 0..NUM_PLAYERS {
                if self.dice_count[p] > 0 {
                    self.elimination_order.push(p);
                    break;
                }
            }
            return;
        }

        // Reset for new round
        self.current_bid = None;
        self.last_bidder = None;
        self.bid_count = 0;
        self.bid_history.clear();

        // Loser starts next round (if still alive), else next alive player
        if self.dice_count[loser] > 0 {
            self.current_player = loser;
        } else {
            self.current_player = self.next_alive_player(loser);
        }

        // Roll new dice for all players
        self.roll_all_dice();
    }

    /// Generate observation vector for current state
    fn get_observation(&self) -> Vec<f32> {
        let mut obs = vec![0.0; OBSERVATION_DIM];
        let mut idx = 0;

        // Own dice one-hot (12 floats: 2 dice × 6 faces)
        for die_idx in 0..self.dice_count[self.current_player] {
            let face = self.dice[self.current_player][die_idx] as usize;
            obs[idx + (face - 1)] = 1.0;
            idx += DICE_FACES;
        }
        // Skip remaining slots if player has fewer dice
        idx = OBS_OWN_DICE;

        // Dice counts per player, normalized to 0-1 (4 floats)
        for player in 0..NUM_PLAYERS {
            obs[idx] = self.dice_count[player] as f32 / DICE_PER_PLAYER as f32;
            idx += 1;
        }

        // Player alive flags (4 floats)
        for player in 0..NUM_PLAYERS {
            obs[idx] = if self.dice_count[player] > 0 {
                1.0
            } else {
                0.0
            };
            idx += 1;
        }

        // Current player one-hot (4 floats)
        obs[idx + self.current_player] = 1.0;
        idx += OBS_CURRENT_PLAYER;

        // Current bid one-hot (48 floats)
        if let Some((quantity, face)) = self.current_bid {
            let bid_idx = encode_bid(quantity, face);
            obs[idx + bid_idx] = 1.0;
        }
        idx += OBS_CURRENT_BID;

        // Has active bid flag (1 float)
        obs[idx] = if self.current_bid.is_some() { 1.0 } else { 0.0 };
        idx += OBS_HAS_BID;

        // Bid count this round, normalized (1 float)
        // Max reasonable bids per round is ~20
        obs[idx] = (self.bid_count as f32 / 20.0).min(1.0);
        idx += OBS_BID_COUNT;

        // Last bidder one-hot (4 floats)
        if let Some(bidder) = self.last_bidder {
            obs[idx + bidder] = 1.0;
        }
        idx += OBS_LAST_BIDDER;

        // Bid history (192 floats)
        let history_obs = self.bid_history.to_observation();
        obs[idx..idx + OBS_BID_HISTORY].copy_from_slice(&history_obs);

        obs
    }

    /// Render the game state as ASCII art
    fn render_state(&self) -> String {
        use std::fmt::Write;

        let format = |output: &mut String| -> std::fmt::Result {
            writeln!(output, "=== Liar's Dice ===")?;
            writeln!(output)?;

            // Show each player's status
            for player in 0..NUM_PLAYERS {
                let marker = if player == self.current_player {
                    "→"
                } else {
                    " "
                };
                let status = if self.dice_count[player] == 0 {
                    "OUT".to_string()
                } else {
                    format!("{} dice", self.dice_count[player])
                };

                // Show dice for current player (hidden for others)
                let dice_str = if player == self.current_player {
                    let dice: Vec<String> = (0..self.dice_count[player])
                        .map(|i| format!("[{}]", self.dice[player][i]))
                        .collect();
                    dice.join(" ")
                } else if self.dice_count[player] > 0 {
                    (0..self.dice_count[player])
                        .map(|_| "[?]")
                        .collect::<Vec<_>>()
                        .join(" ")
                } else {
                    String::new()
                };

                writeln!(output, "{marker} Player {player}: {status}  {dice_str}")?;
            }

            writeln!(output)?;

            // Show current bid
            if let Some((quantity, face)) = self.current_bid {
                writeln!(
                    output,
                    "Current bid: {} {}s (by Player {})",
                    quantity,
                    face,
                    self.last_bidder.unwrap_or(0)
                )?;
            } else {
                writeln!(output, "No bid yet - first player to bid")?;
            }

            if self.game_over {
                writeln!(output)?;
                for p in 0..NUM_PLAYERS {
                    if self.dice_count[p] > 0 {
                        writeln!(output, "Game Over: Player {p} wins!")?;
                        break;
                    }
                }
            }

            Ok(())
        };

        let mut output = String::new();
        let _ = format(&mut output);
        output
    }
}

impl Environment for LiarsDice {
    const OBSERVATION_DIM: usize = OBSERVATION_DIM;
    const ACTION_COUNT: usize = ACTION_COUNT;
    const NAME: &'static str = "liars_dice";
    const NUM_PLAYERS: usize = NUM_PLAYERS;
    const EVAL_TEMP: f32 = 1.0; // Stochastic play essential for bluffing

    fn new(seed: u64) -> Self {
        Self::new_with_config(seed, 0.0) // Default: no reward shaping
    }

    fn reset(&mut self) -> Vec<f32> {
        profile_function!();

        self.dice_count = [DICE_PER_PLAYER; NUM_PLAYERS];
        self.current_player = 0;
        self.current_bid = None;
        self.last_bidder = None;
        self.bid_count = 0;
        self.bid_history.clear();
        self.elimination_order.clear();
        self.game_over = false;
        self.roll_all_dice();

        self.get_observation()
    }

    fn step(&mut self, action: usize) -> (Vec<f32>, Vec<f32>, bool) {
        profile_function!();

        let mut rewards = vec![0.0; NUM_PLAYERS];

        // Invalid if game is over or player is eliminated
        if self.game_over || self.dice_count[self.current_player] == 0 {
            return (self.get_observation(), rewards, true);
        }

        match decode_action(action) {
            Action::Bid { quantity, face } => {
                // Check if bid is valid
                if !self.is_valid_bid(quantity, face) {
                    // Invalid bid - end episode with no rewards
                    self.game_over = true;
                    return (self.get_observation(), rewards, true);
                }

                // Record bid in history before updating state
                self.bid_history.push(self.current_player, quantity, face);

                // Execute the bid
                self.current_bid = Some((quantity, face));
                self.last_bidder = Some(self.current_player);
                self.bid_count += 1;

                // Move to next alive player
                self.current_player = self.next_alive_player(self.current_player);

                (self.get_observation(), rewards, false)
            }
            Action::CallLiar => {
                // Can't call if no bid exists
                if self.current_bid.is_none() {
                    self.game_over = true;
                    return (self.get_observation(), rewards, true);
                }

                // Resolve the call - determine if caller was correct
                let (bid_qty, bid_face) = self.current_bid.expect("Call with no bid");
                let actual_count = self.count_dice(bid_face);
                let caller_correct = actual_count < bid_qty;
                let caller = self.current_player;
                let bidder = self.last_bidder.expect("No last bidder");
                let loser = if caller_correct { bidder } else { caller };

                // Start new round (handles elimination and game end)
                self.start_new_round(loser);

                // Placement-based rewards
                // Survival shaping during game
                for (p, reward) in rewards.iter_mut().enumerate() {
                    if self.dice_count[p] > 0 {
                        *reward += self.reward_shaping_coef;
                    }
                }
                // Final placement rewards at game end
                if self.game_over {
                    // [1st, 2nd, 3rd, 4th] = [+1.0, +0.33, -0.33, -1.0]
                    let placement_rewards = [1.0_f32, 0.33, -0.33, -1.0];
                    for (order, &player) in self.elimination_order.iter().enumerate() {
                        let placement = NUM_PLAYERS - order; // order 0->4th, 3->1st
                        rewards[player] = placement_rewards[placement - 1];
                    }
                }

                (self.get_observation(), rewards, self.game_over)
            }
        }
    }

    fn current_player(&self) -> usize {
        self.current_player
    }

    fn action_mask(&self) -> Option<Vec<bool>> {
        let mut mask = vec![false; ACTION_COUNT];

        // Can't act if eliminated or game over
        if self.dice_count[self.current_player] == 0 || self.game_over {
            return Some(mask);
        }

        // CALL_LIAR: only valid if there's a bid to challenge
        mask[CALL_LIAR_ACTION] = self.current_bid.is_some();

        // BID actions: must be higher than current bid AND within dice count
        let max_qty = self.total_dice_remaining();
        for q in 1..=max_qty {
            for f in 1..=DICE_FACES {
                if self.is_valid_bid(q, f) {
                    let action = encode_bid(q, f);
                    mask[action] = true;
                }
            }
        }

        Some(mask)
    }

    fn render(&self) -> Option<String> {
        Some(self.render_state())
    }

    fn game_outcome(&self) -> Option<GameOutcome> {
        if !self.game_over {
            return None;
        }

        // Convert elimination order to placements (1-indexed)
        // elimination_order[0] = first eliminated (4th place)
        // elimination_order[3] = winner (1st place)
        let mut placements = vec![0; NUM_PLAYERS];
        for (order, &player) in self.elimination_order.iter().enumerate() {
            // order 0 = 4th place, order 3 = 1st place
            placements[player] = NUM_PLAYERS - order;
        }

        Some(GameOutcome(placements))
    }

    fn describe_action(&self, action: usize) -> String {
        match decode_action(action) {
            Action::Bid { quantity, face } => format!("Bid: {quantity} {face}s"),
            Action::CallLiar => "Call Liar!".to_string(),
        }
    }

    fn parse_action(&self, input: &str) -> Result<usize, String> {
        let input = input.trim().to_lowercase();

        // Check for "call" or "liar"
        if input == "call" || input == "liar" || input == "l" {
            return Ok(CALL_LIAR_ACTION);
        }

        // Try to parse "N Fs" format (e.g., "3 4s" or "3 fours")
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.len() >= 2 {
            if let Ok(quantity) = parts[0].parse::<usize>() {
                // Try to parse face as number
                let face_str = parts[1].trim_end_matches('s');
                if let Ok(face) = face_str.parse::<usize>() {
                    if (1..=6).contains(&face) && (1..=8).contains(&quantity) {
                        return Ok(encode_bid(quantity, face));
                    }
                }
            }
        }

        Err("Enter 'N Fs' (e.g., '3 4s') or 'call'".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reset() {
        let mut env = LiarsDice::new(42);
        let obs = env.reset();

        assert_eq!(obs.len(), OBSERVATION_DIM);
        assert_eq!(obs.len(), 270); // 78 base + 192 bid history

        // All players should have 2 dice
        for p in 0..NUM_PLAYERS {
            assert_eq!(env.dice_count[p], 2);
        }

        // No bid yet
        assert!(env.current_bid.is_none());
        assert!(!env.game_over);
    }

    #[test]
    fn test_observation_dimension() {
        let env = LiarsDice::new(42);
        let obs = env.get_observation();
        assert_eq!(obs.len(), 270); // 78 base + 192 bid history
    }

    #[test]
    fn test_valid_bid_first() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Any valid bid should work as first bid
        assert!(env.is_valid_bid(1, 1));
        assert!(env.is_valid_bid(8, 6));
        assert!(!env.is_valid_bid(9, 1)); // Too high
        assert!(!env.is_valid_bid(0, 1)); // quantity 0 not allowed
    }

    #[test]
    fn test_valid_bid_raises() {
        let mut env = LiarsDice::new(42);
        env.reset();

        env.current_bid = Some((3, 4)); // "3 fours"

        // Higher quantity always valid
        assert!(env.is_valid_bid(4, 1));
        assert!(env.is_valid_bid(4, 4));
        assert!(env.is_valid_bid(5, 2));

        // Same quantity, higher face valid
        assert!(env.is_valid_bid(3, 5));
        assert!(env.is_valid_bid(3, 6));

        // Same quantity, same or lower face invalid
        assert!(!env.is_valid_bid(3, 4));
        assert!(!env.is_valid_bid(3, 3));

        // Lower quantity invalid
        assert!(!env.is_valid_bid(2, 6));
    }

    #[test]
    fn test_step_bid() {
        let mut env = LiarsDice::new(42);
        env.reset();

        let action = encode_bid(2, 3); // "2 threes"
        let (obs, rewards, done) = env.step(action);

        assert_eq!(obs.len(), OBSERVATION_DIM);
        assert_eq!(rewards, vec![0.0, 0.0, 0.0, 0.0]);
        assert!(!done);
        assert_eq!(env.current_bid, Some((2, 3)));
        assert_eq!(env.current_player, 1); // Moved to next player
    }

    #[test]
    fn test_call_liar_no_bid() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Can't call liar if no bid
        let (_, rewards, done) = env.step(CALL_LIAR_ACTION);
        assert!(done);
        assert_eq!(rewards, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_call_liar_bidder_loses() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Set up known dice: all players have [1, 2]
        for p in 0..NUM_PLAYERS {
            env.dice[p][0] = 1;
            env.dice[p][1] = 2;
        }
        // Total 1s: 4 (one per player)
        // Total 2s: 4 (actual) + 4 (wild 1s) = 8

        // Player 0 bids "5 ones" (only 4 exist) - an overbid
        env.step(encode_bid(5, 1));

        // Now player 1's turn, they call liar
        let (_, rewards, done) = env.step(CALL_LIAR_ACTION);

        // Bid was false (only 4 ones, not 5), so bidder (P0) loses
        assert!(!done); // Game continues
        assert_eq!(env.dice_count[0], 1); // P0 lost a die
        assert_eq!(env.dice_count[1], 2); // P1 still has 2

        // With default reward_shaping_coef = 0.0, all surviving players get 0.0
        // (P0 lost a die but survives with 1 die remaining)
        assert_eq!(rewards, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_call_liar_caller_loses() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Set up known dice: all players have [3, 3]
        for p in 0..NUM_PLAYERS {
            env.dice[p][0] = 3;
            env.dice[p][1] = 3;
        }
        // Total 3s: 8

        // Player 0 bids "4 threes" (8 exist, so valid)
        env.step(encode_bid(4, 3));

        // Player 1 calls liar (incorrectly - bid is true)
        let (_, rewards, done) = env.step(CALL_LIAR_ACTION);

        // Bid was true (8 threes >= 4), so caller (P1) loses
        assert!(!done);
        assert_eq!(env.dice_count[0], 2); // P0 still has 2
        assert_eq!(env.dice_count[1], 1); // P1 lost a die

        // With default reward_shaping_coef = 0.0, all surviving players get 0.0
        // (P1 lost a die but survives with 1 die remaining)
        assert_eq!(rewards, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_wild_ones_counting() {
        let env = LiarsDice::new(42);

        // Modify dice for testing
        let mut test_env = env;
        test_env.dice[0] = [1, 3];
        test_env.dice[1] = [1, 4];
        test_env.dice[2] = [3, 3];
        test_env.dice[3] = [5, 6];
        test_env.dice_count = [2, 2, 2, 2];

        // Counting 3s: actual 3s (3) + wild 1s (2) = 5
        assert_eq!(test_env.count_dice(3), 5);

        // Counting 1s: only actual 1s = 2
        assert_eq!(test_env.count_dice(1), 2);

        // Counting 4s: actual 4s (1) + wild 1s (2) = 3
        assert_eq!(test_env.count_dice(4), 3);
    }

    #[test]
    fn test_bid_on_ones() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Set up: mix of 1s and other numbers
        env.dice[0] = [1, 1];
        env.dice[1] = [2, 3];
        env.dice[2] = [1, 4];
        env.dice[3] = [5, 6];
        // Total actual 1s: 3

        // Bid "4 ones" (overbid)
        env.step(encode_bid(4, 1));

        // Call liar
        let (_, rewards, _) = env.step(CALL_LIAR_ACTION);

        // Bidder (P0) should lose - only 3 ones, not 4
        assert_eq!(rewards[0], 0.0);
        assert_eq!(env.dice_count[0], 1);
    }

    #[test]
    fn test_elimination() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Set player 0 to 1 die
        env.dice_count[0] = 1;

        // Make P0 lose (set up a bad bid and call)
        // dice array is fixed size, but only dice_count[p] are "active"
        env.dice[0] = [2, 0]; // Only first die matters (dice_count=1)
        env.dice[1] = [3, 4];
        env.dice[2] = [5, 6];
        env.dice[3] = [2, 3];
        // No 1s, so bidding on 1s will fail

        env.step(encode_bid(1, 1)); // P0 bids "1 one" (doesn't exist)
        env.step(CALL_LIAR_ACTION); // P1 calls

        // P0 should be eliminated
        assert_eq!(env.dice_count[0], 0);
        assert!(env.elimination_order.contains(&0));
    }

    #[test]
    fn test_game_end() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Set up: only P3 has dice, others eliminated
        env.dice_count = [0, 0, 0, 2];
        env.elimination_order = vec![0, 1, 2];
        env.current_player = 3;
        env.game_over = true;
        env.elimination_order.push(3); // Winner

        let outcome = env.game_outcome();
        assert!(outcome.is_some());

        let placements = &outcome.unwrap().0;
        // P0 was eliminated first (4th place)
        // P3 won (1st place)
        assert_eq!(placements[0], 4);
        assert_eq!(placements[1], 3);
        assert_eq!(placements[2], 2);
        assert_eq!(placements[3], 1);
    }

    #[test]
    fn test_action_mask() {
        let mut env = LiarsDice::new(42);
        env.reset();

        let mask = env.action_mask().unwrap();
        assert_eq!(mask.len(), ACTION_COUNT);

        // No bid yet, so call liar should be invalid
        assert!(!mask[CALL_LIAR_ACTION]);

        // All initial bids should be valid
        for q in 1..=8 {
            for f in 1..=6 {
                let action = encode_bid(q, f);
                assert!(mask[action], "Bid {q} {f}s should be valid");
            }
        }

        // Make a bid
        env.step(encode_bid(3, 4)); // "3 fours"

        let mask = env.action_mask().unwrap();

        // Now call liar should be valid
        assert!(mask[CALL_LIAR_ACTION]);

        // Lower bids should be invalid
        assert!(!mask[encode_bid(2, 5)]);
        assert!(!mask[encode_bid(3, 3)]);
        assert!(!mask[encode_bid(3, 4)]); // Same bid

        // Higher bids should be valid
        assert!(mask[encode_bid(3, 5)]);
        assert!(mask[encode_bid(4, 1)]);
    }

    #[test]
    fn test_player_switching() {
        let mut env = LiarsDice::new(42);
        env.reset();

        assert_eq!(env.current_player(), 0);

        env.step(encode_bid(1, 1));
        assert_eq!(env.current_player(), 1);

        env.step(encode_bid(1, 2));
        assert_eq!(env.current_player(), 2);

        env.step(encode_bid(1, 3));
        assert_eq!(env.current_player(), 3);

        env.step(encode_bid(1, 4));
        assert_eq!(env.current_player(), 0); // Wraps around
    }

    #[test]
    fn test_skip_eliminated_player() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Eliminate player 1
        env.dice_count[1] = 0;

        env.step(encode_bid(1, 1)); // P0 bids
        assert_eq!(env.current_player(), 2); // Skips P1
    }

    #[test]
    fn test_loser_starts_next_round() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Set up so P1 will lose when they call
        env.dice[0] = [3, 3];
        env.dice[1] = [3, 3];
        env.dice[2] = [3, 3];
        env.dice[3] = [3, 3];
        // Total 3s: 8

        env.step(encode_bid(4, 3)); // P0 bids "4 threes"
        env.step(CALL_LIAR_ACTION); // P1 calls (loses - bid was true)

        // P1 lost, so P1 should start next round (if still alive)
        assert_eq!(env.current_player, 1);
    }

    #[test]
    fn test_rewards_distribution() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Set up: P0 makes a losing bid
        env.dice[0] = [2, 2];
        env.dice[1] = [2, 2];
        env.dice[2] = [2, 2];
        env.dice[3] = [2, 2];
        // No 1s

        env.step(encode_bid(1, 1)); // P0 bids "1 one" (none exist)
        let (_, rewards, _) = env.step(CALL_LIAR_ACTION); // P1 calls

        // With default reward_shaping_coef = 0.0, all surviving players get 0.0
        // (P0 lost a die but survives with 1 die remaining)
        assert_eq!(rewards, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_game_outcome_none_when_not_over() {
        let mut env = LiarsDice::new(42);
        env.reset();

        assert_eq!(env.game_outcome(), None);

        env.step(encode_bid(1, 1));
        assert_eq!(env.game_outcome(), None);
    }

    #[test]
    fn test_render() {
        let mut env = LiarsDice::new(42);
        env.reset();

        let rendered = env.render();
        assert!(rendered.is_some());
        let rendered = rendered.unwrap();

        assert!(rendered.contains("Liar's Dice"));
        assert!(rendered.contains("Player 0"));
        assert!(rendered.contains("2 dice"));
    }

    #[test]
    fn test_describe_action() {
        let env = LiarsDice::new(42);

        assert_eq!(env.describe_action(encode_bid(3, 4)), "Bid: 3 4s");
        assert_eq!(env.describe_action(CALL_LIAR_ACTION), "Call Liar!");
    }

    #[test]
    fn test_parse_action() {
        let env = LiarsDice::new(42);

        assert_eq!(env.parse_action("3 4s"), Ok(encode_bid(3, 4)));
        assert_eq!(env.parse_action("call"), Ok(CALL_LIAR_ACTION));
        assert_eq!(env.parse_action("liar"), Ok(CALL_LIAR_ACTION));
        assert_eq!(env.parse_action("l"), Ok(CALL_LIAR_ACTION));
        assert!(env.parse_action("invalid").is_err());
    }

    #[test]
    fn test_encode_decode_action() {
        // Test all bid actions
        for q in 1..=8 {
            for f in 1..=6 {
                let action = encode_bid(q, f);
                let decoded = decode_action(action);
                assert_eq!(
                    decoded,
                    Action::Bid {
                        quantity: q,
                        face: f
                    }
                );
            }
        }

        // Test call liar
        assert_eq!(decode_action(CALL_LIAR_ACTION), Action::CallLiar);
    }

    #[test]
    fn test_observation_content() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Set known dice for player 0
        env.dice[0] = [3, 5];
        env.current_player = 0;

        let obs = env.get_observation();

        // Own dice one-hot encoding (indices 0-11, 2 dice × 6 faces)
        // Die 1 = 3: index (face-1) = 2 should be 1.0
        assert_eq!(obs[2], 1.0, "Die 1 (face 3) should be at index 2");
        // Die 2 = 5: index 6 + (face-1) = 6 + 4 = 10 should be 1.0
        assert_eq!(obs[10], 1.0, "Die 2 (face 5) should be at index 10");

        // Dice counts (indices 12-15), normalized to 0-1
        assert_eq!(obs[12], 1.0, "P0 has 2/2 dice = 1.0");
        assert_eq!(obs[13], 1.0, "P1 has 2/2 dice = 1.0");
        assert_eq!(obs[14], 1.0, "P2 has 2/2 dice = 1.0");
        assert_eq!(obs[15], 1.0, "P3 has 2/2 dice = 1.0");

        // Player alive flags (indices 16-19)
        assert_eq!(obs[16], 1.0, "P0 alive");
        assert_eq!(obs[17], 1.0, "P1 alive");
        assert_eq!(obs[18], 1.0, "P2 alive");
        assert_eq!(obs[19], 1.0, "P3 alive");

        // Current player one-hot (indices 20-23)
        assert_eq!(obs[20], 1.0, "Current player is P0");
        assert_eq!(obs[21], 0.0);
        assert_eq!(obs[22], 0.0);
        assert_eq!(obs[23], 0.0);

        // Has active bid flag (index 72: after 12+4+4+4+48 = 72)
        assert_eq!(obs[72], 0.0, "No bid yet");

        // Now make a bid and verify observation updates
        env.step(encode_bid(2, 4)); // "2 fours"
        let obs = env.get_observation();

        // Current bid one-hot (indices 24-71)
        // Bid (2, 4) encodes to (2-1)*6 + (4-1) = 1*6 + 3 = 9
        assert_eq!(obs[24 + 9], 1.0, "Bid 2 fours at index 33");

        // Has active bid flag (index 72)
        assert_eq!(obs[72], 1.0, "Has active bid");

        // Bid count (index 73) = 1/20 = 0.05
        assert!((obs[73] - 0.05).abs() < 0.01, "Bid count should be ~0.05");

        // Last bidder one-hot (indices 74-77)
        assert_eq!(obs[74], 1.0, "Last bidder is P0");
    }

    #[test]
    fn test_bid_history() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Initial observation should have empty history (all zeros in history section)
        let obs = env.get_observation();
        let history_start = 78; // After base observation
        for (i, &val) in obs.iter().enumerate().skip(history_start) {
            assert_eq!(val, 0.0, "History should be empty initially at index {i}");
        }

        // Make a bid: P0 bids "2 threes"
        env.step(encode_bid(2, 3));

        // Check history in observation
        let obs = env.get_observation();

        // First bid entry at index 78 (history_start)
        // Bidder P0 one-hot: indices 78-81, expect [1, 0, 0, 0]
        assert_eq!(obs[78], 1.0, "First bid by P0");
        assert_eq!(obs[79], 0.0);
        assert_eq!(obs[80], 0.0);
        assert_eq!(obs[81], 0.0);

        // Quantity normalized: index 82, expect 2/8 = 0.25
        assert!((obs[82] - 0.25).abs() < 0.01, "Quantity should be 0.25");

        // Face one-hot: indices 83-88, expect [0, 0, 1, 0, 0, 0] for face 3
        assert_eq!(obs[83], 0.0);
        assert_eq!(obs[84], 0.0);
        assert_eq!(obs[85], 1.0, "Face 3 should be set");
        assert_eq!(obs[86], 0.0);
        assert_eq!(obs[87], 0.0);
        assert_eq!(obs[88], 0.0);

        // Valid flag: index 89
        assert_eq!(obs[89], 1.0, "Valid flag should be 1");

        // Second bid entry should still be empty
        let second_entry_start = 78 + 12;
        for (i, &val) in obs[second_entry_start..(second_entry_start + 12)]
            .iter()
            .enumerate()
        {
            assert_eq!(
                val,
                0.0,
                "Second entry should be empty at index {}",
                second_entry_start + i
            );
        }

        // Make another bid: P1 bids "3 fours"
        env.step(encode_bid(3, 4));

        let obs = env.get_observation();

        // Second bid entry: P1 bid "3 fours"
        assert_eq!(obs[second_entry_start + 1], 1.0, "Second bid by P1");
        assert!(
            (obs[second_entry_start + 4] - 0.375).abs() < 0.01,
            "Quantity 3/8"
        );
        assert_eq!(obs[second_entry_start + 5 + 3], 1.0, "Face 4");
        assert_eq!(obs[second_entry_start + 11], 1.0, "Valid flag");
    }

    #[test]
    fn test_bid_history_resets_on_round_end() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Set up known dice so we can predict call liar outcome
        env.dice[0] = [3, 3];
        env.dice[1] = [3, 3];
        env.dice[2] = [3, 3];
        env.dice[3] = [3, 3];
        // Total 3s: 8

        // Make some bids
        env.step(encode_bid(2, 3)); // P0 bids "2 threes"
        env.step(encode_bid(4, 3)); // P1 bids "4 threes"

        // Verify history has 2 entries
        assert_eq!(env.bid_history.history.len(), 2);

        // P2 calls liar (incorrectly - bid was true with 8 threes)
        env.step(CALL_LIAR_ACTION);

        // After round ends, history should be cleared
        assert_eq!(
            env.bid_history.history.len(),
            0,
            "History should be cleared after round"
        );

        // Observation should have empty history
        let obs = env.get_observation();
        let history_start = 78;
        for (i, &val) in obs.iter().enumerate().skip(history_start) {
            assert_eq!(val, 0.0, "History should be empty after round at index {i}");
        }
    }

    #[test]
    fn test_full_game_playthrough() {
        let mut env = LiarsDice::new(123);
        env.reset();

        let mut steps = 0;
        let max_steps = 1000;

        while !env.game_over && steps < max_steps {
            let mask = env.action_mask().unwrap();
            // Pick first valid action
            let action = mask.iter().position(|&v| v).expect("No valid action");
            env.step(action);
            steps += 1;
        }

        assert!(env.game_over, "Game should be over");
        assert_eq!(env.alive_players(), 1, "One player should remain");
        assert!(env.game_outcome().is_some(), "Should have outcome");

        // Verify placements
        if let Some(outcome) = env.game_outcome() {
            // All players should have a placement (1-4)
            for &p in &outcome.0 {
                assert!((1..=4).contains(&p), "Placement should be 1-4");
            }
            // Should have exactly one of each placement
            let mut sorted = outcome.0.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, vec![1, 2, 3, 4]);
        }

        assert!(steps < max_steps, "Game should end in reasonable steps");
    }

    #[test]
    fn test_max_bid_reached() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Bid the maximum possible: 8 sixes
        env.step(encode_bid(8, 6));

        let mask = env.action_mask().unwrap();

        // Only call liar should be valid
        assert!(mask[CALL_LIAR_ACTION], "Call liar should be valid");

        // No bid should be valid after max bid
        for q in 1..=8 {
            for f in 1..=6 {
                assert!(
                    !mask[encode_bid(q, f)],
                    "Bid {q} {f}s should be invalid after 8 6s"
                );
            }
        }
    }

    #[test]
    fn test_two_player_endgame() {
        // Test placement-based rewards at game end
        let mut env = LiarsDice::new_with_config(42, 0.0);
        env.reset();

        // Eliminate P0 and P1
        env.dice_count = [0, 0, 1, 1];
        env.elimination_order = vec![0, 1];
        env.current_player = 2;
        env.current_bid = None;
        env.dice[2] = [3, 0];
        env.dice[3] = [4, 0];

        // P2 bids
        env.step(encode_bid(1, 3));
        assert_eq!(env.current_player, 3, "Should skip to P3");

        // P3 raises
        env.step(encode_bid(1, 4));
        assert_eq!(env.current_player, 2, "Should go back to P2");

        // P2 calls liar (incorrectly - there is 1 four + 0 wilds = 1)
        let (_, rewards, done) = env.step(CALL_LIAR_ACTION);

        // P2 called incorrectly, P2 loses their last die
        assert!(done, "Game should be over");
        assert_eq!(env.dice_count[2], 0, "P2 eliminated");
        assert_eq!(env.dice_count[3], 1, "P3 still has dice");
        // Placement rewards: [1st, 2nd, 3rd, 4th] = [+1.0, +0.33, -0.33, -1.0]
        // P0=4th(-1.0), P1=3rd(-0.33), P2=2nd(+0.33), P3=1st(+1.0)
        assert_eq!(rewards[3], 1.0, "P3 gets 1st place reward");
        assert!(
            (rewards[2] - 0.33).abs() < 0.001,
            "P2 gets 2nd place: {:?}",
            rewards[2]
        );
        assert!(
            (rewards[1] - (-0.33)).abs() < 0.001,
            "P1 gets 3rd place: {:?}",
            rewards[1]
        );
        assert_eq!(rewards[0], -1.0, "P0 gets 4th place reward");
    }

    #[test]
    fn test_deterministic_seeding() {
        let mut env1 = LiarsDice::new(12345);
        let mut env2 = LiarsDice::new(12345);

        env1.reset();
        env2.reset();

        // Same seed should produce same dice
        assert_eq!(env1.dice, env2.dice, "Same seed should give same dice");
        assert_eq!(
            env1.current_player, env2.current_player,
            "Same starting player"
        );

        // Same actions should produce same results
        env1.step(encode_bid(1, 1));
        env2.step(encode_bid(1, 1));

        env1.step(CALL_LIAR_ACTION);
        env2.step(CALL_LIAR_ACTION);

        // After call, new dice are rolled - should still be deterministic
        assert_eq!(
            env1.dice, env2.dice,
            "Same seed should give same dice after round"
        );
        assert_eq!(env1.dice_count, env2.dice_count);
        assert_eq!(env1.current_player, env2.current_player);
    }

    #[test]
    fn test_invalid_action_index() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Action index out of bounds (> 48) maps to invalid bid
        // decode_action(100) = Bid { quantity: 100/6 + 1 = 17, face: 100%6 + 1 = 5 }
        // quantity 17 > 8 total dice, so invalid
        let (_, rewards, done) = env.step(100);
        assert!(done, "Invalid action should end game");
        assert_eq!(rewards, vec![0.0; 4], "No rewards for invalid action");
    }

    #[test]
    fn test_action_after_game_over() {
        let mut env = LiarsDice::new(42);
        env.reset();
        env.game_over = true;

        let (_, rewards, done) = env.step(encode_bid(1, 1));
        assert!(done, "Should return done=true");
        assert_eq!(rewards, vec![0.0; 4], "No rewards when game already over");
    }

    #[test]
    fn test_reduced_dice_action_mask() {
        let mut env = LiarsDice::new(42);
        env.reset();

        // Reduce total dice to 3 (one player with 3 dice, others eliminated)
        env.dice_count = [3, 0, 0, 0];
        env.dice[0] = [1, 2]; // Only first dice_count elements matter

        let mask = env.action_mask().unwrap();

        // Bids up to quantity 3 should be valid
        assert!(mask[encode_bid(1, 1)], "1 1s valid");
        assert!(mask[encode_bid(3, 6)], "3 6s valid");

        // Bids above quantity 3 should be invalid
        assert!(!mask[encode_bid(4, 1)], "4 1s invalid - exceeds dice count");
        assert!(!mask[encode_bid(8, 6)], "8 6s invalid - exceeds dice count");
    }

    #[test]
    fn test_reward_shaping() {
        // Test with non-zero reward shaping coefficient
        let mut env = LiarsDice::new_with_config(42, 0.05);
        env.reset();

        // Set up known dice: all players have [2, 2]
        for p in 0..NUM_PLAYERS {
            env.dice[p][0] = 2;
            env.dice[p][1] = 2;
        }
        // No 1s, so bidding on 1s will fail

        env.step(encode_bid(1, 1)); // P0 bids "1 one" (doesn't exist)
        let (_, rewards, _) = env.step(CALL_LIAR_ACTION); // P1 calls

        // P0 lost a die but survives (2 -> 1 dice)
        // With reward_shaping_coef = 0.05, all surviving players get +0.05
        assert!((rewards[0] - 0.05).abs() < 0.001, "P0 survives, gets +0.05");
        assert!((rewards[1] - 0.05).abs() < 0.001, "P1 survives, gets +0.05");
        assert!((rewards[2] - 0.05).abs() < 0.001, "P2 survives, gets +0.05");
        assert!((rewards[3] - 0.05).abs() < 0.001, "P3 survives, gets +0.05");
    }

    #[test]
    fn test_mid_game_elimination() {
        // Test placement rewards: mid-game eliminations give no immediate reward
        // (only survival shaping if enabled, final rewards at game end)
        let mut env = LiarsDice::new_with_config(42, 0.0);
        env.reset();

        // Set P0 to 1 die so they'll be eliminated
        env.dice_count[0] = 1;
        env.dice[0] = [2, 0];
        env.dice[1] = [3, 4];
        env.dice[2] = [5, 6];
        env.dice[3] = [2, 3];
        // No 1s

        env.step(encode_bid(1, 1)); // P0 bids "1 one" (doesn't exist)
        let (_, rewards, done) = env.step(CALL_LIAR_ACTION); // P1 calls

        // P0 eliminated
        assert!(!done, "Game not over yet (3 players remain)");
        assert_eq!(env.dice_count[0], 0, "P0 eliminated");
        // With placement rewards, no immediate penalty - rewards come at game end
        // With reward_shaping_coef=0.0, all rewards are 0 during mid-game
        assert_eq!(rewards[0], 0.0, "P0 no immediate penalty");
        assert_eq!(rewards[1], 0.0, "P1 survives with 0 shaping");
        assert_eq!(rewards[2], 0.0, "P2 survives with 0 shaping");
        assert_eq!(rewards[3], 0.0, "P3 survives with 0 shaping");
    }
}
