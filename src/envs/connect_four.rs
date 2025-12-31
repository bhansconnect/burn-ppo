/// Connect Four environment with self-play
///
/// 7x6 board game where players drop pieces to connect four in a row.
/// Uses self-play: after agent moves, opponent (same/random policy) moves.

use rand::Rng;

use crate::env::Environment;
use crate::profile::profile_function;

const COLS: usize = 7;
const ROWS: usize = 6;
const WIN_LENGTH: usize = 4;

/// Board cell state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Empty,
    Player1,
    Player2,
}

/// Connect Four game state
#[derive(Debug, Clone)]
pub struct ConnectFour {
    board: [[Cell; COLS]; ROWS],
    current_player: Cell,
    game_over: bool,
    winner: Option<Cell>,
    rng: rand::rngs::StdRng,
    /// Opponent policy: random moves for initial training
    random_opponent: bool,
}

impl ConnectFour {
    /// Create new game with seeded RNG
    pub fn new(seed: u64, random_opponent: bool) -> Self {
        use rand::SeedableRng;
        Self {
            board: [[Cell::Empty; COLS]; ROWS],
            current_player: Cell::Player1,
            game_over: false,
            winner: None,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            random_opponent,
        }
    }

    /// Drop a piece in the given column
    /// Returns (row placed, success)
    fn drop_piece(&mut self, col: usize, player: Cell) -> Option<usize> {
        if col >= COLS {
            return None;
        }

        // Find the lowest empty row in this column
        for row in (0..ROWS).rev() {
            if self.board[row][col] == Cell::Empty {
                self.board[row][col] = player;
                return Some(row);
            }
        }
        None // Column is full
    }

    /// Check if a player has won
    fn check_winner(&self, row: usize, col: usize, player: Cell) -> bool {
        let directions: [(i32, i32); 4] = [
            (0, 1),  // Horizontal
            (1, 0),  // Vertical
            (1, 1),  // Diagonal down-right
            (1, -1), // Diagonal down-left
        ];

        for (dr, dc) in directions {
            let mut count = 1;

            // Check forward direction
            for i in 1..WIN_LENGTH as i32 {
                let r = row as i32 + dr * i;
                let c = col as i32 + dc * i;
                if r < 0 || r >= ROWS as i32 || c < 0 || c >= COLS as i32 {
                    break;
                }
                if self.board[r as usize][c as usize] == player {
                    count += 1;
                } else {
                    break;
                }
            }

            // Check backward direction
            for i in 1..WIN_LENGTH as i32 {
                let r = row as i32 - dr * i;
                let c = col as i32 - dc * i;
                if r < 0 || r >= ROWS as i32 || c < 0 || c >= COLS as i32 {
                    break;
                }
                if self.board[r as usize][c as usize] == player {
                    count += 1;
                } else {
                    break;
                }
            }

            if count >= WIN_LENGTH {
                return true;
            }
        }
        false
    }

    /// Check if board is full (draw)
    fn is_full(&self) -> bool {
        for col in 0..COLS {
            if self.board[0][col] == Cell::Empty {
                return false;
            }
        }
        true
    }

    /// Get valid actions (columns that aren't full)
    pub fn valid_actions(&self) -> Vec<usize> {
        (0..COLS)
            .filter(|&col| self.board[0][col] == Cell::Empty)
            .collect()
    }

    /// Convert board to flat observation vector
    /// Uses +1 for Player1, -1 for Player2, 0 for empty
    fn get_observation(&self) -> Vec<f32> {
        let mut obs = Vec::with_capacity(ROWS * COLS);
        for row in 0..ROWS {
            for col in 0..COLS {
                obs.push(match self.board[row][col] {
                    Cell::Empty => 0.0,
                    Cell::Player1 => 1.0,
                    Cell::Player2 => -1.0,
                });
            }
        }
        obs
    }

    /// Make opponent move (random or self-play)
    fn opponent_move(&mut self) {
        if self.game_over {
            return;
        }

        let valid = self.valid_actions();
        if valid.is_empty() {
            self.game_over = true;
            return;
        }

        // Random opponent for now
        let col = if self.random_opponent {
            valid[self.rng.gen_range(0..valid.len())]
        } else {
            // TODO: Use the same model for self-play
            valid[self.rng.gen_range(0..valid.len())]
        };

        if let Some(row) = self.drop_piece(col, Cell::Player2) {
            if self.check_winner(row, col, Cell::Player2) {
                self.game_over = true;
                self.winner = Some(Cell::Player2);
            } else if self.is_full() {
                self.game_over = true;
            }
        }
    }
}

impl Environment for ConnectFour {
    fn reset(&mut self) -> Vec<f32> {
        profile_function!();
        self.board = [[Cell::Empty; COLS]; ROWS];
        self.current_player = Cell::Player1;
        self.game_over = false;
        self.winner = None;
        self.get_observation()
    }

    fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) {
        profile_function!();
        // Invalid action (column full or out of bounds)
        if action >= COLS || self.board[0][action] != Cell::Empty || self.game_over {
            // Penalize invalid moves heavily
            return (self.get_observation(), -1.0, true);
        }

        // Player 1 (agent) moves
        if let Some(row) = self.drop_piece(action, Cell::Player1) {
            if self.check_winner(row, action, Cell::Player1) {
                self.game_over = true;
                self.winner = Some(Cell::Player1);
                return (self.get_observation(), 1.0, true); // Win!
            }
        }

        // Check for draw
        if self.is_full() {
            self.game_over = true;
            return (self.get_observation(), 0.0, true); // Draw
        }

        // Opponent moves
        self.opponent_move();

        // Check if opponent won
        if self.winner == Some(Cell::Player2) {
            return (self.get_observation(), -1.0, true); // Loss
        }

        // Check for draw after opponent
        if self.game_over {
            return (self.get_observation(), 0.0, true); // Draw
        }

        // Game continues
        (self.get_observation(), 0.0, false)
    }

    fn observation_dim(&self) -> usize {
        ROWS * COLS // 42
    }

    fn action_count(&self) -> usize {
        COLS // 7
    }

    fn name(&self) -> &'static str {
        "connect_four"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connect_four_reset() {
        let mut env = ConnectFour::new(42, true);
        let obs = env.reset();

        assert_eq!(obs.len(), 42);
        // All cells should be empty initially
        assert!(obs.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_connect_four_step() {
        let mut env = ConnectFour::new(42, true);
        env.reset();

        // Make a move in column 3
        let (obs, reward, done) = env.step(3);

        assert_eq!(obs.len(), 42);
        // Bottom row, column 3 should have Player1's piece
        assert_eq!(obs[5 * COLS + 3], 1.0);
        // Opponent should have moved too
        let opponent_pieces: f32 = obs.iter().filter(|&&x| x == -1.0).sum();
        assert_eq!(opponent_pieces, -1.0);
        assert!(!done || reward != 0.0);
    }

    #[test]
    fn test_connect_four_invalid_move() {
        let mut env = ConnectFour::new(42, false);
        env.reset();

        // Manually fill column 0
        for row in 0..ROWS {
            env.board[row][0] = Cell::Player1;
        }

        // Column is full, move should be invalid
        let (_, reward, done) = env.step(0);
        assert_eq!(reward, -1.0);
        assert!(done);
    }

    #[test]
    fn test_connect_four_out_of_bounds() {
        let mut env = ConnectFour::new(42, true);
        env.reset();

        // Out of bounds column
        let (_, reward, done) = env.step(10);
        assert_eq!(reward, -1.0);
        assert!(done);
    }

    #[test]
    fn test_connect_four_horizontal_win() {
        let mut env = ConnectFour::new(42, false);
        env.reset();

        // Manual setup: Player1 has 4 in a row at bottom
        env.board[5][0] = Cell::Player1;
        env.board[5][1] = Cell::Player1;
        env.board[5][2] = Cell::Player1;

        // Winning move
        env.drop_piece(3, Cell::Player1);
        assert!(env.check_winner(5, 3, Cell::Player1));
    }

    #[test]
    fn test_connect_four_vertical_win() {
        let mut env = ConnectFour::new(42, false);
        env.reset();

        // Stack 4 in column 0
        env.board[5][0] = Cell::Player1;
        env.board[4][0] = Cell::Player1;
        env.board[3][0] = Cell::Player1;
        env.board[2][0] = Cell::Player1;

        assert!(env.check_winner(2, 0, Cell::Player1));
    }

    #[test]
    fn test_connect_four_diagonal_win() {
        let mut env = ConnectFour::new(42, false);
        env.reset();

        // Diagonal from (5,0) to (2,3)
        env.board[5][0] = Cell::Player1;
        env.board[4][1] = Cell::Player1;
        env.board[3][2] = Cell::Player1;
        env.board[2][3] = Cell::Player1;

        assert!(env.check_winner(2, 3, Cell::Player1));
    }

    #[test]
    fn test_valid_actions() {
        let mut env = ConnectFour::new(42, true);
        env.reset();

        assert_eq!(env.valid_actions().len(), 7);

        // Fill column 0
        for row in 0..ROWS {
            env.board[row][0] = Cell::Player1;
        }

        assert_eq!(env.valid_actions().len(), 6);
        assert!(!env.valid_actions().contains(&0));
    }
}
