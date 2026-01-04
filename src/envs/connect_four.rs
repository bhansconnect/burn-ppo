/// Connect Four environment with N-player self-play
///
/// 7x6 board game where players drop pieces to connect four in a row.
/// True self-play: same network plays both sides, one move per step.
use crate::env::{Environment, GameOutcome};
use crate::profile::profile_function;

const COLS: usize = 7;
const ROWS: usize = 6;
const WIN_LENGTH: usize = 4;
const BOARD_SIZE: usize = ROWS * COLS; // 42

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
}

impl ConnectFour {
    /// Render the board as ASCII art
    pub fn render_board(&self) -> String {
        use std::fmt::Write;

        // Inner closure uses ? operator; writing to String is infallible
        let format = |output: &mut String| -> std::fmt::Result {
            writeln!(output, "  0 1 2 3 4 5 6")?;
            writeln!(output, " ---------------")?;

            for row in 0..ROWS {
                write!(output, "|")?;
                for col in 0..COLS {
                    let symbol = match self.board[row][col] {
                        Cell::Empty => '.',
                        Cell::Player1 => 'X',
                        Cell::Player2 => 'O',
                    };
                    write!(output, " {symbol}")?;
                }
                writeln!(output, " |")?;
            }

            writeln!(output, " ---------------")?;

            let turn = match self.current_player {
                Cell::Player1 => "X (Player 0)",
                Cell::Player2 => "O (Player 1)",
                Cell::Empty => "N/A",
            };
            if self.game_over {
                match self.winner {
                    Some(Cell::Player1) => writeln!(output, "Game Over: X (Player 0) wins!")?,
                    Some(Cell::Player2) => writeln!(output, "Game Over: O (Player 1) wins!")?,
                    _ => writeln!(output, "Game Over: Draw!")?,
                }
            } else {
                writeln!(output, "Turn: {turn}")?;
            }
            Ok(())
        };

        let mut output = String::new();
        let _ = format(&mut output);
        output
    }

    /// Get current player index (0 or 1)
    fn current_player_idx(&self) -> usize {
        match self.current_player {
            Cell::Player1 => 0,
            Cell::Player2 => 1,
            Cell::Empty => unreachable!(),
        }
    }

    /// Get the other player
    fn other_player(&self) -> Cell {
        match self.current_player {
            Cell::Player1 => Cell::Player2,
            Cell::Player2 => Cell::Player1,
            Cell::Empty => unreachable!(),
        }
    }

    /// Get list of valid action indices (non-full columns)
    #[cfg(test)]
    pub fn valid_actions(&self) -> Vec<usize> {
        (0..COLS)
            .filter(|&col| self.board[0][col] == Cell::Empty)
            .collect()
    }

    /// Drop a piece in the given column
    /// Returns row placed if successful
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
    #[expect(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "board dimensions are small fixed constants"
    )]
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

    /// Multi-plane observation encoding:
    /// - P0 plane [0..42): 1.0 if Player1 piece, else 0.0
    /// - P1 plane [42..84): 1.0 if Player2 piece, else 0.0
    /// - Turn indicator [84..86): one-hot, [1,0] if P0's turn, [0,1] if P1's turn
    fn write_observation(&self, obs: &mut [f32]) {
        obs.fill(0.0);

        // Board planes
        for row in 0..ROWS {
            for col in 0..COLS {
                let idx = row * COLS + col;
                match self.board[row][col] {
                    Cell::Empty => {}                             // Both planes are 0
                    Cell::Player1 => obs[idx] = 1.0,              // P0 plane [0..42)
                    Cell::Player2 => obs[BOARD_SIZE + idx] = 1.0, // P1 plane [42..84)
                }
            }
        }

        // Turn indicator (one-hot) [84..86)
        let current = self.current_player_idx();
        obs[BOARD_SIZE * 2 + current] = 1.0;
    }
}

impl Environment for ConnectFour {
    /// 42 cells × 2 players + 2 turn indicator = 86
    const OBSERVATION_DIM: usize = BOARD_SIZE * 2 + 2;
    const ACTION_COUNT: usize = COLS; // 7
    const NAME: &'static str = "connect_four";
    const NUM_PLAYERS: usize = 2;

    /// Create new game (seed ignored - game is deterministic)
    fn new(_seed: u64) -> Self {
        Self {
            board: [[Cell::Empty; COLS]; ROWS],
            current_player: Cell::Player1,
            game_over: false,
            winner: None,
        }
    }

    fn reset(&mut self, obs: &mut [f32]) {
        profile_function!();
        self.board = [[Cell::Empty; COLS]; ROWS];
        self.current_player = Cell::Player1;
        self.game_over = false;
        self.winner = None;
        self.write_observation(obs);
    }

    /// Single-move step with N-player rewards
    ///
    /// Reward structure (non-zero-sum, non-negative):
    /// - Win: 1.0
    /// - Draw: 0.5 (1/N for N players)
    /// - Loss: 0.0
    /// - Invalid move: 0.0 for all (action masking should prevent this)
    fn step(&mut self, action: usize, obs: &mut [f32], rewards: &mut [f32]) -> bool {
        profile_function!();

        let current = self.current_player_idx();
        let other = 1 - current;
        rewards.fill(0.0);

        // Invalid action: end episode (action masking should prevent this)
        if action >= COLS || self.board[0][action] != Cell::Empty || self.game_over {
            self.write_observation(obs);
            return true;
        }

        // Execute move
        if let Some(row) = self.drop_piece(action, self.current_player) {
            if self.check_winner(row, action, self.current_player) {
                self.game_over = true;
                self.winner = Some(self.current_player);
                rewards[current] = 1.0; // Winner gets 1.0
                rewards[other] = 0.0; // Loser gets 0.0
                self.write_observation(obs);
                return true;
            }
        }

        // Draw
        if self.is_full() {
            self.game_over = true;
            rewards[0] = 0.5; // Both get 1/N = 0.5
            rewards[1] = 0.5;
            self.write_observation(obs);
            return true;
        }

        // Switch player
        self.current_player = self.other_player();
        self.write_observation(obs);
        false
    }

    fn current_player(&self) -> usize {
        self.current_player_idx()
    }

    fn action_mask(&self, mask: &mut [bool]) {
        for (col, m) in mask.iter_mut().enumerate().take(COLS) {
            *m = self.board[0][col] == Cell::Empty;
        }
    }

    fn render(&self) -> Option<String> {
        Some(self.render_board())
    }

    fn game_outcome(&self) -> Option<GameOutcome> {
        if !self.game_over {
            return None;
        }
        match self.winner {
            Some(Cell::Player1) => Some(GameOutcome::Winner(0)),
            Some(Cell::Player2) => Some(GameOutcome::Winner(1)),
            _ => Some(GameOutcome::Tie),
        }
    }

    fn describe_action(&self, action: usize) -> String {
        format!("Column {}", action + 1) // 1-indexed for humans
    }

    fn parse_action(&self, input: &str) -> Result<usize, String> {
        let input = input.trim();
        if let Ok(col) = input.parse::<usize>() {
            // Accept 1-7 (human-friendly) or 0-6 (0-indexed)
            if (1..=7).contains(&col) {
                return Ok(col - 1); // Convert 1-indexed to 0-indexed
            }
            if col < 7 {
                return Ok(col); // Allow 0-indexed too
            }
        }
        Err("Enter column 1-7".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connect_four_reset() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        env.reset(&mut obs);

        // 86 = 42*2 + 2
        assert_eq!(obs.len(), 86);
        // Board planes should be all zeros (empty)
        assert!(obs[..84].iter().all(|&x| x == 0.0));
        // Turn indicator: P0's turn [1, 0]
        assert_eq!(obs[84], 1.0);
        assert_eq!(obs[85], 0.0);
    }

    #[test]
    fn test_connect_four_step() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut rewards = [0.0; ConnectFour::NUM_PLAYERS];
        env.reset(&mut obs);

        // Make a move in column 3 (P0's turn)
        let done = env.step(3, &mut obs, &mut rewards);

        assert_eq!(obs.len(), 86);
        // Bottom row (row 5), column 3 should have P0's piece in P0 plane
        assert_eq!(obs[5 * COLS + 3], 1.0);
        // P1 plane should be empty at that position
        assert_eq!(obs[BOARD_SIZE + 5 * COLS + 3], 0.0);
        // Now it's P1's turn [0, 1]
        assert_eq!(obs[84], 0.0);
        assert_eq!(obs[85], 1.0);
        // No rewards yet
        assert_eq!(rewards, [0.0, 0.0]);
        assert!(!done);
    }

    #[test]
    fn test_connect_four_invalid_move() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut rewards = [0.0; ConnectFour::NUM_PLAYERS];
        env.reset(&mut obs);

        // Manually fill column 0
        for row in 0..ROWS {
            env.board[row][0] = Cell::Player1;
        }

        // Column is full, move should end game with no rewards
        let done = env.step(0, &mut obs, &mut rewards);
        assert_eq!(rewards, [0.0, 0.0]);
        assert!(done);
    }

    #[test]
    fn test_connect_four_out_of_bounds() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut rewards = [0.0; ConnectFour::NUM_PLAYERS];
        env.reset(&mut obs);

        // Out of bounds column
        let done = env.step(10, &mut obs, &mut rewards);
        assert_eq!(rewards, [0.0, 0.0]);
        assert!(done);
    }

    #[test]
    fn test_connect_four_horizontal_win() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut rewards = [0.0; ConnectFour::NUM_PLAYERS];
        env.reset(&mut obs);

        // Manual setup: Player1 has 3 in a row at bottom
        env.board[5][0] = Cell::Player1;
        env.board[5][1] = Cell::Player1;
        env.board[5][2] = Cell::Player1;

        // Winning move
        let done = env.step(3, &mut obs, &mut rewards);
        assert!(done);
        assert_eq!(rewards[0], 1.0); // P0 wins
        assert_eq!(rewards[1], 0.0); // P1 loses
    }

    #[test]
    fn test_connect_four_vertical_win() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut rewards = [0.0; ConnectFour::NUM_PLAYERS];
        env.reset(&mut obs);

        // Stack 3 in column 0 (P0's pieces)
        env.board[5][0] = Cell::Player1;
        env.board[4][0] = Cell::Player1;
        env.board[3][0] = Cell::Player1;

        // Winning move at row 2
        let done = env.step(0, &mut obs, &mut rewards);
        assert!(done);
        assert_eq!(rewards[0], 1.0); // P0 wins
        assert_eq!(rewards[1], 0.0); // P1 loses
    }

    #[test]
    fn test_connect_four_diagonal_win() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut rewards = [0.0; ConnectFour::NUM_PLAYERS];
        env.reset(&mut obs);

        // Diagonal from (5,0) to (2,3) - need 3 pieces for P0
        env.board[5][0] = Cell::Player1;
        env.board[4][1] = Cell::Player1;
        env.board[3][2] = Cell::Player1;
        // Fill supporting pieces so we can drop at col 3
        env.board[5][3] = Cell::Player2;
        env.board[4][3] = Cell::Player2;
        env.board[3][3] = Cell::Player2;

        // Winning move at (2,3)
        let done = env.step(3, &mut obs, &mut rewards);
        assert!(done);
        assert_eq!(rewards[0], 1.0);
        assert_eq!(rewards[1], 0.0);
    }

    #[test]
    fn test_valid_actions() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        env.reset(&mut obs);

        assert_eq!(env.valid_actions().len(), 7);

        // Fill column 0
        for row in 0..ROWS {
            env.board[row][0] = Cell::Player1;
        }

        assert_eq!(env.valid_actions().len(), 6);
        assert!(!env.valid_actions().contains(&0));
    }

    #[test]
    fn test_action_mask() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut mask = [false; ConnectFour::ACTION_COUNT];
        env.reset(&mut obs);

        env.action_mask(&mut mask);
        assert_eq!(mask.len(), 7);
        assert!(mask.iter().all(|&v| v)); // All valid initially

        // Fill column 3
        for row in 0..ROWS {
            env.board[row][3] = Cell::Player1;
        }

        env.action_mask(&mut mask);
        assert!(mask[0]); // Col 0 valid
        assert!(!mask[3]); // Col 3 full
        assert!(mask[6]); // Col 6 valid
    }

    #[test]
    fn test_current_player_switching() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut rewards = [0.0; ConnectFour::NUM_PLAYERS];
        env.reset(&mut obs);

        assert_eq!(env.current_player(), 0); // P0 starts

        env.step(0, &mut obs, &mut rewards); // P0 moves
        assert_eq!(env.current_player(), 1); // Now P1

        env.step(1, &mut obs, &mut rewards); // P1 moves
        assert_eq!(env.current_player(), 0); // Back to P0
    }

    #[test]
    fn test_draw() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut rewards = [0.0; ConnectFour::NUM_PLAYERS];
        env.reset(&mut obs);

        // Fill board in a way that creates no winner
        // Pattern that avoids 4-in-a-row
        // Row 0-2: P1 P2 P1 P2 P1 P2 P1
        // Row 3-5: P2 P1 P2 P1 P2 P1 P2
        for row in 0..ROWS {
            for col in 0..COLS {
                if row < 3 {
                    env.board[row][col] = if col % 2 == 0 {
                        Cell::Player1
                    } else {
                        Cell::Player2
                    };
                } else {
                    env.board[row][col] = if col % 2 == 0 {
                        Cell::Player2
                    } else {
                        Cell::Player1
                    };
                }
            }
        }

        // Leave one cell empty to trigger draw
        env.board[0][0] = Cell::Empty;

        let done = env.step(0, &mut obs, &mut rewards);
        assert!(done);
        assert_eq!(rewards[0], 0.5);
        assert_eq!(rewards[1], 0.5);
    }

    #[test]
    fn test_multi_plane_encoding() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut rewards = [0.0; ConnectFour::NUM_PLAYERS];
        env.reset(&mut obs);

        // Place P0 piece at (5, 0)
        env.step(0, &mut obs, &mut rewards);
        // Place P1 piece at (5, 1)
        env.step(1, &mut obs, &mut rewards);

        // P0 plane: (5,0) should be 1.0
        assert_eq!(obs[5 * COLS], 1.0);
        // P1 plane: (5,1) should be 1.0
        assert_eq!(obs[BOARD_SIZE + 5 * COLS + 1], 1.0);
        // Turn indicator: P0's turn [1, 0]
        assert_eq!(obs[84], 1.0);
        assert_eq!(obs[85], 0.0);
    }

    #[test]
    fn test_game_outcome_none_when_not_over() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut rewards = [0.0; ConnectFour::NUM_PLAYERS];
        env.reset(&mut obs);

        // Game just started - no outcome yet
        assert_eq!(env.game_outcome(), None);

        // Make a move, still not over
        env.step(0, &mut obs, &mut rewards);
        assert_eq!(env.game_outcome(), None);
    }

    #[test]
    fn test_game_outcome_player0_wins() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut rewards = [0.0; ConnectFour::NUM_PLAYERS];
        env.reset(&mut obs);

        // Set up P0 with 3 in a row at bottom
        env.board[5][0] = Cell::Player1;
        env.board[5][1] = Cell::Player1;
        env.board[5][2] = Cell::Player1;

        // Winning move for P0
        env.step(3, &mut obs, &mut rewards);

        assert!(env.game_over);
        assert_eq!(env.game_outcome(), Some(GameOutcome::Winner(0)));
    }

    #[test]
    fn test_game_outcome_player1_wins() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut rewards = [0.0; ConnectFour::NUM_PLAYERS];
        env.reset(&mut obs);

        // Set up board: P1 has 3 in a row, P0 plays elsewhere
        env.board[5][0] = Cell::Player2;
        env.board[5][1] = Cell::Player2;
        env.board[5][2] = Cell::Player2;
        // P0's pieces elsewhere (so it's valid game state)
        env.board[4][0] = Cell::Player1;
        env.board[4][1] = Cell::Player1;
        env.board[4][2] = Cell::Player1;

        // Switch to P1's turn
        env.current_player = Cell::Player2;

        // P1 wins with column 3
        env.step(3, &mut obs, &mut rewards);

        assert!(env.game_over);
        assert_eq!(env.game_outcome(), Some(GameOutcome::Winner(1)));
    }

    #[test]
    fn test_game_outcome_tie() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut rewards = [0.0; ConnectFour::NUM_PLAYERS];
        env.reset(&mut obs);

        // Fill board without winner (same pattern as test_draw)
        for row in 0..ROWS {
            for col in 0..COLS {
                if row < 3 {
                    env.board[row][col] = if col % 2 == 0 {
                        Cell::Player1
                    } else {
                        Cell::Player2
                    };
                } else {
                    env.board[row][col] = if col % 2 == 0 {
                        Cell::Player2
                    } else {
                        Cell::Player1
                    };
                }
            }
        }

        // Leave one cell empty to trigger draw on final move
        env.board[0][0] = Cell::Empty;
        env.step(0, &mut obs, &mut rewards);

        assert!(env.game_over);
        assert_eq!(env.game_outcome(), Some(GameOutcome::Tie));
    }

    #[test]
    fn test_render_board_empty() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        env.reset(&mut obs);

        let rendered = env.render_board();

        // Check column headers
        assert!(rendered.contains("0 1 2 3 4 5 6"));
        // Check empty board has dots
        assert!(rendered.contains(". . . . . . ."));
        // Check turn indicator
        assert!(rendered.contains("Turn: X (Player 0)"));
    }

    #[test]
    fn test_render_board_with_pieces() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        env.reset(&mut obs);

        // Place some pieces
        env.board[5][0] = Cell::Player1;
        env.board[5][1] = Cell::Player2;

        let rendered = env.render_board();

        // Check pieces are rendered
        assert!(rendered.contains("X O"));
    }

    #[test]
    fn test_render_board_game_over() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        let mut rewards = [0.0; ConnectFour::NUM_PLAYERS];
        env.reset(&mut obs);

        // Set up and trigger P0 win
        env.board[5][0] = Cell::Player1;
        env.board[5][1] = Cell::Player1;
        env.board[5][2] = Cell::Player1;
        env.step(3, &mut obs, &mut rewards);

        let rendered = env.render_board();
        assert!(rendered.contains("Game Over: X (Player 0) wins!"));
    }

    #[test]
    fn test_render_trait_method() {
        let mut env = ConnectFour::new(42);
        let mut obs = [0.0; ConnectFour::OBSERVATION_DIM];
        env.reset(&mut obs);

        // The Environment::render() method should return Some(String)
        let rendered = env.render();
        assert!(rendered.is_some());
        assert!(rendered.unwrap().contains("0 1 2 3 4 5 6"));
    }
}
