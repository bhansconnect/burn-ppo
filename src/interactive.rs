//! Interactive web interface for Skull AI - assists a physical game

use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use burn::prelude::*;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::checkpoint::{self, CheckpointManager, CheckpointMetadata};
use crate::config::{Config, InteractiveArgs};
use crate::env::Environment;
use crate::envs::{Skull, SkullCard, SkullDiscardChoice, SkullPhase};
use crate::network::ActorCriticNetwork;
use crate::schedule::Schedule;
use crate::tournament::{
    compute_display_names, enumerate_checkpoints, is_checkpoint_dir, is_run_checkpoints_dir,
    is_run_dir, select_checkpoints_with_priority,
};

// Re-export for use in responses
use SkullCard as Card;
use SkullPhase as Phase;

// Constants from skull.rs
const ACTION_COUNT: usize = 33;
const PLACE_SKULL: usize = 0;
const PLACE_ROSE: usize = 1;
const REVEAL_PLAYER_BASE: usize = 27;

/// Action history entry
#[derive(Clone, Debug, Serialize)]
struct ActionHistoryEntry {
    player: usize,
    action_name: String,
    is_private: bool,               // true for place skull/rose actions
    reveal_outcome: Option<String>, // "Skull" or "Rose" for reveal actions
}

/// Loaded network with metadata and display name
struct NetworkEntry<B: Backend> {
    #[expect(dead_code, reason = "May be useful for debugging/logging")]
    path: PathBuf,
    display_name: String,
    model: ActorCriticNetwork<B>,
    metadata: CheckpointMetadata,
}

// ===== Session Types =====

type SessionId = String;

/// Pending discard state when AI player needs to choose what to discard
#[derive(Clone)]
struct PendingDiscard {
    game_before_step: Skull,
    bidder: usize,
}

/// Per-session state (each browser tab gets its own isolated session)
struct SessionState {
    selected_network: Option<usize>,
    game: Option<Skull>,
    game_history: Vec<Skull>, // Stack of previous game states for undo
    pending_discard: Option<PendingDiscard>, // Pending discard choice
    ai_seat: usize,
    hide_private_info: bool,
    action_history: Vec<ActionHistoryEntry>,
    last_active: Instant,
}

/// Application state shared across handlers
struct AppState<B: Backend> {
    device: B::Device,

    // Networks (pre-loaded from CLI sources, shared read-only)
    networks: Vec<NetworkEntry<B>>,

    // Per-session state (complete isolation per tab)
    sessions: HashMap<SessionId, SessionState>,
}

// ===== Session Helpers =====

/// Extract session ID from request headers
fn extract_session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .map(String::from)
}

/// Get or create a session, returning the session ID
fn get_or_create_session<B: Backend>(
    state: &mut AppState<B>,
    session_id: Option<String>,
) -> SessionId {
    let id = session_id.unwrap_or_else(|| format!("{:016x}", thread_rng().gen::<u64>()));

    state
        .sessions
        .entry(id.clone())
        .or_insert_with(|| SessionState {
            selected_network: Some(0), // Default to first network
            game: None,
            game_history: vec![],
            pending_discard: None,
            ai_seat: 0,
            hide_private_info: false,
            action_history: vec![],
            last_active: Instant::now(),
        });

    // Update last active time
    if let Some(session) = state.sessions.get_mut(&id) {
        session.last_active = Instant::now();
    }

    id
}

// ===== API Request/Response Types =====

#[derive(Serialize)]
struct NetworkInfo {
    index: usize,
    display_name: String,
    step: usize,
    rating: Option<f64>,
}

#[derive(Serialize)]
struct NetworksResponse {
    networks: Vec<NetworkInfo>,
    selected: Option<usize>,
}

#[derive(Deserialize)]
struct SelectNetworkRequest {
    index: usize,
}

#[derive(Deserialize)]
struct NewGameRequest {
    num_players: usize,
    ai_seat: usize,
}

#[derive(Serialize)]
struct PlayerState {
    seat: usize,
    is_ai: bool,
    is_alive: bool,
    coasters: usize,
    wins: usize,
    stack_size: usize,
    revealed: usize,
    passed: bool,
    // Only for AI player
    hand: Option<HandInfo>,
    stack_contents: Option<Vec<String>>,
}

#[derive(Serialize)]
struct HandInfo {
    has_skull: bool,
    roses: usize,
}

#[derive(Serialize)]
struct BiddingState {
    current_bid: usize,
    current_bidder: Option<usize>,
}

#[derive(Serialize)]
struct ValidAction {
    index: usize,
    name: String,
    enabled: bool,
}

#[derive(Serialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "JSON API response struct - bools are appropriate for client consumption"
)]
struct GameStateResponse {
    phase: String,
    current_player: usize,
    ai_seat: usize,
    is_ai_turn: bool,
    num_players: usize,
    players: Vec<PlayerState>,
    bidding: BiddingState,
    game_over: bool,
    winner: Option<usize>,
    valid_actions: Vec<ValidAction>,
    must_reveal_own: bool,
    can_undo: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ActionValue {
    Index(usize),
    Special(String),
}

#[derive(Deserialize)]
struct ExecuteActionRequest {
    action: ActionValue,
    reveal_outcome: Option<String>, // "skull" or "rose" for reveal actions on non-AI players
    temperature: Option<f32>,       // Temperature for place_any sampling (default 1.0)
}

#[derive(Serialize)]
struct DiscardOptions {
    bidder: usize,
    has_skull: bool,
    has_rose: bool,
}

#[derive(Serialize)]
struct ExecuteActionResponse {
    success: bool,
    error: Option<String>,
    reveal_result: Option<String>,
    needs_discard_choice: bool,
    discard_options: Option<DiscardOptions>,
}

#[derive(Deserialize)]
struct AiProbsRequest {
    temperature: f32,
}

#[derive(Serialize)]
struct ActionProb {
    index: usize,
    name: String,
    prob: f32,
    enabled: bool,
}

#[derive(Serialize)]
struct AiProbsResponse {
    actions: Vec<ActionProb>,
    is_ai_turn: bool,
}

#[derive(Deserialize)]
struct AiSampleRequest {
    temperature: f32,
    random: bool,
}

#[derive(Serialize)]
struct AiSampleResponse {
    action: usize,
    action_name: String,
    prob: Option<f32>,
}

#[derive(Deserialize)]
struct SetAiSeatRequest {
    ai_seat: usize,
}

#[derive(Deserialize)]
struct SetHideInfoRequest {
    hide: bool,
}

// ===== Helper Functions =====

/// Convert action index to human-readable name
fn action_name(action: usize) -> String {
    match action {
        0 => "Place Skull".to_string(),
        1 => "Place Rose".to_string(),
        2..=25 => format!("Bid {}", action - 1),
        26 => "Pass".to_string(),
        27..=32 => format!("Reveal P{}", action - 27),
        _ => "Invalid".to_string(),
    }
}

/// Convert action to display name (respects privacy)
fn action_display_name(action: usize, is_private: bool, hide_info: bool) -> String {
    if hide_info && is_private {
        match action {
            0 | 1 => "Place Card".to_string(),
            _ => action_name(action),
        }
    } else {
        action_name(action)
    }
}

/// Check if an action reveals private information
fn is_private_action(action: usize) -> bool {
    action == PLACE_SKULL || action == PLACE_ROSE
}

/// Get observation from AI's perspective
fn get_ai_observation<B: Backend>(state: &AppState<B>, session_id: &str) -> Option<Vec<f32>> {
    let session = state.sessions.get(session_id)?;
    let game = session.game.as_ref()?;

    // Temporarily think from AI's perspective
    let mut game_clone = game.clone();
    game_clone.set_current_player(session.ai_seat);

    // Get observation uses relative indexing based on current_player
    Some(game_clone.observation())
}

// ===== API Handler Functions =====

/// Serve the main HTML page
async fn serve_html() -> Html<&'static str> {
    Html(HTML_CONTENT)
}

/// Get list of available networks
async fn get_networks<B: Backend>(
    headers: HeaderMap,
    State(state): State<Arc<Mutex<AppState<B>>>>,
) -> Json<NetworksResponse> {
    let mut state_guard = state.lock().expect("Failed to lock state");
    let session_id = get_or_create_session(&mut state_guard, extract_session_id(&headers));

    let networks: Vec<NetworkInfo> = state_guard
        .networks
        .iter()
        .enumerate()
        .map(|(i, n)| NetworkInfo {
            index: i,
            display_name: n.display_name.clone(),
            step: n.metadata.step,
            rating: Some(f64::from(n.metadata.avg_return)),
        })
        .collect();

    let selected = state_guard
        .sessions
        .get(&session_id)
        .and_then(|s| s.selected_network);

    Json(NetworksResponse { networks, selected })
}

/// Select a network by index
async fn select_network<B: Backend>(
    headers: HeaderMap,
    State(state): State<Arc<Mutex<AppState<B>>>>,
    Json(req): Json<SelectNetworkRequest>,
) -> Result<Json<NetworksResponse>, StatusCode> {
    let mut state_guard = state.lock().expect("Failed to lock state");
    let session_id = get_or_create_session(&mut state_guard, extract_session_id(&headers));

    if req.index >= state_guard.networks.len() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Update session's selected network
    if let Some(session) = state_guard.sessions.get_mut(&session_id) {
        session.selected_network = Some(req.index);
    }

    let networks: Vec<NetworkInfo> = state_guard
        .networks
        .iter()
        .enumerate()
        .map(|(i, n)| NetworkInfo {
            index: i,
            display_name: n.display_name.clone(),
            step: n.metadata.step,
            rating: Some(f64::from(n.metadata.avg_return)),
        })
        .collect();

    let selected = state_guard
        .sessions
        .get(&session_id)
        .and_then(|s| s.selected_network);

    Ok(Json(NetworksResponse { networks, selected }))
}

/// Initialize a new game
async fn new_game<B: Backend>(
    headers: HeaderMap,
    State(state): State<Arc<Mutex<AppState<B>>>>,
    Json(req): Json<NewGameRequest>,
) -> Result<Json<GameStateResponse>, StatusCode> {
    let mut state_guard = state.lock().expect("Failed to lock state");
    let session_id = get_or_create_session(&mut state_guard, extract_session_id(&headers));

    let num_players = req.num_players.clamp(2, 6);
    let ai_seat = req.ai_seat.min(num_players - 1);

    // Create new game
    let mut game =
        Skull::new_with_players(num_players, Schedule::constant(0.0), thread_rng().gen());
    game.reset();

    // Update session state
    if let Some(session) = state_guard.sessions.get_mut(&session_id) {
        session.game = Some(game);
        session.ai_seat = ai_seat;
        session.action_history.clear();
        session.game_history.clear();
        session.pending_discard = None;
    }

    Ok(build_game_state_response(&state_guard, &session_id))
}

/// Get current game state
async fn get_game_state<B: Backend>(
    headers: HeaderMap,
    State(state): State<Arc<Mutex<AppState<B>>>>,
) -> Json<GameStateResponse> {
    let mut state_guard = state.lock().expect("Failed to lock state");
    let session_id = get_or_create_session(&mut state_guard, extract_session_id(&headers));
    build_game_state_response(&state_guard, &session_id)
}

/// Build game state response from state guard (sync helper)
fn build_game_state_response<B: Backend>(
    state_guard: &AppState<B>,
    session_id: &str,
) -> Json<GameStateResponse> {
    let session = state_guard.sessions.get(session_id);
    let ai_seat = session.map_or(0, |s| s.ai_seat);

    let can_undo = session.is_some_and(|s| !s.game_history.is_empty());

    let Some(game) = session.and_then(|s| s.game.as_ref()) else {
        return Json(GameStateResponse {
            phase: "none".to_string(),
            current_player: 0,
            ai_seat,
            is_ai_turn: false,
            num_players: 0,
            players: vec![],
            bidding: BiddingState {
                current_bid: 0,
                current_bidder: None,
            },
            game_over: false,
            winner: None,
            valid_actions: vec![],
            must_reveal_own: false,
            can_undo,
        });
    };

    let num_players = game.num_players();
    let current_player = game.current_player();
    let is_ai_turn = current_player == ai_seat;

    // Build player states
    let mut players = Vec::new();
    for seat in 0..num_players {
        let is_ai = seat == ai_seat;
        let mut player_state = PlayerState {
            seat,
            is_ai,
            is_alive: game.player_is_alive(seat),
            coasters: game.player_coaster_count(seat),
            wins: game.player_wins(seat),
            stack_size: game.player_stack_size(seat),
            revealed: game.player_revealed(seat),
            passed: game.player_passed(seat),
            hand: None,
            stack_contents: None,
        };

        // Only include private info for AI player
        if is_ai {
            let (has_skull, roses) = game.player_hand_info(seat);
            player_state.hand = Some(HandInfo { has_skull, roses });
            player_state.stack_contents = Some(
                game.player_stack_contents(seat)
                    .iter()
                    .map(|c| match c {
                        Card::Skull => "skull".to_string(),
                        Card::Rose => "rose".to_string(),
                    })
                    .collect(),
            );
        }

        players.push(player_state);
    }

    // Get valid actions
    let action_mask = game.valid_actions();
    let valid_actions: Vec<ValidAction> = (0..ACTION_COUNT)
        .filter(|&i| action_mask[i])
        .map(|i| ValidAction {
            index: i,
            name: action_name(i),
            enabled: true,
        })
        .collect();

    let phase = match game.phase() {
        Phase::Placing => "placing",
        Phase::Bidding => "bidding",
        Phase::Revealing => "revealing",
    };

    Json(GameStateResponse {
        phase: phase.to_string(),
        current_player,
        ai_seat,
        is_ai_turn,
        num_players,
        players,
        bidding: BiddingState {
            current_bid: game.current_bid(),
            current_bidder: game.current_bidder(),
        },
        game_over: game.is_game_over(),
        winner: game.winner(),
        valid_actions,
        must_reveal_own: game.must_reveal_own_stack(),
        can_undo,
    })
}

/// Sample a place action (skull or rose) from the policy among valid options
fn sample_place_action<B: Backend>(
    state_guard: &AppState<B>,
    session_id: &str,
    valid_place_actions: &[usize],
    temperature: f32,
) -> Result<usize, StatusCode> {
    let session = state_guard
        .sessions
        .get(session_id)
        .ok_or(StatusCode::BAD_REQUEST)?;
    let network_idx = session.selected_network.ok_or(StatusCode::BAD_REQUEST)?;
    let model = &state_guard.networks[network_idx].model;

    // Verify game exists (get_ai_observation needs it)
    if session.game.is_none() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Get observation from AI's perspective
    let obs =
        get_ai_observation(state_guard, session_id).ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    // Convert observation to tensor [1, dim]
    let obs_tensor =
        Tensor::<B, 1>::from_floats(obs.as_slice(), &state_guard.device).unsqueeze::<2>();

    // Forward pass to get logits
    let logits = match model {
        ActorCriticNetwork::Mlp(net) => {
            let (action_logits, _values) = net.forward(obs_tensor);
            action_logits
        }
        ActorCriticNetwork::Ctde(net) => net.forward_actor(obs_tensor),
        ActorCriticNetwork::Cnn(net) => {
            let (action_logits, _values) = net.forward(obs_tensor);
            action_logits
        }
    };

    // Extract logits for place actions only
    let logits_data: Vec<f32> = logits
        .to_data()
        .to_vec()
        .expect("Failed to convert logits to vec");

    // Get logits for valid place actions, apply temperature, then softmax
    let valid_logits: Vec<f32> = valid_place_actions
        .iter()
        .map(|&a| logits_data[a] / temperature)
        .collect();

    let max_logit = valid_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = valid_logits
        .iter()
        .map(|&x| (x - max_logit).exp())
        .collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    // Sample from the distribution
    let sample: f32 = thread_rng().gen();
    let mut cumsum = 0.0;
    for (i, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if sample < cumsum {
            return Ok(valid_place_actions[i]);
        }
    }

    // Fallback to last action
    Ok(*valid_place_actions
        .last()
        .expect("valid_place_actions should not be empty"))
}

/// Execute an action
async fn execute_action<B: Backend>(
    headers: HeaderMap,
    State(state): State<Arc<Mutex<AppState<B>>>>,
    Json(req): Json<ExecuteActionRequest>,
) -> Result<Json<ExecuteActionResponse>, StatusCode> {
    let mut state_guard = state.lock().expect("Failed to lock state");
    let session_id = get_or_create_session(&mut state_guard, extract_session_id(&headers));

    // Get session's ai_seat and hide_private_info for later use
    let (ai_seat, hide_private_info) = {
        let session = state_guard
            .sessions
            .get(&session_id)
            .ok_or(StatusCode::BAD_REQUEST)?;
        (session.ai_seat, session.hide_private_info)
    };

    // Resolve action first (before mutable borrow of game)
    // This handles "place_any" which needs to sample from the policy
    let action = match &req.action {
        ActionValue::Index(idx) => *idx,
        ActionValue::Special(s) if s == "place_any" => {
            // Get valid actions from game (immutable access)
            let session = state_guard
                .sessions
                .get(&session_id)
                .ok_or(StatusCode::BAD_REQUEST)?;
            let Some(game) = &session.game else {
                return Ok(Json(ExecuteActionResponse {
                    success: false,
                    error: Some("No game in progress".to_string()),
                    reveal_result: None,
                    needs_discard_choice: false,
                    discard_options: None,
                }));
            };
            let action_mask = game.valid_actions();
            let valid_place_actions: Vec<usize> = (0..=1).filter(|&a| action_mask[a]).collect();

            if valid_place_actions.is_empty() {
                return Ok(Json(ExecuteActionResponse {
                    success: false,
                    error: Some("No valid place actions".to_string()),
                    reveal_result: None,
                    needs_discard_choice: false,
                    discard_options: None,
                }));
            }

            // If only one option, use it directly
            if valid_place_actions.len() == 1 {
                valid_place_actions[0]
            } else {
                // Sample from policy using temperature (default 1.0)
                let temperature = req.temperature.unwrap_or(1.0);
                match sample_place_action(
                    &state_guard,
                    &session_id,
                    &valid_place_actions,
                    temperature,
                ) {
                    Ok(action) => action,
                    Err(_) => {
                        // Fallback: random choice
                        valid_place_actions[thread_rng().gen_range(0..valid_place_actions.len())]
                    }
                }
            }
        }
        ActionValue::Special(s) => {
            return Ok(Json(ExecuteActionResponse {
                success: false,
                error: Some(format!("Unknown special action: {s}")),
                reveal_result: None,
                needs_discard_choice: false,
                discard_options: None,
            }));
        }
    };

    // Now take mutable borrow of session for the rest of the operation
    let session = state_guard
        .sessions
        .get_mut(&session_id)
        .ok_or(StatusCode::BAD_REQUEST)?;

    let Some(game) = &mut session.game else {
        return Ok(Json(ExecuteActionResponse {
            success: false,
            error: Some("No game in progress".to_string()),
            reveal_result: None,
            needs_discard_choice: false,
            discard_options: None,
        }));
    };

    let current_player = game.current_player();

    // Validate action
    let action_mask = game.valid_actions();
    if action >= ACTION_COUNT || !action_mask[action] {
        return Ok(Json(ExecuteActionResponse {
            success: false,
            error: Some(format!("Invalid action: {action}")),
            reveal_result: None,
            needs_discard_choice: false,
            discard_options: None,
        }));
    }

    // Handle reveal actions specially for non-AI players
    if (REVEAL_PLAYER_BASE..REVEAL_PLAYER_BASE + 6).contains(&action) {
        let target_player = action - REVEAL_PLAYER_BASE;

        // If revealing non-AI player, we need to set the card based on user input
        if target_player != ai_seat {
            let outcome = req.reveal_outcome.as_deref().unwrap_or("rose");
            let card = if outcome.to_lowercase() == "skull" {
                Card::Skull
            } else {
                Card::Rose
            };

            // Set the card that will be revealed
            let stack_size = game.player_stack_size(target_player);
            let revealed_count = game.player_revealed(target_player);
            if revealed_count < stack_size {
                let reveal_index = stack_size - 1 - revealed_count;
                game.set_stack_card(target_player, reveal_index, card);
            }
        }
    }

    // Save game state for undo before executing action
    let game_before_step = game.clone();
    let ai_coasters_before = game.player_coaster_count(ai_seat);
    let ai_has_skull_before = game.player_has_skull(ai_seat);
    let ai_has_rose_before = game.player_has_rose(ai_seat);
    session.game_history.push(game_before_step.clone());

    // Execute the action
    let (_, rewards, _done) = game.step(action);

    // Determine reveal outcome for history
    let reveal_outcome = if (REVEAL_PLAYER_BASE..REVEAL_PLAYER_BASE + 6).contains(&action) {
        // Check if a skull was hit (negative reward for bidder indicates skull)
        let bidder = game.current_bidder();
        if let Some(b) = bidder {
            if rewards.get(b).is_some_and(|&r| r < 0.0) {
                Some("Skull".to_string())
            } else {
                Some("Rose".to_string())
            }
        } else {
            req.reveal_outcome.clone()
        }
    } else {
        None
    };

    // Check if AI player hit a skull and needs to choose discard
    let ai_coasters_after = game.player_coaster_count(ai_seat);
    let ai_lost_coaster = ai_coasters_after < ai_coasters_before;
    let needs_discard_choice =
        ai_lost_coaster && !hide_private_info && ai_has_skull_before && ai_has_rose_before;

    if needs_discard_choice {
        // Pop the game state we just pushed (we'll restore it for discard choice)
        session.game_history.pop();

        // Store pending discard state
        session.pending_discard = Some(PendingDiscard {
            game_before_step,
            bidder: ai_seat,
        });

        return Ok(Json(ExecuteActionResponse {
            success: true,
            error: None,
            reveal_result: reveal_outcome,
            needs_discard_choice: true,
            discard_options: Some(DiscardOptions {
                bidder: ai_seat,
                has_skull: ai_has_skull_before,
                has_rose: ai_has_rose_before,
            }),
        }));
    }

    // Add to session's history
    let is_private = is_private_action(action);
    let name = action_display_name(action, is_private, hide_private_info);

    session.action_history.push(ActionHistoryEntry {
        player: current_player,
        action_name: name,
        is_private,
        reveal_outcome: reveal_outcome.clone(),
    });

    Ok(Json(ExecuteActionResponse {
        success: true,
        error: None,
        reveal_result: reveal_outcome,
        needs_discard_choice: false,
        discard_options: None,
    }))
}

/// Compute AI probabilities (sync helper)
fn compute_ai_probs<B: Backend>(
    state_guard: &AppState<B>,
    session_id: &str,
    temperature: f32,
) -> Result<Vec<ActionProb>, StatusCode> {
    let session = state_guard
        .sessions
        .get(session_id)
        .ok_or(StatusCode::BAD_REQUEST)?;
    let network_idx = session.selected_network.ok_or(StatusCode::BAD_REQUEST)?;
    let model = &state_guard.networks[network_idx].model;

    let Some(game) = &session.game else {
        return Err(StatusCode::BAD_REQUEST);
    };

    // Get observation from AI's perspective
    let obs =
        get_ai_observation(state_guard, session_id).ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    // Convert observation to tensor [1, dim]
    let obs_tensor =
        Tensor::<B, 1>::from_floats(obs.as_slice(), &state_guard.device).unsqueeze::<2>();

    // Forward pass to get logits
    let logits = match model {
        ActorCriticNetwork::Mlp(net) => {
            let (action_logits, _values) = net.forward(obs_tensor);
            action_logits
        }
        ActorCriticNetwork::Ctde(net) => net.forward_actor(obs_tensor),
        ActorCriticNetwork::Cnn(net) => {
            let (action_logits, _values) = net.forward(obs_tensor);
            action_logits
        }
    };

    // Apply temperature
    let logits = logits / temperature;

    // Get action mask
    let action_mask = game.valid_actions();

    // Convert logits to vec and apply mask
    #[expect(
        clippy::cast_possible_wrap,
        reason = "ACTION_COUNT is 33, always fits in i32"
    )]
    let mut logits_vec: Vec<f32> = logits
        .reshape([ACTION_COUNT as i32])
        .into_data()
        .to_vec()
        .expect("Failed to convert logits to vec");
    for (i, &valid) in action_mask.iter().enumerate() {
        if !valid {
            logits_vec[i] = f32::NEG_INFINITY;
        }
    }

    // Compute softmax probabilities
    let max_logit = logits_vec
        .iter()
        .filter(|&&x| x.is_finite())
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits_vec.iter().map(|&x| (x - max_logit).exp()).sum();
    let probs: Vec<f32> = logits_vec
        .iter()
        .map(|&x| (x - max_logit).exp() / exp_sum)
        .collect();

    // Build response with only enabled actions
    let actions: Vec<ActionProb> = (0..ACTION_COUNT)
        .filter(|&i| action_mask[i])
        .map(|i| ActionProb {
            index: i,
            name: action_name(i),
            prob: probs[i],
            enabled: true,
        })
        .collect();

    Ok(actions)
}

/// Get AI probabilities for valid actions
async fn ai_probs<B: Backend>(
    headers: HeaderMap,
    State(state): State<Arc<Mutex<AppState<B>>>>,
    Json(req): Json<AiProbsRequest>,
) -> Result<Json<AiProbsResponse>, StatusCode> {
    let mut state_guard = state.lock().expect("Failed to lock state");
    let session_id = get_or_create_session(&mut state_guard, extract_session_id(&headers));

    let actions = compute_ai_probs(&state_guard, &session_id, req.temperature)?;

    // Get AI seat and current player to determine if it's AI's turn
    let is_ai_turn = state_guard.sessions.get(&session_id).is_some_and(|s| {
        s.game
            .as_ref()
            .is_some_and(|g| g.current_player() == s.ai_seat)
    });

    Ok(Json(AiProbsResponse {
        actions,
        is_ai_turn,
    }))
}

/// Sample action from AI policy
async fn ai_sample<B: Backend>(
    headers: HeaderMap,
    State(state): State<Arc<Mutex<AppState<B>>>>,
    Json(req): Json<AiSampleRequest>,
) -> Result<Json<AiSampleResponse>, StatusCode> {
    let mut state_guard = state.lock().expect("Failed to lock state");
    let session_id = get_or_create_session(&mut state_guard, extract_session_id(&headers));

    let session = state_guard
        .sessions
        .get(&session_id)
        .ok_or(StatusCode::BAD_REQUEST)?;
    let Some(game) = &session.game else {
        return Err(StatusCode::BAD_REQUEST);
    };

    let action_mask = game.valid_actions();
    let valid_actions: Vec<usize> = (0..ACTION_COUNT).filter(|&i| action_mask[i]).collect();

    if valid_actions.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    if req.random {
        // Random uniform from valid actions
        let mut rng = thread_rng();
        let idx = rng.gen_range(0..valid_actions.len());
        let action = valid_actions[idx];

        return Ok(Json(AiSampleResponse {
            action,
            action_name: action_name(action),
            prob: None,
        }));
    }

    // Get probabilities using helper function
    let probs = compute_ai_probs(&state_guard, &session_id, req.temperature)?;

    if probs.is_empty() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Sample from distribution
    let mut rng = thread_rng();
    let rand_val: f32 = rng.gen();
    let mut cumsum = 0.0;
    let mut sampled_action = probs[0].index;
    let mut sampled_prob = probs[0].prob;

    for ap in &probs {
        cumsum += ap.prob;
        if rand_val < cumsum {
            sampled_action = ap.index;
            sampled_prob = ap.prob;
            break;
        }
    }

    Ok(Json(AiSampleResponse {
        action: sampled_action,
        action_name: action_name(sampled_action),
        prob: Some(sampled_prob),
    }))
}

/// Set AI seat
async fn set_ai_seat<B: Backend>(
    headers: HeaderMap,
    State(state): State<Arc<Mutex<AppState<B>>>>,
    Json(req): Json<SetAiSeatRequest>,
) -> Json<()> {
    let mut state_guard = state.lock().expect("Failed to lock state");
    let session_id = get_or_create_session(&mut state_guard, extract_session_id(&headers));

    if let Some(session) = state_guard.sessions.get_mut(&session_id) {
        let ai_seat = if let Some(game) = &session.game {
            req.ai_seat.min(game.num_players() - 1)
        } else {
            req.ai_seat
        };
        session.ai_seat = ai_seat;
    }
    Json(())
}

/// Set hide private info
async fn set_hide_info<B: Backend>(
    headers: HeaderMap,
    State(state): State<Arc<Mutex<AppState<B>>>>,
    Json(req): Json<SetHideInfoRequest>,
) -> Json<()> {
    let mut state_guard = state.lock().expect("Failed to lock state");
    let session_id = get_or_create_session(&mut state_guard, extract_session_id(&headers));

    if let Some(session) = state_guard.sessions.get_mut(&session_id) {
        session.hide_private_info = req.hide;
    }
    Json(())
}

/// Get action history
async fn get_history<B: Backend>(
    headers: HeaderMap,
    State(state): State<Arc<Mutex<AppState<B>>>>,
) -> Json<Vec<ActionHistoryEntry>> {
    let mut state_guard = state.lock().expect("Failed to lock state");
    let session_id = get_or_create_session(&mut state_guard, extract_session_id(&headers));

    let Some(session) = state_guard.sessions.get(&session_id) else {
        return Json(vec![]);
    };

    // Apply privacy filtering
    let history: Vec<ActionHistoryEntry> = session
        .action_history
        .iter()
        .map(|entry| {
            if session.hide_private_info && entry.is_private {
                ActionHistoryEntry {
                    player: entry.player,
                    action_name: "Place Card".to_string(),
                    is_private: true,
                    reveal_outcome: entry.reveal_outcome.clone(),
                }
            } else {
                entry.clone()
            }
        })
        .collect();

    Json(history)
}

/// Clear action history
async fn clear_history<B: Backend>(
    headers: HeaderMap,
    State(state): State<Arc<Mutex<AppState<B>>>>,
) -> Json<()> {
    let mut state_guard = state.lock().expect("Failed to lock state");
    let session_id = get_or_create_session(&mut state_guard, extract_session_id(&headers));

    if let Some(session) = state_guard.sessions.get_mut(&session_id) {
        session.action_history.clear();
    }
    Json(())
}

#[derive(Serialize)]
struct UndoResponse {
    success: bool,
    error: Option<String>,
    can_undo: bool,
}

/// Undo the last action
async fn undo<B: Backend>(
    headers: HeaderMap,
    State(state): State<Arc<Mutex<AppState<B>>>>,
) -> Json<UndoResponse> {
    let mut state_guard = state.lock().expect("Failed to lock state");
    let session_id = get_or_create_session(&mut state_guard, extract_session_id(&headers));

    let Some(session) = state_guard.sessions.get_mut(&session_id) else {
        return Json(UndoResponse {
            success: false,
            error: Some("No session found".to_string()),
            can_undo: false,
        });
    };

    if session.game_history.is_empty() {
        return Json(UndoResponse {
            success: false,
            error: Some("No actions to undo".to_string()),
            can_undo: false,
        });
    }

    // Pop the previous game state
    session.game = session.game_history.pop();

    // Also pop the action history entry
    session.action_history.pop();

    // Clear pending discard if any
    session.pending_discard = None;

    Json(UndoResponse {
        success: true,
        error: None,
        can_undo: !session.game_history.is_empty(),
    })
}

#[derive(Deserialize)]
struct DiscardChoiceRequest {
    choice: String, // "skull", "rose", or "random"
}

/// Execute the discard choice after AI hits a skull
async fn execute_discard<B: Backend>(
    headers: HeaderMap,
    State(state): State<Arc<Mutex<AppState<B>>>>,
    Json(req): Json<DiscardChoiceRequest>,
) -> Result<Json<ExecuteActionResponse>, StatusCode> {
    let mut state_guard = state.lock().expect("Failed to lock state");
    let session_id = get_or_create_session(&mut state_guard, extract_session_id(&headers));

    let session = state_guard
        .sessions
        .get_mut(&session_id)
        .ok_or(StatusCode::BAD_REQUEST)?;

    // Check if there's a pending discard
    let Some(pending) = session.pending_discard.take() else {
        return Ok(Json(ExecuteActionResponse {
            success: false,
            error: Some("No pending discard".to_string()),
            reveal_result: None,
            needs_discard_choice: false,
            discard_options: None,
        }));
    };

    // Get pre-step state info
    let game_before_step = pending.game_before_step.clone();
    let bidder = pending.bidder;
    let current_player = game_before_step.current_player();

    // Parse the discard choice
    let discard_choice = match req.choice.to_lowercase().as_str() {
        "skull" => SkullDiscardChoice::Skull,
        "rose" => SkullDiscardChoice::Rose,
        _ => SkullDiscardChoice::Random,
    };

    // Save pre-step state to history for undo
    session.game_history.push(game_before_step.clone());

    // For random choice, just execute step normally
    if discard_choice == SkullDiscardChoice::Random {
        let mut game = game_before_step;
        let action_mask = game.valid_actions();
        let reveal_action = (REVEAL_PLAYER_BASE..REVEAL_PLAYER_BASE + 6)
            .find(|&a| action_mask[a])
            .unwrap_or(REVEAL_PLAYER_BASE);

        game.step(reveal_action);
        session.game = Some(game);

        session.action_history.push(ActionHistoryEntry {
            player: current_player,
            action_name: format!("Reveal P{}", reveal_action - REVEAL_PLAYER_BASE),
            is_private: false,
            reveal_outcome: Some("Skull".to_string()),
        });
    } else {
        // For specific choice: execute step, then adjust if needed
        let mut game = game_before_step.clone();
        let action_mask = game.valid_actions();
        let reveal_action = (REVEAL_PLAYER_BASE..REVEAL_PLAYER_BASE + 6)
            .find(|&a| action_mask[a])
            .unwrap_or(REVEAL_PLAYER_BASE);

        let coasters_before = game.player_coaster_count(bidder);
        let has_skull_before = game.player_has_skull(bidder);

        game.step(reveal_action);

        let coasters_after = game.player_coaster_count(bidder);
        let has_skull_after = game.player_has_skull(bidder);
        let lost_coaster = coasters_after < coasters_before;
        let random_lost_skull = has_skull_before && !has_skull_after;

        // Check if we need to adjust
        let need_adjustment = lost_coaster
            && ((discard_choice == SkullDiscardChoice::Skull && !random_lost_skull)
                || (discard_choice == SkullDiscardChoice::Rose && random_lost_skull));

        if need_adjustment {
            // Restore pre-step state and manually apply discard
            game = game_before_step;

            // Apply the specified discard
            game.lose_coaster_specified(bidder, discard_choice);

            // Start new round with current player
            game.interactive_start_new_round(current_player);
        }

        session.game = Some(game);

        session.action_history.push(ActionHistoryEntry {
            player: current_player,
            action_name: format!("Reveal P{}", reveal_action - REVEAL_PLAYER_BASE),
            is_private: false,
            reveal_outcome: Some("Skull".to_string()),
        });
    }

    Ok(Json(ExecuteActionResponse {
        success: true,
        error: None,
        reveal_result: Some("Skull".to_string()),
        needs_discard_choice: false,
        discard_options: None,
    }))
}

// ===== Server Entry Point =====

/// Discover checkpoints from a source path (run dir, checkpoints dir, or single checkpoint)
fn discover_checkpoints(source: &std::path::Path, limit_per_run: usize) -> Result<Vec<PathBuf>> {
    let source = source
        .canonicalize()
        .with_context(|| format!("Failed to resolve path: {}", source.display()))?;

    if is_checkpoint_dir(&source) {
        // Single checkpoint
        Ok(vec![source])
    } else if is_run_dir(&source) {
        // Run directory - use checkpoints subdir
        let checkpoints_dir = source.join("checkpoints");
        let all_checkpoints = enumerate_checkpoints(&checkpoints_dir)?;
        Ok(select_checkpoints_with_priority(
            &checkpoints_dir,
            &all_checkpoints,
            limit_per_run,
        ))
    } else if is_run_checkpoints_dir(&source) {
        // Checkpoints directory directly
        let all_checkpoints = enumerate_checkpoints(&source)?;
        Ok(select_checkpoints_with_priority(
            &source,
            &all_checkpoints,
            limit_per_run,
        ))
    } else {
        anyhow::bail!("Invalid source: {}", source.display());
    }
}

/// Run the interactive web server
pub async fn run_interactive<B: Backend>(args: InteractiveArgs, device: B::Device) -> Result<()> {
    // Discover and load networks from CLI sources
    println!("\nDiscovering networks...");
    let mut checkpoint_paths: Vec<PathBuf> = Vec::new();
    for source in &args.sources {
        let discovered = discover_checkpoints(source, args.limit_per_run)?;
        checkpoint_paths.extend(discovered);
    }

    if checkpoint_paths.is_empty() {
        anyhow::bail!("No checkpoints found in provided sources");
    }

    // Compute display names
    let display_names = compute_display_names(&checkpoint_paths);

    // Load all networks
    println!("Loading {} network(s)...", checkpoint_paths.len());
    let mut networks: Vec<NetworkEntry<B>> = Vec::new();
    for (path, display_name) in checkpoint_paths.into_iter().zip(display_names) {
        let metadata = checkpoint::load_metadata(&path)
            .with_context(|| format!("Failed to load metadata from {}", path.display()))?;

        let config = Config {
            env: metadata.env_name.clone(),
            hidden_size: metadata.hidden_size,
            num_hidden: metadata.num_hidden,
            activation: metadata.activation.clone(),
            ..Config::default()
        };

        let (model, loaded_metadata) = CheckpointManager::load::<B>(&path, &config, &device)
            .with_context(|| format!("Failed to load model from {}", path.display()))?;

        println!("  Loaded: {display_name}");
        networks.push(NetworkEntry {
            path,
            display_name,
            model,
            metadata: loaded_metadata,
        });
    }

    // Sort networks by display name
    networks.sort_by(|a, b| a.display_name.cmp(&b.display_name));

    let state = Arc::new(Mutex::new(AppState {
        device,
        networks,
        sessions: HashMap::new(),
    }));

    // Build router
    let app = Router::new()
        .route("/", get(serve_html))
        .route("/api/networks", get(get_networks::<B>))
        .route("/api/select_network", post(select_network::<B>))
        .route("/api/new_game", post(new_game::<B>))
        .route("/api/game_state", get(get_game_state::<B>))
        .route("/api/execute_action", post(execute_action::<B>))
        .route("/api/ai_probs", post(ai_probs::<B>))
        .route("/api/ai_sample", post(ai_sample::<B>))
        .route("/api/set_ai_seat", post(set_ai_seat::<B>))
        .route("/api/set_hide_info", post(set_hide_info::<B>))
        .route("/api/history", get(get_history::<B>))
        .route("/api/clear_history", post(clear_history::<B>))
        .route("/api/undo", post(undo::<B>))
        .route("/api/execute_discard", post(execute_discard::<B>))
        .with_state(state);

    // Start server
    let addr = format!("127.0.0.1:{}", args.port);
    println!("\nSkull AI Assistant");
    println!("==================");
    println!("Server running at: http://{addr}");
    println!("Press Ctrl+C to stop\n");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ===== Embedded HTML Content =====

const HTML_CONTENT: &str = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skull AI Assistant</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Monaco', 'Courier New', monospace;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a1a;
            color: #e0e0e0;
            font-size: 14px;
        }

        h1 { color: #4CAF50; margin-bottom: 5px; font-size: 24px; }
        h2 { color: #81C784; margin: 0 0 10px 0; font-size: 16px; border-bottom: 1px solid #333; padding-bottom: 5px; }
        h3 { color: #A5D6A7; margin: 10px 0 5px 0; font-size: 14px; }

        .section {
            margin-bottom: 20px;
            padding: 15px;
            border: 1px solid #333;
            background: #2a2a2a;
            border-radius: 4px;
        }

        button {
            padding: 6px 12px;
            background: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            border-radius: 3px;
            font-family: inherit;
            font-size: 13px;
            margin: 2px;
        }

        button:hover { background: #45a049; }
        button:disabled { background: #555; cursor: not-allowed; }
        button.active { background: #FF9800; }
        button.danger { background: #f44336; }
        button.danger:hover { background: #d32f2f; }

        .setup-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 5px 0; }
        .setup-row label { min-width: 80px; }
        .setup-row select, .setup-row input {
            padding: 5px;
            background: #1a1a1a;
            border: 1px solid #444;
            color: #e0e0e0;
            font-family: inherit;
        }

        #game-state {
            font-family: 'Monaco', 'Courier New', monospace;
            white-space: pre;
            background: #1a1a1a;
            padding: 10px;
            border: 1px solid #444;
            border-radius: 4px;
            overflow-x: auto;
        }

        .turn-indicator { color: #FFD700; }
        .ai-indicator { color: #4CAF50; }
        .dead-indicator { color: #666; }

        .action-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin: 5px 0;
        }

        .action-btn { min-width: 80px; }
        .action-btn.place { background: #2196F3; }
        .action-btn.place:hover { background: #1976D2; }
        .action-btn.bid { background: #9C27B0; }
        .action-btn.bid:hover { background: #7B1FA2; }
        .action-btn.reveal { background: #FF5722; }
        .action-btn.reveal:hover { background: #E64A19; }

        .prob-display {
            font-size: 11px;
            color: #aaa;
        }

        .ai-probs {
            background: #1a1a1a;
            padding: 10px;
            border: 1px solid #444;
            border-radius: 4px;
            margin: 10px 0;
        }

        .prob-bar {
            display: inline-block;
            height: 20px;
            background: linear-gradient(90deg, #4CAF50, #81C784);
            margin-right: 5px;
            vertical-align: middle;
        }

        #history {
            max-height: 200px;
            overflow-y: auto;
            background: #1a1a1a;
            border: 1px solid #444;
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
        }

        #history div { padding: 2px 0; }

        .private-section {
            transition: filter 0.3s;
        }
        .private-section.hidden {
            filter: blur(8px);
            user-select: none;
        }

        .reveal-prompt, .discard-prompt {
            display: none;
            background: #333;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .reveal-prompt.active, .discard-prompt.active { display: block; }
        .discard-prompt { background: #442; }

        .controls-row {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
            margin: 10px 0;
        }

        .toggle-group {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        #temp-slider { width: 150px; }
    </style>
</head>
<body>
    <h1>Skull AI Assistant</h1>
    <p style="color: #888; margin-top: 0;">Control an AI player in your physical Skull game</p>

    <!-- Setup Section -->
    <div class="section">
        <h2>Setup</h2>
        <div class="setup-row">
            <label>Players:</label>
            <button onclick="initGame(2)">2</button>
            <button onclick="initGame(3)">3</button>
            <button onclick="initGame(4)">4</button>
            <button onclick="initGame(5)">5</button>
            <button onclick="initGame(6)">6</button>
            <span style="margin-left: 20px;">AI Seat:</span>
            <select id="ai-seat-select" onchange="changeAiSeat()">
                <option value="0">P0</option>
                <option value="1">P1</option>
                <option value="2">P2</option>
                <option value="3">P3</option>
                <option value="4">P4</option>
                <option value="5">P5</option>
            </select>
        </div>
        <div class="setup-row">
            <label>Network:</label>
            <select id="network-select" onchange="selectNetwork()" style="width: 400px;"></select>
            <span id="network-status" style="color: #888;"></span>
        </div>
    </div>

    <!-- Game State Section -->
    <div class="section">
        <h2>Game State</h2>
        <div id="game-state">No game started. Select player count above.</div>
    </div>

    <!-- AI Private Info Section -->
    <div class="section">
        <h2>AI Private Info</h2>
        <div id="ai-private" class="private-section">
            <div id="ai-hand">Hand: -</div>
            <div id="ai-stack">Stack: -</div>
        </div>
    </div>

    <!-- Actions Section -->
    <div class="section">
        <h2>Actions</h2>
        <div id="current-turn">Current turn: -</div>

        <h3>Place</h3>
        <div id="place-actions" class="action-grid"></div>

        <h3>Bid</h3>
        <div id="bid-actions" class="action-grid"></div>

        <h3>Other</h3>
        <div id="other-actions" class="action-grid"></div>

        <h3>Reveal</h3>
        <div id="reveal-actions" class="action-grid"></div>

        <div id="reveal-prompt" class="reveal-prompt">
            <strong>What was revealed?</strong>
            <button onclick="confirmReveal('rose')">Rose</button>
            <button onclick="confirmReveal('skull')" class="danger">Skull</button>
            <button onclick="cancelReveal()">Cancel</button>
        </div>

        <div id="discard-prompt" class="discard-prompt">
            <strong>AI hit a skull! Choose which card to discard:</strong>
            <button id="discard-skull-btn" onclick="executeDiscard('skull')">Skull</button>
            <button id="discard-rose-btn" onclick="executeDiscard('rose')">Rose</button>
            <button onclick="executeDiscard('random')">Random</button>
        </div>
    </div>

    <!-- AI Probabilities Section -->
    <div class="section">
        <h2>AI Probabilities</h2>
        <div class="controls-row">
            <div class="toggle-group">
                <label>Temp:</label>
                <input type="range" id="temp-slider" min="0.1" max="2.0" step="0.1" value="1.0" oninput="updateTempDisplay()">
                <span id="temp-value">1.0</span>
            </div>
            <button onclick="getAiProbs()">Compute Probs</button>
            <button onclick="aiChoose()">AI Choose</button>
            <button onclick="aiRandom()">Random</button>
        </div>
        <div id="ai-probs" class="ai-probs">Click "Compute Probs" to see AI recommendations</div>
    </div>

    <!-- Controls Section -->
    <div class="section">
        <h2>Controls</h2>
        <div class="controls-row">
            <div class="toggle-group">
                <button id="hide-cards-btn" onclick="toggleHideCards()">Hide Cards</button>
            </div>
            <div class="toggle-group">
                <button id="hide-probs-btn" onclick="toggleHideProbs()">Hide Probs</button>
            </div>
            <div class="toggle-group">
                <button id="undo-btn" onclick="undo()" disabled>Undo</button>
            </div>
        </div>
    </div>

    <!-- History Section -->
    <div class="section">
        <h2>Action History</h2>
        <button onclick="clearHistory()">Clear</button>
        <div id="history"></div>
    </div>

    <script>
        let gameState = null;
        let hideCards = false;
        let hideProbs = false;
        let pendingRevealAction = null;
        let pendingRevealTarget = null;

        // Session management - each tab gets its own isolated session
        function getSessionId() {
            let sessionId = sessionStorage.getItem('skull-session');
            if (!sessionId) {
                sessionId = crypto.randomUUID();
                sessionStorage.setItem('skull-session', sessionId);
            }
            return sessionId;
        }

        // Wrapper for fetch that includes session header
        function apiFetch(url, options = {}) {
            return fetch(url, {
                ...options,
                headers: {
                    ...options.headers,
                    'Content-Type': 'application/json',
                    'X-Session-Id': getSessionId(),
                }
            });
        }

        // Initialize
        window.addEventListener('load', () => {
            updateAiSeatOptions(4);
            loadNetworks();
        });

        function updateAiSeatOptions(numPlayers) {
            const select = document.getElementById('ai-seat-select');
            select.innerHTML = '';
            for (let i = 0; i < numPlayers; i++) {
                const opt = document.createElement('option');
                opt.value = i;
                opt.textContent = `P${i}`;
                select.appendChild(opt);
            }
        }

        async function initGame(numPlayers) {
            updateAiSeatOptions(numPlayers);
            const aiSeat = parseInt(document.getElementById('ai-seat-select').value);

            await apiFetch('/api/new_game', {
                method: 'POST',
                body: JSON.stringify({ num_players: numPlayers, ai_seat: aiSeat })
            });

            await refreshState();
        }

        async function changeAiSeat() {
            const aiSeat = parseInt(document.getElementById('ai-seat-select').value);
            await apiFetch('/api/set_ai_seat', {
                method: 'POST',
                body: JSON.stringify({ ai_seat: aiSeat })
            });
            await refreshState();
        }

        async function loadNetworks() {
            const response = await apiFetch('/api/networks');
            const data = await response.json();

            const select = document.getElementById('network-select');
            select.innerHTML = '';

            for (const net of data.networks) {
                const option = document.createElement('option');
                option.value = net.index;
                option.textContent = net.display_name + ` (step ${net.step})`;
                select.appendChild(option);
            }

            if (data.selected !== null) {
                select.value = data.selected;
                updateNetworkStatus(data.networks[data.selected]);
            }
        }

        async function selectNetwork() {
            const select = document.getElementById('network-select');
            const index = parseInt(select.value);
            const statusEl = document.getElementById('network-status');
            statusEl.textContent = 'Loading...';

            const response = await apiFetch('/api/select_network', {
                method: 'POST',
                body: JSON.stringify({ index })
            });
            const data = await response.json();

            if (data.selected !== null) {
                updateNetworkStatus(data.networks[data.selected]);
            }
        }

        function updateNetworkStatus(network) {
            const statusEl = document.getElementById('network-status');
            statusEl.textContent = `Active (step ${network.step})`;
            statusEl.style.color = '#4CAF50';
        }

        async function refreshState() {
            const response = await apiFetch('/api/game_state');
            gameState = await response.json();
            renderState();
            renderActions();
            await refreshHistory();
        }

        function renderState() {
            if (!gameState || gameState.phase === 'none') {
                document.getElementById('game-state').textContent = 'No game started.';
                document.getElementById('ai-hand').textContent = 'Hand: -';
                document.getElementById('ai-stack').textContent = 'Stack: -';
                return;
            }

            const lines = [];
            const phase = gameState.phase.toUpperCase();
            const bidInfo = gameState.bidding.current_bid > 0
                ? `  |  Bid: ${gameState.bidding.current_bid} by P${gameState.bidding.current_bidder}`
                : '';
            lines.push(`Phase: ${phase}${bidInfo}`);
            lines.push('');

            for (const p of gameState.players) {
                let marker = '';
                if (p.seat === gameState.current_player) marker = '>';
                if (!p.is_alive) marker = 'X';

                const aiTag = p.is_ai ? '(AI)' : '    ';
                const turnTag = p.seat === gameState.current_player ? ' <--' : '';
                const passedTag = p.passed ? ' passed' : '';

                const line = `${marker.padStart(1)} P${p.seat}${aiTag}: ${p.coasters}C ${p.wins}W stack:${p.stack_size} rev:${p.revealed}${passedTag}${turnTag}`;
                lines.push(line);
            }

            if (gameState.game_over) {
                lines.push('');
                lines.push(`GAME OVER! Winner: P${gameState.winner}`);
            }

            document.getElementById('game-state').textContent = lines.join('\n');

            // Update AI private info
            const aiPlayer = gameState.players.find(p => p.is_ai);
            if (aiPlayer && aiPlayer.hand) {
                const handParts = [];
                if (aiPlayer.hand.has_skull) handParts.push(hideCards ? 'Card' : 'Skull');
                for (let i = 0; i < aiPlayer.hand.roses; i++) handParts.push(hideCards ? 'Card' : 'Rose');
                if (handParts.length > 0) {
                    document.getElementById('ai-hand').textContent = `Hand: [${handParts.join('] [')}]`;
                } else {
                    document.getElementById('ai-hand').textContent = 'Hand: (empty)';
                }

                if (aiPlayer.stack_contents && aiPlayer.stack_contents.length > 0) {
                    const stackStr = aiPlayer.stack_contents.map(c => hideCards ? 'Card' : c.charAt(0).toUpperCase() + c.slice(1)).join(', ');
                    document.getElementById('ai-stack').textContent = `Stack: [${stackStr}] (bottom to top)`;
                } else {
                    document.getElementById('ai-stack').textContent = 'Stack: (empty)';
                }
            }

            // Update current turn indicator
            const turnText = gameState.is_ai_turn
                ? `Current turn: P${gameState.current_player} (AI)`
                : `Current turn: P${gameState.current_player}`;
            document.getElementById('current-turn').textContent = turnText;

            // Update undo button state
            document.getElementById('undo-btn').disabled = !gameState.can_undo;
        }

        function renderActions() {
            const placeDiv = document.getElementById('place-actions');
            const bidDiv = document.getElementById('bid-actions');
            const otherDiv = document.getElementById('other-actions');
            const revealDiv = document.getElementById('reveal-actions');

            placeDiv.innerHTML = '';
            bidDiv.innerHTML = '';
            otherDiv.innerHTML = '';
            revealDiv.innerHTML = '';

            if (!gameState || gameState.phase === 'none') return;

            // For non-AI players, show generic "Place Card" button
            const isAiTurn = gameState.is_ai_turn;

            for (const action of gameState.valid_actions) {
                const btn = document.createElement('button');
                btn.className = 'action-btn';

                if (action.index <= 1) {
                    // Place actions
                    if (isAiTurn && hideCards) {
                        // When hiding cards on AI turn, show single "Place Card" button
                        // Only create it once (when we see the first place action)
                        if (action.index === 0 || (action.index === 1 && !gameState.valid_actions.some(a => a.index === 0))) {
                            btn.textContent = 'Place Card';
                            btn.onclick = () => {
                                const temp = parseFloat(document.getElementById('temp-slider').value);
                                executeAction('place_any', null, temp);
                            };
                            btn.className += ' place';
                            placeDiv.appendChild(btn);
                        }
                        continue;
                    } else if (isAiTurn) {
                        // AI turn without hiding: show specific buttons
                        btn.textContent = action.name;
                    } else {
                        // For non-AI, just show "Place Card" once for the first place action
                        if (action.index === 0 || (action.index === 1 && !gameState.valid_actions.some(a => a.index === 0))) {
                            btn.textContent = 'Place Card';
                            btn.onclick = () => executeAction(action.index);
                            btn.className += ' place';
                            placeDiv.appendChild(btn);
                        }
                        continue;
                    }
                    btn.onclick = () => executeAction(action.index);
                    btn.className += ' place';
                    placeDiv.appendChild(btn);
                } else if (action.index >= 2 && action.index <= 25) {
                    // Bid actions
                    btn.textContent = action.name;
                    btn.onclick = () => executeAction(action.index);
                    btn.className += ' bid';
                    bidDiv.appendChild(btn);
                } else if (action.index === 26) {
                    // Pass
                    btn.textContent = 'Pass';
                    btn.onclick = () => executeAction(action.index);
                    otherDiv.appendChild(btn);
                } else if (action.index >= 27) {
                    // Reveal actions
                    const targetPlayer = action.index - 27;
                    btn.textContent = `Reveal P${targetPlayer}`;
                    btn.onclick = () => startReveal(action.index, targetPlayer);
                    btn.className += ' reveal';
                    revealDiv.appendChild(btn);
                }
            }
        }

        function startReveal(action, targetPlayer) {
            // If revealing AI's stack, we know the cards
            if (targetPlayer === gameState.ai_seat) {
                executeAction(action);
                return;
            }

            // Otherwise, prompt for outcome
            pendingRevealAction = action;
            pendingRevealTarget = targetPlayer;
            document.getElementById('reveal-prompt').classList.add('active');
        }

        function confirmReveal(outcome) {
            document.getElementById('reveal-prompt').classList.remove('active');
            executeAction(pendingRevealAction, outcome);
            pendingRevealAction = null;
            pendingRevealTarget = null;
        }

        function cancelReveal() {
            document.getElementById('reveal-prompt').classList.remove('active');
            pendingRevealAction = null;
            pendingRevealTarget = null;
        }

        async function executeAction(action, revealOutcome = null, temperature = null) {
            const body = { action };
            if (revealOutcome) body.reveal_outcome = revealOutcome;
            if (temperature !== null) body.temperature = temperature;

            const response = await apiFetch('/api/execute_action', {
                method: 'POST',
                body: JSON.stringify(body)
            });
            const data = await response.json();

            // Check if we need to show discard choice
            if (data.needs_discard_choice && data.discard_options) {
                showDiscardPrompt(data.discard_options);
                return;
            }

            await refreshState();
        }

        function showDiscardPrompt(options) {
            const prompt = document.getElementById('discard-prompt');
            const skullBtn = document.getElementById('discard-skull-btn');
            const roseBtn = document.getElementById('discard-rose-btn');

            // Enable/disable buttons based on what AI has
            skullBtn.disabled = !options.has_skull;
            roseBtn.disabled = !options.has_rose;

            prompt.classList.add('active');
        }

        async function executeDiscard(choice) {
            document.getElementById('discard-prompt').classList.remove('active');

            await apiFetch('/api/execute_discard', {
                method: 'POST',
                body: JSON.stringify({ choice })
            });

            await refreshState();
        }

        async function getAiProbs() {
            const temp = parseFloat(document.getElementById('temp-slider').value);
            const response = await apiFetch('/api/ai_probs', {
                method: 'POST',
                body: JSON.stringify({ temperature: temp })
            });
            const data = await response.json();

            const div = document.getElementById('ai-probs');

            if (hideProbs) {
                div.textContent = '(Probabilities hidden)';
                return;
            }

            if (!data.actions || data.actions.length === 0) {
                div.textContent = 'No valid actions';
                return;
            }

            // Sort by probability descending
            data.actions.sort((a, b) => b.prob - a.prob);

            let html = '';
            for (const a of data.actions) {
                const pct = (a.prob * 100).toFixed(1);
                const barWidth = Math.max(2, a.prob * 200);
                html += `<div><span class="prob-bar" style="width:${barWidth}px"></span>${a.name}: ${pct}%</div>`;
            }
            div.innerHTML = html;
        }

        async function aiChoose() {
            const temp = parseFloat(document.getElementById('temp-slider').value);
            const response = await apiFetch('/api/ai_sample', {
                method: 'POST',
                body: JSON.stringify({ temperature: temp, random: false })
            });
            const data = await response.json();

            // For reveal actions on non-AI players, we need outcome
            if (data.action >= 27 && data.action - 27 !== gameState.ai_seat) {
                alert(`AI chose: ${data.action_name}\nYou need to specify the reveal outcome.`);
                startReveal(data.action, data.action - 27);
                return;
            }

            await executeAction(data.action);
        }

        async function aiRandom() {
            const response = await apiFetch('/api/ai_sample', {
                method: 'POST',
                body: JSON.stringify({ temperature: 1.0, random: true })
            });
            const data = await response.json();

            // For reveal actions on non-AI players, we need outcome
            if (data.action >= 27 && data.action - 27 !== gameState.ai_seat) {
                alert(`Random chose: ${data.action_name}\nYou need to specify the reveal outcome.`);
                startReveal(data.action, data.action - 27);
                return;
            }

            await executeAction(data.action);
        }

        function updateTempDisplay() {
            document.getElementById('temp-value').textContent = document.getElementById('temp-slider').value;
        }

        function toggleHideCards() {
            hideCards = !hideCards;
            const btn = document.getElementById('hide-cards-btn');
            btn.classList.toggle('active', hideCards);
            btn.textContent = hideCards ? 'Show Cards' : 'Hide Cards';

            // Re-render to show Card vs Skull/Rose
            renderState();
            renderActions();

            apiFetch('/api/set_hide_info', {
                method: 'POST',
                body: JSON.stringify({ hide: hideCards })
            }).then(() => refreshHistory());
        }

        function toggleHideProbs() {
            hideProbs = !hideProbs;
            const btn = document.getElementById('hide-probs-btn');
            btn.classList.toggle('active', hideProbs);
            btn.textContent = hideProbs ? 'Show Probs' : 'Hide Probs';

            if (hideProbs) {
                document.getElementById('ai-probs').textContent = '(Probabilities hidden)';
            }
        }

        async function refreshHistory() {
            const response = await apiFetch('/api/history');
            const history = await response.json();

            const div = document.getElementById('history');
            div.innerHTML = '';

            for (const entry of history) {
                const line = document.createElement('div');
                let text = `P${entry.player}: ${entry.action_name}`;
                if (entry.reveal_outcome) {
                    text += ` → ${entry.reveal_outcome}`;
                }
                line.textContent = text;
                div.appendChild(line);
            }

            div.scrollTop = div.scrollHeight;
        }

        async function clearHistory() {
            await apiFetch('/api/clear_history', { method: 'POST' });
            await refreshHistory();
        }

        async function undo() {
            const response = await apiFetch('/api/undo', { method: 'POST' });
            const data = await response.json();
            if (!data.success) {
                console.log('Undo failed:', data.error);
            }
            await refreshState();
        }
    </script>
</body>
</html>
"#;
