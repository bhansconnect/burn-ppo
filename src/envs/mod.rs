pub mod cartpole;
pub mod connect_four;
pub mod liars_dice;

pub use cartpole::CartPole;
pub use connect_four::ConnectFour;
pub use liars_dice::LiarsDice;

/// Dispatch to the correct environment type based on `env_name`.
/// Uses compile-time monomorphization for zero runtime overhead.
///
/// Usage:
/// ```ignore
/// dispatch_env!("cartpole", {
///     // Type E is now CartPole
///     run_something::<E>()
/// });
/// ```
#[macro_export]
macro_rules! dispatch_env {
    ($env_name:expr, $callback:expr) => {{
        let name: &str = $env_name.as_str();
        match name {
            "cartpole" => {
                type E = $crate::envs::CartPole;
                $callback
            }
            "connect_four" => {
                type E = $crate::envs::ConnectFour;
                $callback
            }
            "liars_dice" => {
                type E = $crate::envs::LiarsDice;
                $callback
            }
            _ => {
                anyhow::bail!(
                    "Unknown environment: '{}'. Supported: cartpole, connect_four, liars_dice",
                    name
                )
            }
        }
    }};
}

/// Like [`dispatch_env!`] but for callbacks that return `T` instead of `Result<T>`.
/// Wraps the callback result in `Ok()` automatically.
#[macro_export]
macro_rules! dispatch_env_ok {
    ($env_name:expr, $callback:expr) => {{
        let name: &str = $env_name.as_str();
        match name {
            "cartpole" => {
                type E = $crate::envs::CartPole;
                Ok($callback)
            }
            "connect_four" => {
                type E = $crate::envs::ConnectFour;
                Ok($callback)
            }
            "liars_dice" => {
                type E = $crate::envs::LiarsDice;
                Ok($callback)
            }
            _ => {
                anyhow::bail!(
                    "Unknown environment: '{}'. Supported: cartpole, connect_four, liars_dice",
                    name
                )
            }
        }
    }};
}

#[cfg(test)]
mod tests {
    use crate::env::Environment;

    fn get_env_name<E: Environment>() -> &'static str {
        E::NAME
    }

    fn get_obs_dim<E: Environment>() -> usize {
        E::OBSERVATION_DIM
    }

    // Wrapper functions that return Result to allow anyhow::bail! to work
    fn dispatch_cartpole() -> anyhow::Result<&'static str> {
        let name = "cartpole".to_string();
        crate::dispatch_env!(name, Ok(get_env_name::<E>()))
    }

    fn dispatch_connect_four() -> anyhow::Result<&'static str> {
        let name = "connect_four".to_string();
        crate::dispatch_env!(name, Ok(get_env_name::<E>()))
    }

    fn dispatch_liars_dice() -> anyhow::Result<&'static str> {
        let name = "liars_dice".to_string();
        crate::dispatch_env!(name, Ok(get_env_name::<E>()))
    }

    fn dispatch_get_obs_dim(env_name: &str) -> anyhow::Result<usize> {
        let name = env_name.to_string();
        crate::dispatch_env!(name, Ok(get_obs_dim::<E>()))
    }

    fn dispatch_unknown() -> anyhow::Result<&'static str> {
        let name = "unknown_env".to_string();
        crate::dispatch_env!(name, Ok(E::NAME))
    }

    fn dispatch_case_sensitive() -> anyhow::Result<&'static str> {
        let name = "CartPole".to_string();
        crate::dispatch_env!(name, Ok(E::NAME))
    }

    #[test]
    fn test_dispatch_env_cartpole() {
        let result = dispatch_cartpole();
        assert_eq!(result.unwrap(), "cartpole");
    }

    #[test]
    fn test_dispatch_env_connect_four() {
        let result = dispatch_connect_four();
        assert_eq!(result.unwrap(), "connect_four");
    }

    #[test]
    fn test_dispatch_env_liars_dice() {
        let result = dispatch_liars_dice();
        assert_eq!(result.unwrap(), "liars_dice");
    }

    #[test]
    fn test_dispatch_env_gets_correct_obs_dim() {
        assert_eq!(dispatch_get_obs_dim("cartpole").unwrap(), 5); // CartPole: [x, x_dot, theta, theta_dot, time]
        assert_eq!(dispatch_get_obs_dim("connect_four").unwrap(), 86); // ConnectFour: 42*2 + 2
        assert_eq!(dispatch_get_obs_dim("liars_dice").unwrap(), 270); // LiarsDice: 78 base + 192 bid history
    }

    #[test]
    fn test_dispatch_env_unknown_returns_error() {
        let result = dispatch_unknown();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Unknown environment"));
        assert!(err_msg.contains("unknown_env"));
    }

    #[test]
    fn test_dispatch_env_case_sensitive() {
        // Environment names are case-sensitive
        let result = dispatch_case_sensitive();
        assert!(result.is_err()); // Should fail - it's "cartpole", not "CartPole"
    }
}
