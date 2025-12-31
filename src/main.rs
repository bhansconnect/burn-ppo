mod config;
mod env;
mod envs;
mod network;
mod utils;

use anyhow::Result;
use clap::Parser;

use crate::config::{CliArgs, Config};

fn main() -> Result<()> {
    let args = CliArgs::parse();
    let config = Config::load(&args)?;

    println!("burn-ppo v{}", env!("CARGO_PKG_VERSION"));
    println!("Environment: {}", config.env);
    println!(
        "Num envs: {} (resolved: {})",
        match &config.num_envs {
            config::NumEnvs::Auto(_) => "auto".to_string(),
            config::NumEnvs::Explicit(n) => n.to_string(),
        },
        config.num_envs()
    );
    println!("Seed: {}", config.seed);
    println!("Run: {}", config.run_name.as_ref().unwrap());

    // TODO: Initialize device, environments, network, and training loop

    Ok(())
}
