//! Schedule type for piecewise-linear parameter schedules.
//!
//! Schedules are defined as a list of (value, step) milestones with linear interpolation
//! between them. After the last milestone, the value is held constant.
//!
//! # Examples
//!
//! ```
//! use burn_ppo::schedule::Schedule;
//!
//! // Constant schedule
//! let s = Schedule::constant(0.001);
//! assert_eq!(s.get(0), 0.001);
//! assert_eq!(s.get(1_000_000), 0.001);
//!
//! // Linear decay from 0.001 to 0.0001 over 30M steps
//! let s = Schedule::new(vec![(0.001, 0), (0.0001, 30_000_000)]);
//! assert_eq!(s.get(15_000_000), 0.00055);  // halfway
//! ```

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// A piecewise-linear schedule defined by (value, step) milestones.
///
/// Values are linearly interpolated between milestones. After the last
/// milestone, the final value is held constant.
#[derive(Debug, Clone, PartialEq)]
pub struct Schedule(pub Vec<(f64, u64)>);

impl Schedule {
    /// Create a new schedule from a list of (value, step) milestones.
    ///
    /// Milestones should be sorted by step in ascending order.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "used in tests and for external API")
    )]
    pub fn new(milestones: Vec<(f64, u64)>) -> Self {
        Schedule(milestones)
    }

    /// Create a constant schedule that always returns the given value.
    pub fn constant(value: f64) -> Self {
        Schedule(vec![(value, 0)])
    }

    /// Get the interpolated value at the given step.
    ///
    /// - Before the first milestone: returns the first value
    /// - Between milestones: linear interpolation
    /// - After the last milestone: returns the last value
    /// - Empty schedule: returns 0.0
    pub fn get(&self, step: u64) -> f64 {
        if self.0.is_empty() {
            return 0.0;
        }

        // Before first milestone or single point: return first value
        if self.0.len() == 1 || step <= self.0[0].1 {
            return self.0[0].0;
        }

        // Find the segment we're in
        for i in 0..self.0.len() - 1 {
            let (v1, s1) = self.0[i];
            let (v2, s2) = self.0[i + 1];

            if step >= s1 && step < s2 {
                // Linear interpolation
                let t = (step - s1) as f64 / (s2 - s1) as f64;
                return v1 + (v2 - v1) * t;
            }
        }

        // Past last milestone: return last value
        self.0.last().map_or(0.0, |(v, _)| *v)
    }

    /// Check if this is a constant schedule (single milestone at step 0).
    pub fn is_constant(&self) -> bool {
        self.0.len() == 1 && self.0[0].1 == 0
    }

    /// Get the initial value (value at step 0).
    pub fn initial_value(&self) -> f64 {
        self.get(0)
    }

    /// Parse a schedule from CLI string format.
    ///
    /// Format: `value` for static, or `value@step,value@step,...` for schedule.
    /// Step values can use K (1000) or M (1000000) suffixes.
    ///
    /// # Examples
    ///
    /// ```
    /// use burn_ppo::schedule::Schedule;
    ///
    /// // Static value
    /// let s = Schedule::parse_cli("0.001").unwrap();
    /// assert_eq!(s.get(0), 0.001);
    ///
    /// // Schedule with suffixes
    /// let s = Schedule::parse_cli("0.001@0,0.0001@30M").unwrap();
    /// assert_eq!(s.0[1].1, 30_000_000);
    /// ```
    pub fn parse_cli(s: &str) -> Result<Self, String> {
        let s = s.trim();

        // If no @ present, treat as static value
        if !s.contains('@') {
            let value: f64 = s
                .parse()
                .map_err(|e| format!("Invalid number '{s}': {e}"))?;
            return Ok(Schedule::constant(value));
        }

        // Parse as schedule: value@step,value@step,...
        let mut milestones = Vec::new();

        for part in s.split(',') {
            let part = part.trim();
            let parts: Vec<&str> = part.split('@').collect();

            if parts.len() != 2 {
                return Err(format!("Invalid milestone '{part}': expected 'value@step'"));
            }

            let value: f64 = parts[0]
                .parse()
                .map_err(|e| format!("Invalid value '{}': {}", parts[0], e))?;

            let step = parse_step_with_suffix(parts[1])?;

            milestones.push((value, step));
        }

        if milestones.is_empty() {
            return Err("Empty schedule".to_string());
        }

        // Sort by step
        milestones.sort_by_key(|(_, s)| *s);

        Ok(Schedule(milestones))
    }
}

/// Parse a step value with optional K/M suffix.
fn parse_step_with_suffix(s: &str) -> Result<u64, String> {
    let s = s.trim();

    if s.is_empty() {
        return Err("Empty step value".to_string());
    }

    // Check for suffix
    let (num_str, multiplier) = if s.ends_with('M') || s.ends_with('m') {
        (&s[..s.len() - 1], 1_000_000u64)
    } else if s.ends_with('K') || s.ends_with('k') {
        (&s[..s.len() - 1], 1_000u64)
    } else {
        (s, 1u64)
    };

    let num: f64 = num_str
        .parse()
        .map_err(|e| format!("Invalid step '{num_str}': {e}"))?;

    if num < 0.0 {
        return Err(format!("Step value cannot be negative: '{num_str}'"));
    }

    // Safety: num >= 0.0 is validated above, so cast cannot lose sign
    #[expect(
        clippy::cast_sign_loss,
        reason = "num >= 0.0 validated above, sign cannot be lost"
    )]
    let result = (num * multiplier as f64) as u64;
    Ok(result)
}

impl Default for Schedule {
    fn default() -> Self {
        Schedule::constant(0.0)
    }
}

impl fmt::Display for Schedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_constant() {
            write!(f, "{}", self.0[0].0)
        } else {
            let parts: Vec<String> = self.0.iter().map(|(v, s)| format!("{v}@{s}")).collect();
            write!(f, "{}", parts.join(","))
        }
    }
}

// Custom serde implementation to handle both f64 and [[f64, u64], ...]

impl Serialize for Schedule {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.is_constant() {
            // Serialize as single f64
            self.0[0].0.serialize(serializer)
        } else {
            // Serialize as array of [value, step] pairs
            self.0.serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for Schedule {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ScheduleVisitor;

        impl<'de> de::Visitor<'de> for ScheduleVisitor {
            type Value = Schedule;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a number or array of [value, step] pairs")
            }

            fn visit_f64<E>(self, value: f64) -> Result<Schedule, E>
            where
                E: de::Error,
            {
                Ok(Schedule::constant(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Schedule, E>
            where
                E: de::Error,
            {
                Ok(Schedule::constant(value as f64))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Schedule, E>
            where
                E: de::Error,
            {
                Ok(Schedule::constant(value as f64))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Schedule, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let mut milestones = Vec::new();

                while let Some(pair) = seq.next_element::<(f64, u64)>()? {
                    milestones.push(pair);
                }

                if milestones.is_empty() {
                    return Err(de::Error::custom("empty schedule"));
                }

                // Sort by step
                milestones.sort_by_key(|(_, s)| *s);

                Ok(Schedule(milestones))
            }
        }

        deserializer.deserialize_any(ScheduleVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[test]
    fn test_constant_schedule() {
        let s = Schedule::constant(0.001);
        assert_eq!(s.get(0), 0.001);
        assert_eq!(s.get(1_000_000), 0.001);
        assert_eq!(s.get(100_000_000), 0.001);
    }

    #[test]
    fn test_linear_decay() {
        let s = Schedule::new(vec![(1.0, 0), (0.0, 100)]);
        assert_eq!(s.get(0), 1.0);
        assert_eq!(s.get(50), 0.5);
        assert_eq!(s.get(100), 0.0);
        assert_eq!(s.get(200), 0.0); // holds at end
    }

    #[test]
    fn test_multi_stage() {
        let s = Schedule::new(vec![(1.0, 0), (0.5, 100), (0.1, 200)]);
        assert_eq!(s.get(0), 1.0);
        assert_eq!(s.get(50), 0.75);
        assert_eq!(s.get(100), 0.5);
        assert_eq!(s.get(150), 0.3);
        assert_eq!(s.get(200), 0.1);
        assert_eq!(s.get(300), 0.1);
    }

    #[test]
    fn test_warmup() {
        let s = Schedule::new(vec![(0.0, 0), (1.0, 100)]);
        assert_eq!(s.get(0), 0.0);
        assert_eq!(s.get(50), 0.5);
        assert_eq!(s.get(100), 1.0);
    }

    #[test]
    fn test_hold_period() {
        let s = Schedule::new(vec![(1.0, 0), (1.0, 100), (0.0, 200)]);
        assert_eq!(s.get(0), 1.0);
        assert_eq!(s.get(50), 1.0);
        assert_eq!(s.get(100), 1.0);
        assert_eq!(s.get(150), 0.5);
    }

    #[test]
    fn test_parse_cli_static() {
        let s = Schedule::parse_cli("0.001").unwrap();
        assert_eq!(s.0, vec![(0.001, 0)]);
    }

    #[test]
    fn test_parse_cli_schedule() {
        let s = Schedule::parse_cli("0.001@0,0.0001@30000000").unwrap();
        assert_eq!(s.0, vec![(0.001, 0), (0.0001, 30_000_000)]);
    }

    #[test]
    fn test_parse_cli_k_suffix() {
        let s = Schedule::parse_cli("0.001@0,0.0001@500K").unwrap();
        assert_eq!(s.0, vec![(0.001, 0), (0.0001, 500_000)]);
    }

    #[test]
    fn test_parse_cli_m_suffix() {
        let s = Schedule::parse_cli("0.001@0,0.0001@30M").unwrap();
        assert_eq!(s.0, vec![(0.001, 0), (0.0001, 30_000_000)]);
    }

    #[test]
    fn test_parse_cli_mixed_suffixes() {
        let s = Schedule::parse_cli("0.001@0,0.0005@500K,0.0001@30M").unwrap();
        assert_eq!(
            s.0,
            vec![(0.001, 0), (0.0005, 500_000), (0.0001, 30_000_000)]
        );
    }

    #[test]
    fn test_parse_cli_lowercase_suffix() {
        let s = Schedule::parse_cli("0.001@0,0.0001@30m").unwrap();
        assert_eq!(s.0, vec![(0.001, 0), (0.0001, 30_000_000)]);

        let s = Schedule::parse_cli("0.001@0,0.0001@500k").unwrap();
        assert_eq!(s.0, vec![(0.001, 0), (0.0001, 500_000)]);
    }

    #[test]
    fn test_parse_cli_fractional_with_suffix() {
        // 1.5M = 1,500,000
        let s = Schedule::parse_cli("0.001@0,0.0001@1.5M").unwrap();
        assert_eq!(s.0, vec![(0.001, 0), (0.0001, 1_500_000)]);
    }

    #[test]
    fn test_single_point_behavior() {
        let s = Schedule::new(vec![(0.5, 1000)]);
        // Before the single point - returns first value
        assert_eq!(s.get(0), 0.5);
        assert_eq!(s.get(999), 0.5);
        // At and after
        assert_eq!(s.get(1000), 0.5);
        assert_eq!(s.get(2000), 0.5);
    }

    #[test]
    fn test_empty_schedule_returns_zero() {
        let s = Schedule::new(vec![]);
        assert_eq!(s.get(0), 0.0);
        assert_eq!(s.get(1000), 0.0);
    }

    #[test]
    fn test_is_constant() {
        assert!(Schedule::constant(0.001).is_constant());
        assert!(Schedule::new(vec![(0.001, 0)]).is_constant());
        assert!(!Schedule::new(vec![(0.001, 0), (0.0001, 100)]).is_constant());
        assert!(!Schedule::new(vec![(0.001, 100)]).is_constant()); // not at step 0
    }

    #[test]
    fn test_display() {
        assert_eq!(Schedule::constant(0.001).to_string(), "0.001");
        assert_eq!(
            Schedule::new(vec![(0.001, 0), (0.0001, 30_000_000)]).to_string(),
            "0.001@0,0.0001@30000000"
        );
    }

    #[test]
    fn test_serialize_constant() {
        #[derive(Serialize)]
        struct Test {
            value: Schedule,
        }

        let t = Test {
            value: Schedule::constant(0.001),
        };
        let toml_str = toml::to_string(&t).unwrap();
        assert_eq!(toml_str.trim(), "value = 0.001");
    }

    #[test]
    fn test_serialize_schedule() {
        #[derive(Serialize)]
        struct Test {
            value: Schedule,
        }

        let t = Test {
            value: Schedule::new(vec![(0.001, 0), (0.0001, 30_000_000)]),
        };
        let toml_str = toml::to_string(&t).unwrap();
        assert!(toml_str.contains("0.001"));
        assert!(toml_str.contains("30000000"));
    }

    #[test]
    fn test_deserialize_float() {
        #[derive(Deserialize)]
        struct Test {
            value: Schedule,
        }

        let t: Test = toml::from_str("value = 0.001").unwrap();
        assert_eq!(t.value.0, vec![(0.001, 0)]);
    }

    #[test]
    fn test_deserialize_integer() {
        #[derive(Deserialize)]
        struct Test {
            value: Schedule,
        }

        let t: Test = toml::from_str("value = 1").unwrap();
        assert_eq!(t.value.0, vec![(1.0, 0)]);
    }

    #[test]
    fn test_deserialize_schedule() {
        #[derive(Deserialize)]
        struct Test {
            value: Schedule,
        }

        let t: Test = toml::from_str("value = [[0.001, 0], [0.0001, 30000000]]").unwrap();
        assert_eq!(t.value.0, vec![(0.001, 0), (0.0001, 30_000_000)]);
    }

    #[test]
    fn test_deserialize_unsorted_gets_sorted() {
        #[derive(Deserialize)]
        struct Test {
            value: Schedule,
        }

        // Milestones out of order
        let t: Test = toml::from_str("value = [[0.0001, 30000000], [0.001, 0]]").unwrap();
        // Should be sorted by step
        assert_eq!(t.value.0, vec![(0.001, 0), (0.0001, 30_000_000)]);
    }

    #[test]
    fn test_parse_cli_sorts_milestones() {
        let s = Schedule::parse_cli("0.0001@30M,0.001@0").unwrap();
        assert_eq!(s.0, vec![(0.001, 0), (0.0001, 30_000_000)]);
    }

    #[test]
    fn test_initial_value() {
        assert_eq!(Schedule::constant(0.001).initial_value(), 0.001);
        assert_eq!(
            Schedule::new(vec![(0.001, 0), (0.0001, 100)]).initial_value(),
            0.001
        );
        // Even if first milestone isn't at 0, initial_value returns value at step 0
        assert_eq!(Schedule::new(vec![(0.5, 100)]).initial_value(), 0.5);
    }

    #[test]
    fn test_boundary_conditions() {
        let s = Schedule::new(vec![(1.0, 0), (0.0, 100)]);

        // Exactly at boundaries
        assert_eq!(s.get(0), 1.0);
        assert_eq!(s.get(100), 0.0);

        // One step before/after boundaries
        assert!((s.get(1) - 0.99).abs() < 0.001);
        assert!((s.get(99) - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_very_large_steps() {
        let s = Schedule::new(vec![(1.0, 0), (0.0, 1_000_000_000)]);
        assert_eq!(s.get(500_000_000), 0.5);
        assert_eq!(s.get(1_000_000_000), 0.0);
        assert_eq!(s.get(2_000_000_000), 0.0);
    }

    #[test]
    fn test_parse_cli_whitespace() {
        let s = Schedule::parse_cli("  0.001@0 , 0.0001@30M  ").unwrap();
        assert_eq!(s.0, vec![(0.001, 0), (0.0001, 30_000_000)]);
    }

    #[test]
    fn test_parse_cli_error_invalid_number() {
        assert!(Schedule::parse_cli("abc").is_err());
        assert!(Schedule::parse_cli("0.001@abc").is_err());
    }

    #[test]
    fn test_parse_cli_error_invalid_format() {
        assert!(Schedule::parse_cli("0.001@0@100").is_err());
    }
}
