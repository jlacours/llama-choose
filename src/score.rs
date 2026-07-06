//! Normalized 0–100 scores so every model can be compared at a glance.
//!
//! Raw stats live in different units (tokens/s, pass rates, launch counts);
//! this module maps each onto a common 0–100 scale and folds them into one
//! weighted overall score. A score is `None` until its underlying data
//! exists, and missing components simply drop out of the overall instead of
//! dragging it down.

/// Generation speed that earns a full 100 — anything at or above this feels
/// instant for interactive use on this machine.
const TG_CEIL: f64 = 60.0;

/// Prompt-processing speed that earns a full 100 — a 4K-token prompt in
/// about four seconds.
const PP_CEIL: f64 = 1000.0;

/// Overall-score weights: what the model answers matters most, how fast it
/// answers second, whether it actually gets used third, and prompt-eval
/// speed is a tie-breaker.
const W_INTELLIGENCE: f64 = 0.4;
const W_OUTPUT: f64 = 0.3;
const W_USAGE: f64 = 0.2;
const W_INPUT: f64 = 0.1;

#[derive(Clone, Copy, Default)]
pub struct Scores {
    /// Generation (output tokens/s) mapped onto 0–100.
    pub output: Option<f64>,
    /// Prompt processing (input tokens/s) mapped onto 0–100.
    pub input: Option<f64>,
    /// Average of the chat/code benchmark scores (already 0–100).
    pub intelligence: Option<f64>,
    /// Launch count relative to the most-used model in the fleet.
    pub usage: Option<f64>,
    /// Weighted average of whichever components exist.
    pub overall: Option<f64>,
}

/// Square-root ramp to a ceiling: early tokens/s gains matter more than the
/// difference between fast and very fast, and everything at or past `ceil`
/// is a 100.
fn ramp(v: f64, ceil: f64) -> f64 {
    (v / ceil).clamp(0.0, 1.0).sqrt() * 100.0
}

pub fn compute(
    tg_avg: Option<f64>,
    pp_avg: Option<f64>,
    chat: Option<f64>,
    code: Option<f64>,
    launches: u64,
    fleet_max_launches: u64,
) -> Scores {
    let output = tg_avg.map(|v| ramp(v, TG_CEIL));
    let input = pp_avg.map(|v| ramp(v, PP_CEIL));

    let intelligence = match (chat, code) {
        (Some(c), Some(d)) => Some((c + d) / 2.0),
        (Some(s), None) | (None, Some(s)) => Some(s),
        (None, None) => None,
    };

    // Relative share of fleet usage: the most-launched model scores 100 and a
    // never-launched one scores 0. The square root keeps a lightly used model
    // from looking abandoned next to the daily driver. With no launches
    // anywhere there is nothing to compare against.
    let usage = (fleet_max_launches > 0).then(|| ramp(launches as f64, fleet_max_launches as f64));

    let mut weighted = 0.0;
    let mut weight = 0.0;
    for (score, w) in [
        (intelligence, W_INTELLIGENCE),
        (output, W_OUTPUT),
        (usage, W_USAGE),
        (input, W_INPUT),
    ] {
        if let Some(s) = score {
            weighted += s * w;
            weight += w;
        }
    }
    let overall = (weight > 0.0).then(|| weighted / weight);

    Scores {
        output,
        input,
        intelligence,
        usage,
        overall,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_caps_at_ceiling_and_floors_at_zero() {
        assert_eq!(ramp(TG_CEIL, TG_CEIL), 100.0);
        assert_eq!(ramp(TG_CEIL * 3.0, TG_CEIL), 100.0);
        assert_eq!(ramp(0.0, TG_CEIL), 0.0);
        // Square root: a quarter of the ceiling scores half, not a quarter.
        assert!((ramp(15.0, 60.0) - 50.0).abs() < 1e-9);
    }

    #[test]
    fn no_data_means_no_scores() {
        let s = compute(None, None, None, None, 0, 0);
        assert!(s.output.is_none());
        assert!(s.input.is_none());
        assert!(s.intelligence.is_none());
        assert!(s.usage.is_none());
        assert!(s.overall.is_none());
    }

    #[test]
    fn intelligence_averages_available_suites() {
        let s = compute(None, None, Some(80.0), Some(60.0), 0, 0);
        assert_eq!(s.intelligence, Some(70.0));
        let s = compute(None, None, Some(80.0), None, 0, 0);
        assert_eq!(s.intelligence, Some(80.0));
    }

    #[test]
    fn usage_is_relative_to_fleet_max() {
        let s = compute(None, None, None, None, 19, 19);
        assert_eq!(s.usage, Some(100.0));
        let s = compute(None, None, None, None, 0, 19);
        assert_eq!(s.usage, Some(0.0));
        // Never-launched fleet: usage is unknown, not zero.
        let s = compute(None, None, None, None, 0, 0);
        assert!(s.usage.is_none());
    }

    #[test]
    fn overall_renormalizes_over_missing_components() {
        // Only intelligence exists → overall equals it exactly.
        let s = compute(None, None, Some(90.0), Some(90.0), 0, 0);
        assert_eq!(s.overall, Some(90.0));

        // All components present → plain weighted average.
        let s = compute(Some(60.0), Some(1000.0), Some(100.0), Some(100.0), 19, 19);
        // output 100, input 100, intelligence 100, usage 100 → overall 100.
        assert!((s.overall.unwrap() - 100.0).abs() < 1e-9);
    }
}
