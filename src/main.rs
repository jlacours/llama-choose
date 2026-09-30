//! llama-choose: pick and launch a local inference server, and quietly keep
//! score so you know which models are worth their disk space.
//!
//! Configured models come from `~/.local/share/llama-models.ini` (the same
//! preset file `llama-server` router mode reads), while disk-only GGUFs are
//! discovered under `~/models`. Usage counts and generation speeds are
//! scraped from the server's own logs into a SQLite store and surfaced in the
//! picker.
//!
//! Subcommands:
//!   llama-choose            interactive picker (default)
//!   llama-choose stats      print the usage/speed table and exit
//!   llama-choose stop       stop any running inference server
//!   llama-choose -h         help

mod benchmark;
mod config;
mod db;
mod launch;
mod meta;
mod score;
mod tui;

use std::io::Read;
use std::process::Command;

use config::Model;
use rusqlite::Connection;
use tui::ModelView;

const SERVER_PORT: u16 = 3002;
const VLLM_PORT: u16 = 8000;

/// vLLM presets are independent of the GGUF INI (different runtime, different
/// log format), so they stay hard-coded as in the original script.
const VLLM_MODELS: [(&str, &str); 3] = [
    ("qwen3-0.6b", "Tiny smoke test, Qwen3 tools, low VRAM"),
    (
        "qwen2.5-1.5b-instruct",
        "Small chat model, comfortable on 12GB",
    ),
    (
        "qwen2.5-7b-instruct-awq",
        "Quantized 7B chat model, fits 12GB sanely",
    ),
];

fn home() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/root".into())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = match args.get(1).map(String::as_str) {
        None => interactive(),
        Some("stats") => cmd_stats(),
        Some("check") => cmd_check(args.get(2)),
        Some("bench") => cmd_bench(args.get(2), args.get(3)),
        Some("launch") => cmd_launch(args.get(2), args.get(3)),
        Some("stop") => cmd_stop(),
        Some("-h" | "--help" | "help") => {
            print_help();
            0
        }
        Some(other) => {
            eprintln!("llama-choose: unknown argument '{other}'\n");
            print_help();
            2
        }
    };
    std::process::exit(code);
}

fn print_help() {
    println!(
        "llama-choose — pick & launch a local inference server, with usage stats\n\n\
         usage:\n  \
         llama-choose          interactive model picker (default)\n  \
         llama-choose stats    print the usage / tokens-per-second table\n  \
         llama-choose check [ALIAS]\n                       validate model files without launching anything\n  \
         llama-choose bench ALIAS|all [chat|code]\n                       run dynamic correctness benchmarks via the router\n  \
         llama-choose launch ALIAS [tools|server|phone|shared|cli]\n                       launch one model non-interactively (default: tools)\n  \
         llama-choose stop     stop the running inference server\n  \
         llama-choose -h       show this help\n\n\
         configured models are read from ~/.local/share/llama-models.ini\n\
         standalone GGUFs are discovered recursively under ~/models"
    );
}

// ---------------------------------------------------------------------------
// running-process detection
// ---------------------------------------------------------------------------

fn pgrep(args: &[&str]) -> Vec<String> {
    Command::new("pgrep")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn running_pids() -> (Vec<String>, Vec<String>) {
    (pgrep(&["-x", "llama-server"]), pgrep(&["-f", "vllm serve"]))
}

// ---------------------------------------------------------------------------
// building the preview model
// ---------------------------------------------------------------------------

fn format_ctx(n: u64) -> String {
    if n >= 1024 && n.is_multiple_of(1024) {
        format!("{}K ({n})", n / 1024)
    } else {
        n.to_string()
    }
}

fn offload_str(m: &Model) -> String {
    match (m.get("n-gpu-layers"), m.get("n-cpu-moe")) {
        (Some(g), Some(c)) => format!("GPU layers {g} · CPU-MoE {c}"),
        (Some(g), None) => format!("GPU layers {g}"),
        (None, Some(c)) => format!("CPU-MoE {c}"),
        (None, None) => "default".into(),
    }
}

fn spec_str(m: &Model) -> Option<String> {
    let ty = m.get("spec-type")?;
    let drafter = if m.get("model-draft").is_some() {
        "external drafter"
    } else {
        "embedded"
    };
    let n = m
        .get("spec-draft-n-max")
        .map(|n| format!(", n={n}"))
        .unwrap_or_default();
    Some(format!("{ty} ({drafter}{n})"))
}

fn kv_cache_str(m: &Model) -> String {
    match (m.get("cache-type-k"), m.get("cache-type-v")) {
        (Some(k), Some(v)) if k == v => k.to_string(),
        (Some(k), Some(v)) => format!("{k} / {v}"),
        (Some(k), None) => k.to_string(),
        _ => "f16 (default)".into(),
    }
}

fn zero_stats() -> db::ModelStats {
    db::ModelStats {
        launches: 0,
        last_used: None,
        tg_avg: None,
        tg_min: None,
        tg_max: None,
        tg_n: 0,
        pp_avg: None,
        pp_n: 0,
    }
}

fn build_views(models: &[Model], conn: Option<&Connection>) -> Vec<ModelView> {
    let now = db::now();
    let mut views: Vec<ModelView> = models
        .iter()
        .map(|m| {
            let path = m.model_path();
            let path_str = path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "(no model path)".into());
            let size_opt = path.as_ref().and_then(|p| meta::file_size(p));
            let missing = path.is_none() || size_opt.is_none();
            let s = conn
                .and_then(|c| db::stats_for(c, &m.stats_key).ok())
                .unwrap_or_else(zero_stats);
            let spark = conn
                .map(|c| db::recent_tg(c, &m.stats_key, 32))
                .unwrap_or_default()
                .iter()
                .map(|x| x.round().max(0.0) as u64)
                .collect();
            let benchmark = conn
                .map(|c| db::benchmark_scores_for(c, &m.stats_key))
                .unwrap_or(db::BenchmarkScores {
                    chat: None,
                    code: None,
                });
            ModelView {
                alias: m.alias.clone(),
                desc: m.desc.clone(),
                configured: m.configured,
                path: path_str,
                missing,
                size: size_opt.map(meta::human_size).unwrap_or_else(|| "—".into()),
                quant: path
                    .as_ref()
                    .map(|p| meta::detect_quant(p))
                    .unwrap_or_else(|| "?".into()),
                ctx: m.ctx_size().map(format_ctx).unwrap_or_else(|| "?".into()),
                offload: offload_str(m),
                kv_cache: kv_cache_str(m),
                spec: spec_str(m),
                launches: s.launches,
                last_used: meta::relative_time(s.last_used, now),
                tg_avg: s.tg_avg,
                tg_min: s.tg_min,
                tg_max: s.tg_max,
                tg_n: s.tg_n,
                pp_avg: s.pp_avg,
                pp_n: s.pp_n,
                chat_score: benchmark.chat,
                code_score: benchmark.code,
                scores: score::Scores::default(),
                spark,
            }
        })
        .collect();

    // The usage score is relative to the busiest model, so scores can only be
    // filled in once every launch count is known.
    let fleet_max_launches = views.iter().map(|v| v.launches).max().unwrap_or(0);
    for v in &mut views {
        v.scores = score::compute(
            v.tg_avg,
            v.pp_avg,
            v.chat_score,
            v.code_score,
            v.launches,
            fleet_max_launches,
        );
    }
    views
}

fn load_models() -> Result<Vec<Model>, String> {
    let configured = config::parse(&config::default_ini_path())?;
    config::merge_discovered(configured, &config::default_models_dir())
}

// ---------------------------------------------------------------------------
// interactive flow
// ---------------------------------------------------------------------------

/// What the TUI decided to do, executed after the terminal is restored.
enum Decision {
    Quit,
    Stop,
    Router,
    Vllm(usize),
    Launch { mode: String, index: usize },
}

fn interactive() -> i32 {
    let models = match load_models() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("llama-choose: {e}");
            return 1;
        }
    };

    let (llama_pids, vllm_pids) = running_pids();
    let running = !llama_pids.is_empty() || !vllm_pids.is_empty();

    // Stats are decoration; a broken stats DB must never block the picker.
    let conn = db::open().ok();
    let views = build_views(&models, conn.as_ref());

    let mut term = ratatui::init();
    let decision = run_tui(&mut term, running, &views);
    ratatui::restore();

    let decision = match decision {
        Ok(d) => d,
        Err(e) => {
            eprintln!("llama-choose: tui error: {e}");
            return 1;
        }
    };

    match decision {
        Decision::Quit => 0,
        Decision::Stop => stop_servers(&llama_pids, &vllm_pids),
        Decision::Router => launch_router(),
        Decision::Vllm(i) => launch_vllm(i),
        Decision::Launch { mode, index } => launch_model(&models[index], &mode, conn.as_ref()),
    }
}

fn run_tui(
    term: &mut ratatui::DefaultTerminal,
    running: bool,
    views: &[ModelView],
) -> std::io::Result<Decision> {
    // If anything is already on the GPU, the only sane move is to stop it.
    if running {
        let actions = vec![(
            "stop".to_string(),
            "Stop the running inference server".to_string(),
        )];
        return Ok(
            match tui::pick_action(term, "something is running", &actions)? {
                Some(_) => Decision::Stop,
                None => Decision::Quit,
            },
        );
    }

    let actions = vec![
        (
            "tools".to_string(),
            "Start llama-server with built-in tools (127.0.0.1:3002)".to_string(),
        ),
        (
            "server".to_string(),
            "Start llama-server (127.0.0.1:3002)".to_string(),
        ),
        (
            "router".to_string(),
            "Start llama-server router mode (all models, built-in tools)".to_string(),
        ),
        ("vllm".to_string(), "Start vLLM OpenAI server".to_string()),
        (
            "phone".to_string(),
            "Start llama-server on Tailscale".to_string(),
        ),
        (
            "shared".to_string(),
            "Start llama-server on Tailscale for 2 users".to_string(),
        ),
        ("cli".to_string(), "Start interactive llama-cli".to_string()),
        (
            "stats".to_string(),
            "Full-screen stats & scores dashboard".to_string(),
        ),
    ];

    // Looped so "stats" can pop up the dashboard and drop the user right back
    // at the action menu instead of ending the picker session.
    loop {
        let Some(ai) = tui::pick_action(term, "llama-choose — action", &actions)? else {
            return Ok(Decision::Quit);
        };
        let mode = actions[ai].0.clone();

        match mode.as_str() {
            "stats" => {
                tui::show_stats(term, views)?;
                continue;
            }
            "router" => return Ok(Decision::Router),
            "vllm" => {
                let items: Vec<(String, String)> = VLLM_MODELS
                    .iter()
                    .map(|(k, d)| (k.to_string(), d.to_string()))
                    .collect();
                return Ok(match tui::pick_action(term, "vLLM model", &items)? {
                    Some(i) => Decision::Vllm(i),
                    None => Decision::Quit,
                });
            }
            _ => {
                // tools / server / phone / shared / cli all pick a single GGUF model.
                return Ok(match tui::pick_model(term, views)? {
                    Some(index) => Decision::Launch { mode, index },
                    None => Decision::Quit,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// launching
// ---------------------------------------------------------------------------

fn tailscale_ip() -> Option<String> {
    Command::new("tailscale")
        .args(["ip", "-4"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .map(|s| s.trim().to_string())
        })
        .filter(|s| !s.is_empty())
}

const GGUF_HEADER_LEN: u64 = 24;
const MIN_GGUF_VERSION: u32 = 2;
const MAX_GGUF_VERSION: u32 = 3;
const MAX_SPLIT_SHARDS: u32 = 1000;

fn read_gguf_header(path: &std::path::Path) -> Result<(), String> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| format!("cannot stat GGUF file {}: {e}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "GGUF path is not a regular file: {}",
            path.display()
        ));
    }
    if metadata.len() < GGUF_HEADER_LEN {
        return Err(format!(
            "truncated GGUF header ({} bytes): {}",
            metadata.len(),
            path.display()
        ));
    }
    if metadata.len() == GGUF_HEADER_LEN {
        return Err(format!(
            "GGUF file is header-only ({} bytes): {}",
            metadata.len(),
            path.display()
        ));
    }

    let mut header = [0u8; GGUF_HEADER_LEN as usize];
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("cannot open GGUF file {}: {e}", path.display()))?;
    file.read_exact(&mut header)
        .map_err(|e| format!("truncated GGUF header in {}: {e}", path.display()))?;
    if &header[..4] != b"GGUF" {
        return Err(format!("invalid GGUF magic: {}", path.display()));
    }

    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if !(MIN_GGUF_VERSION..=MAX_GGUF_VERSION).contains(&version) {
        return Err(format!(
            "unsupported GGUF version {version} (llama.cpp supports v{MIN_GGUF_VERSION}-v{MAX_GGUF_VERSION}) in {}",
            path.display()
        ));
    }
    let tensors = i64::from_le_bytes(header[8..16].try_into().unwrap());
    if tensors <= 0 {
        return Err(format!("GGUF has no tensors: {}", path.display()));
    }
    let n_kv = i64::from_le_bytes(header[16..24].try_into().unwrap());
    if n_kv < 0 {
        return Err(format!(
            "GGUF has an invalid metadata count in {}",
            path.display()
        ));
    }
    let minimum_len = 24u128 + tensors as u128 * 24 + n_kv as u128 * 13;
    if minimum_len > metadata.len() as u128 {
        return Err(format!(
            "GGUF file is too short for its header counts ({} bytes; at least {minimum_len}): {}",
            metadata.len(),
            path.display()
        ));
    }
    Ok(())
}

/// Return every expected shard for a filename such as `model-00001-of-00002.gguf`.
fn split_shards(path: &std::path::Path) -> Result<Vec<std::path::PathBuf>, String> {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let Some((before_total, total_text)) = stem.rsplit_once("-of-") else {
        return Ok(vec![path.to_path_buf()]);
    };
    let Some((prefix, shard_text)) = before_total.rsplit_once('-') else {
        return Ok(vec![path.to_path_buf()]);
    };
    if shard_text.len() != 5
        || total_text.len() != 5
        || !shard_text.bytes().all(|b| b.is_ascii_digit())
        || !total_text.bytes().all(|b| b.is_ascii_digit())
    {
        return Ok(vec![path.to_path_buf()]);
    }
    let shard: u32 = shard_text.parse().unwrap();
    let total: u32 = total_text.parse().unwrap();
    if total == 0 || shard == 0 || shard > total {
        return Err(format!("invalid GGUF shard numbering: {}", path.display()));
    }
    if total > MAX_SPLIT_SHARDS {
        return Err(format!(
            "GGUF shard count {total} is unreasonable: {}",
            path.display()
        ));
    }

    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("gguf");
    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    Ok((1..=total)
        .map(|n| parent.join(format!("{prefix}-{n:05}-of-{total:05}.{extension}")))
        .collect())
}

fn validate_gguf(key: &str, path: &str) -> Result<(), String> {
    let path = std::path::Path::new(path);
    for shard in split_shards(path)? {
        if !shard.is_file() {
            return Err(format!(
                "missing {key} shard (expected {}): {}",
                shard.display(),
                path.display()
            ));
        }
        read_gguf_header(&shard)?;
    }
    Ok(())
}

fn validate_template(path: &str) -> Result<(), String> {
    let path = std::path::Path::new(path);
    let metadata = std::fs::metadata(path)
        .map_err(|e| format!("cannot stat chat-template-file {}: {e}", path.display()))?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(format!(
            "chat-template-file is missing or empty: {}",
            path.display()
        ));
    }
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("cannot read chat-template-file {}: {e}", path.display()))?;
    let mut byte = [0u8; 1];
    file.read_exact(&mut byte)
        .map_err(|e| format!("cannot read chat-template-file {}: {e}", path.display()))?;
    Ok(())
}

fn artifact_paths(m: &Model) -> impl Iterator<Item = (&'static str, &str)> {
    ["model", "model-draft", "mmproj", "chat-template-file"]
        .into_iter()
        .filter_map(|key| m.get(key).map(|path| (key, path)))
}

/// Confirm model files, split siblings, and optional template are usable before
/// spawning. This is a bounded header check; it does not hash or fully parse GGUF.
fn require_files(m: &Model) -> Result<(), String> {
    let Some(model) = m.get("model") else {
        return Err(format!("missing model path for alias {}", m.alias));
    };
    for (key, path) in artifact_paths(m) {
        if key == "chat-template-file" {
            validate_template(path)?;
        } else {
            validate_gguf(key, path)?;
        }
    }
    // Keep the mandatory model check explicit even if the artifact iterator is
    // changed later.
    if model.is_empty() {
        return Err(format!("empty model path for alias {}", m.alias));
    }
    Ok(())
}

fn launch_model(m: &Model, mode: &str, conn: Option<&Connection>) -> i32 {
    if let Err(e) = require_files(m) {
        eprintln!("llama-choose: {e}");
        return 1;
    }

    if mode == "cli" {
        let lib_dir = format!("{}/repos/llama.cpp/build/bin", home());
        let (env, cmd) = launch::build_cli_command(m, &lib_dir);
        if let Some(c) = conn {
            let _ = db::record_launch(c, &m.stats_key, mode);
        }
        launch::exec_replace(&env, &cmd); // never returns
    }

    let (host, parallel) = match mode {
        "tools" | "server" => ("127.0.0.1".to_string(), 1),
        "phone" | "shared" => match tailscale_ip() {
            Some(ip) => (ip, if mode == "shared" { 2 } else { 1 }),
            None => {
                eprintln!("llama-choose: could not determine Tailscale IPv4 for {mode} mode");
                return 1;
            }
        },
        other => {
            eprintln!("llama-choose: unknown server mode '{other}'");
            return 1;
        }
    };

    let cmd = launch::build_server_command(m, &host, SERVER_PORT, parallel, mode == "tools");
    if let Some(c) = conn {
        let _ = db::record_launch(c, &m.stats_key, mode);
    }
    launch::run_with_scrape(&cmd, &m.stats_key)
}

fn launch_router() -> i32 {
    let ini = config::default_ini_path();
    let cmd = launch::build_router_command(&ini.display().to_string(), "127.0.0.1", SERVER_PORT, 1);
    launch::exec_replace(&[], &cmd);
}

fn launch_vllm(index: usize) -> i32 {
    let (key, _) = VLLM_MODELS[index];
    let hf_home = format!("{}/models/huggingface", home());
    let _ = std::fs::create_dir_all(&hf_home);
    let vllm_bin = format!("{}/repos/vllm/.venv/bin/vllm", home());

    let mut cmd = vec![vllm_bin, "serve".into()];

    let (hf_model, served, max_len, tool_args): (&str, &str, &str, Vec<&str>) = match key {
        "qwen3-0.6b" => (
            "Qwen/Qwen3-0.6B",
            "qwen3-0.6b",
            "8192",
            vec![
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_xml",
            ],
        ),
        "qwen2.5-1.5b-instruct" => (
            "Qwen/Qwen2.5-1.5B-Instruct",
            "qwen2.5-1.5b-instruct",
            "16384",
            vec![],
        ),
        "qwen2.5-7b-instruct-awq" => (
            "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "qwen2.5-7b-instruct",
            "8192",
            vec![],
        ),
        other => {
            eprintln!("llama-choose: unknown vLLM model '{other}'");
            return 1;
        }
    };

    cmd.push(hf_model.into());
    for a in [
        "--host",
        "127.0.0.1",
        "--port",
        &VLLM_PORT.to_string(),
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        "0.80",
        "--max-num-seqs",
        "1",
        "--served-model-name",
        served,
        "--max-model-len",
        max_len,
    ] {
        cmd.push(a.to_string());
    }
    for a in tool_args {
        cmd.push(a.to_string());
    }

    launch::exec_replace(&[("HF_HOME".to_string(), hf_home)], &cmd);
}

// ---------------------------------------------------------------------------
// stop
// ---------------------------------------------------------------------------

fn stop_servers(llama_pids: &[String], vllm_pids: &[String]) -> i32 {
    let all: Vec<&String> = llama_pids.iter().chain(vllm_pids.iter()).collect();
    if all.is_empty() {
        println!("no inference server running");
        return 0;
    }
    let pids: Vec<String> = all.iter().map(|s| s.to_string()).collect();
    println!("+ kill {}", pids.join(" "));
    let status = Command::new("kill").args(&pids).status();
    match status {
        Ok(s) if s.success() => {
            println!("inference server stopped");
            0
        }
        _ => {
            eprintln!("llama-choose: failed to stop one or more processes");
            1
        }
    }
}

fn cmd_stop() -> i32 {
    let (llama_pids, vllm_pids) = running_pids();
    stop_servers(&llama_pids, &vllm_pids)
}

// ---------------------------------------------------------------------------
// non-interactive stats table (the cull dashboard)
// ---------------------------------------------------------------------------

fn cmd_stats() -> i32 {
    let models = match load_models() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("llama-choose: {e}");
            return 1;
        }
    };
    let conn = match db::open() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("llama-choose: {e}");
            return 1;
        }
    };

    let mut views = build_views(&models, Some(&conn));
    // Best overall score first; scoreless and flunking models sink to the
    // bottom as cull bait.
    views.sort_by(|a, b| {
        b.scores
            .overall
            .unwrap_or(-1.0)
            .total_cmp(&a.scores.overall.unwrap_or(-1.0))
            .then_with(|| b.launches.cmp(&a.launches))
            .then_with(|| a.alias.cmp(&b.alias))
    });

    println!(
        "{:<24} {:<6} {:>5}  {:<9} {:>8} {:>8}  {:>4} {:>4}  {:>3} {:>3} {:>3} {:>3} {:>3}  {:<8}  VERDICT",
        "MODEL",
        "SOURCE",
        "RUNS",
        "LAST",
        "GEN t/s",
        "PP t/s",
        "CHAT",
        "CODE",
        "OUT",
        "IN",
        "IQ",
        "USE",
        "ALL",
        "SIZE"
    );
    println!("{}", "─".repeat(126));
    for v in &views {
        let gen = v
            .tg_avg
            .map(|x| format!("{x:.1}"))
            .unwrap_or_else(|| "—".into());
        let pp = v
            .pp_avg
            .map(|x| format!("{x:.0}"))
            .unwrap_or_else(|| "—".into());
        let (verdict, _) = tui::verdict(v);
        let size = if v.missing {
            "MISSING".to_string()
        } else {
            v.size.clone()
        };
        println!(
            "{:<24} {:<6} {:>5}  {:<9} {:>8} {:>8}  {:>4} {:>4}  {:>3} {:>3} {:>3} {:>3} {:>3}  {:<8}  {}",
            meta::truncate(&v.alias, 24),
            if v.configured { "INI" } else { "disk" },
            v.launches,
            v.last_used,
            gen,
            pp,
            tui::fmt_pts(v.chat_score),
            tui::fmt_pts(v.code_score),
            tui::fmt_pts(v.scores.output),
            tui::fmt_pts(v.scores.input),
            tui::fmt_pts(v.scores.intelligence),
            tui::fmt_pts(v.scores.usage),
            tui::fmt_pts(v.scores.overall),
            size,
            verdict
        );
    }
    0
}

/// Check configured model artifacts without launching or hashing them.
fn cmd_check(alias: Option<&String>) -> i32 {
    let models = match config::parse(&config::default_ini_path()) {
        Ok(models) => models,
        Err(e) => {
            eprintln!("llama-choose: {e}");
            return 1;
        }
    };
    let selected: Vec<&Model> = if let Some(alias) = alias {
        match models.iter().find(|m| &m.alias == alias) {
            Some(model) => vec![model],
            None => {
                eprintln!("llama-choose: unknown configured model alias '{alias}'");
                return 1;
            }
        }
    } else {
        models.iter().collect()
    };

    let mut failed = 0u32;
    println!(
        "Checking {} configured model(s) (metadata/header only)",
        selected.len()
    );
    for model in &selected {
        let result = require_files(model);
        println!(
            "\n{}: {}",
            model.alias,
            if result.is_ok() { "OK" } else { "FAIL" }
        );
        for (key, path) in artifact_paths(model) {
            println!("  {key}: {path}");
        }
        if let Err(error) = result {
            failed += 1;
            println!("  error: {error}");
        }
    }
    println!(
        "\nSummary: {} passed, {} failed; no checksum or full tensor validation performed",
        selected.len() as u32 - failed,
        failed
    );
    if failed == 0 {
        0
    } else {
        1
    }
}

/// Launch one model by alias without the TUI, e.g. from a script or tmux.
/// Uses the exact same launch path as the picker, so stats still accrue.
fn cmd_launch(alias: Option<&String>, mode: Option<&String>) -> i32 {
    let Some(alias) = alias else {
        eprintln!("usage: llama-choose launch ALIAS [tools|server|phone|shared|cli]");
        return 2;
    };
    let mode = mode.map(String::as_str).unwrap_or("tools");
    if !matches!(mode, "tools" | "server" | "phone" | "shared" | "cli") {
        eprintln!("llama-choose: launch mode must be tools, server, phone, shared, or cli");
        return 2;
    }

    let models = match load_models() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("llama-choose: {e}");
            return 1;
        }
    };
    let Some(model) = models.iter().find(|m| &m.alias == alias) else {
        eprintln!("llama-choose: unknown model alias '{alias}'");
        return 1;
    };

    let (llama_pids, vllm_pids) = running_pids();
    if !llama_pids.is_empty() || !vllm_pids.is_empty() {
        eprintln!(
            "llama-choose: an inference server is already running ('llama-choose stop' first)"
        );
        return 1;
    }

    let conn = db::open().ok();
    launch_model(model, mode, conn.as_ref())
}

fn cmd_bench(alias: Option<&String>, suite: Option<&String>) -> i32 {
    let Some(alias) = alias else {
        eprintln!("usage: llama-choose bench ALIAS|all [chat|code]");
        return 2;
    };
    let models = match load_models() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("llama-choose: {e}");
            return 1;
        }
    };
    let selected: Vec<&Model> = if alias == "all" {
        models.iter().filter(|m| m.configured).collect()
    } else {
        let Some(model) = models.iter().find(|m| &m.alias == alias) else {
            eprintln!("llama-choose: unknown model alias '{alias}'");
            return 1;
        };
        if !model.configured {
            eprintln!("llama-choose: benchmark requires a configured INI alias");
            return 1;
        }
        vec![model]
    };

    let suites = match suite.map(String::as_str) {
        None => vec![benchmark::Suite::Chat, benchmark::Suite::Code],
        Some(name) => match benchmark::Suite::parse(name) {
            Some(s) => vec![s],
            None => {
                eprintln!("llama-choose: benchmark suite must be 'chat' or 'code'");
                return 2;
            }
        },
    };
    let conn = match db::open() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("llama-choose: {e}");
            return 1;
        }
    };

    for model in selected {
        for &suite in &suites {
            println!("{} {} benchmark:", model.alias, suite.kind());
            let previous = db::latest_benchmark(&conn, &model.stats_key, suite.kind());
            let result = match benchmark::run(&model.alias, suite) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("llama-choose: {e}");
                    eprintln!("start llama-server router mode before benchmarking");
                    return 1;
                }
            };
            if let Err(e) = db::record_benchmark(
                &conn,
                &model.stats_key,
                suite.kind(),
                result.score,
                result.passed,
                result.total,
            ) {
                eprintln!("llama-choose: cannot save benchmark: {e}");
                return 1;
            }
            let was = previous
                .map(|p| format!(", was {p:.0}"))
                .unwrap_or_default();
            println!(
                "  score: {:.0}/100 ({}/{}{was})",
                result.score, result.passed, result.total
            );
        }
    }
    0
}

#[cfg(test)]
mod preflight_tests {
    use super::*;
    use std::path::Path;

    fn temp_root(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "llama-choose-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn write_header(path: &Path, len: usize) {
        let mut bytes = vec![0u8; len];
        if len >= GGUF_HEADER_LEN as usize {
            bytes[..4].copy_from_slice(b"GGUF");
            bytes[4..8].copy_from_slice(&3u32.to_le_bytes());
            bytes[8..16].copy_from_slice(&1u64.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn write_minimal_gguf(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one tensor
        bytes.extend_from_slice(&0u64.to_le_bytes()); // no metadata
        bytes.extend_from_slice(&1u64.to_le_bytes()); // name length
        bytes.push(b'x');
        bytes.extend_from_slice(&1u32.to_le_bytes()); // one dimension
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one element
        bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor data offset
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&0f32.to_le_bytes());
        std::fs::write(path, bytes).unwrap();
    }

    fn model(path: &Path, extras: &[(&str, &str)]) -> Model {
        let mut kv = vec![("model".into(), path.display().to_string())];
        kv.extend(
            extras
                .iter()
                .map(|(key, value)| ((*key).into(), (*value).into())),
        );
        Model {
            alias: "test-model".into(),
            kv,
            desc: None,
            configured: true,
            stats_key: "test-model".into(),
        }
    }

    #[test]
    fn rejects_truncated_and_header_only_gguf() {
        let root = temp_root("truncated");
        std::fs::create_dir_all(&root).unwrap();
        let truncated = root.join("truncated.gguf");
        write_header(&truncated, 4);
        assert!(require_files(&model(&truncated, &[]))
            .unwrap_err()
            .contains("truncated GGUF header"));

        let header_only = root.join("header-only.gguf");
        write_header(&header_only, GGUF_HEADER_LEN as usize);
        assert!(require_files(&model(&header_only, &[]))
            .unwrap_err()
            .contains("header-only"));

        let header_plus_one = root.join("header-plus-one.gguf");
        write_header(&header_plus_one, GGUF_HEADER_LEN as usize + 1);
        assert!(require_files(&model(&header_plus_one, &[]))
            .unwrap_err()
            .contains("too short for its header counts"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_invalid_magic_and_missing_shard() {
        let root = temp_root("shards");
        std::fs::create_dir_all(&root).unwrap();
        let invalid = root.join("invalid.gguf");
        write_header(&invalid, 25);
        std::fs::write(&invalid, vec![0u8; 25]).unwrap();
        assert!(require_files(&model(&invalid, &[]))
            .unwrap_err()
            .contains("invalid GGUF magic"));

        let first = root.join("split-00001-of-00002.gguf");
        write_minimal_gguf(&first);
        assert!(require_files(&model(&first, &[]))
            .unwrap_err()
            .contains("missing model shard"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn accepts_valid_gguf_and_rejects_missing_template() {
        let root = temp_root("template");
        std::fs::create_dir_all(&root).unwrap();
        let model_path = root.join("existing-model.gguf");
        write_minimal_gguf(&model_path);
        assert!(require_files(&model(&model_path, &[])).is_ok());

        let missing_template = root.join("missing.jinja");
        assert!(require_files(&model(
            &model_path,
            &[("chat-template-file", missing_template.to_str().unwrap())]
        ))
        .unwrap_err()
        .contains("chat-template-file"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
