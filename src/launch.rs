//! Building and running the actual inference commands.
//!
//! Single-model server modes (`server`/`phone`/`shared`) are *spawned* with
//! their stderr piped through us so we can scrape `tokens per second` lines
//! into the stats DB while echoing everything to the terminal untouched.
//! The other modes (`router`/`cli`/`vllm`) need no per-model scraping, so we
//! `exec()` into them and hand the process table straight over, exactly like
//! the old bash `exec`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::config::Model;
use crate::db;

fn ui_config_file() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    format!("{home}/.config/llama.cpp/ui-config.json")
}

/// Resolve a llama.cpp binary: prefer the home repo build (which is not on
/// PATH; its RPATH already covers the sibling shared libs), fall back to the
/// bare name for a regular PATH lookup.
pub fn resolve_bin(name: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    let candidate = format!("{home}/repos/llama.cpp/build/bin/{name}");
    if std::path::Path::new(&candidate).is_file() {
        candidate
    } else {
        name.to_string()
    }
}

/// Translate INI key/values into `llama` long-option flags.
///
/// `key = value` becomes `--key value`; the boolean form `key = true`
/// becomes a bare `--key` flag (e.g. `jinja = true` -> `--jinja`).
pub fn build_load_args(model: &Model) -> Vec<String> {
    let mut args = Vec::new();
    for (k, v) in &model.kv {
        if v == "true" {
            args.push(format!("--{k}"));
        } else if v == "false" {
            // A llama.cpp boolean explicitly disabled: drop it, there is no
            // generic negative form we can rely on across options.
            continue;
        } else {
            args.push(format!("--{k}"));
            args.push(v.clone());
        }
    }
    args
}

/// `llama-server` command for a single model in a server-style mode.
///
/// The tools mode enables every built-in llama.cpp tool, as does router mode.
/// Ordinary server launches do not enable these tools.
pub fn build_server_command(
    model: &Model,
    host: &str,
    port: u16,
    parallel: u32,
    tools: bool,
) -> Vec<String> {
    let mut cmd = vec![resolve_bin("llama-server")];
    cmd.extend(build_load_args(model));
    if tools {
        cmd.extend(["--tools".into(), "all".into()]);
    }
    cmd.extend([
        "--alias".into(),
        model.alias.clone(),
        "--host".into(),
        host.to_string(),
        "--port".into(),
        port.to_string(),
        "--timeout".into(),
        "600".into(),
        "--parallel".into(),
        parallel.to_string(),
        "--ui".into(),
        "--ui-mcp-proxy".into(),
        "--ui-config-file".into(),
        ui_config_file(),
    ]);
    cmd
}

/// `llama-server` router mode: serve every model with built-in tools.
/// llama.cpp overlays these router options onto each child model's preset.
pub fn build_router_command(ini_path: &str, host: &str, port: u16, parallel: u32) -> Vec<String> {
    vec![
        resolve_bin("llama-server"),
        "--models-preset".into(),
        ini_path.to_string(),
        "--host".into(),
        host.to_string(),
        "--port".into(),
        port.to_string(),
        "--timeout".into(),
        "600".into(),
        "--ui".into(),
        "--ui-mcp-proxy".into(),
        "--ui-config-file".into(),
        ui_config_file(),
        "--parallel".into(),
        parallel.to_string(),
        "--tools".into(),
        "all".into(),
    ]
}

/// Interactive `llama-cli` for a single model. Returns (env, command).
pub fn build_cli_command(model: &Model, lib_dir: &str) -> (Vec<(String, String)>, Vec<String>) {
    let ld = match std::env::var("LD_LIBRARY_PATH") {
        Ok(existing) if !existing.is_empty() => format!("{lib_dir}:{existing}"),
        _ => lib_dir.to_string(),
    };
    let mut cmd = vec![resolve_bin("llama-cli")];
    cmd.extend(build_load_args(model));
    cmd.extend(["--conversation".into(), "--color".into(), "auto".into()]);
    (vec![("LD_LIBRARY_PATH".into(), ld)], cmd)
}

/// Shell-quote a single argument for the echoed `+ ...` command line.
fn shell_quote(s: &str) -> String {
    if !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"@%-_=+:,./".contains(&b))
    {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

/// Echo the command the way the old script did: `+ prog arg arg`.
pub fn echo_command(env: &[(String, String)], cmd: &[String]) {
    let mut line = String::from("+");
    for (k, v) in env {
        line.push(' ');
        line.push_str(&format!("{k}={}", shell_quote(v)));
    }
    for part in cmd {
        line.push(' ');
        line.push_str(&shell_quote(part));
    }
    eprintln!("{line}");
}

/// Replace this process with `cmd` (router / cli / vllm). Never returns on success.
pub fn exec_replace(env: &[(String, String)], cmd: &[String]) -> ! {
    echo_command(env, cmd);
    let mut c = Command::new(&cmd[0]);
    c.args(&cmd[1..]);
    for (k, v) in env {
        c.env(k, v);
    }
    let err = c.exec(); // only returns on failure
    eprintln!("llama-choose: failed to exec {}: {err}", cmd[0]);
    std::process::exit(127);
}

/// How long a throwaway preset-check router gets to come up.
const PRESET_CHECK_TIMEOUT: Duration = Duration::from_secs(30);

/// Kills and reaps the throwaway router however the check ends.
struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Validate the preset file the way router mode will read it.
///
/// llama.cpp's preset parser is stricter than the CLI and aborts the whole
/// router on the first unknown key, so a llama.cpp update can break router
/// mode while every model file still looks fine. This starts the exact router
/// command on a free loopback port with `--no-models-autoload` (no weights
/// are loaded), waits for `/models`, checks that every `expected` alias is
/// listed, and always kills the router. Returns the number of listed models.
pub fn validate_router_preset(ini_path: &str, expected: &[&str]) -> Result<usize, String> {
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map_err(|e| format!("cannot reserve a loopback port: {e}"))?
        .port();
    let mut cmd = build_router_command(ini_path, "127.0.0.1", port, 1);
    cmd.push("--no-models-autoload".into());

    let mut child = Command::new(&cmd[0])
        .args(&cmd[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run {}: {e}", cmd[0]))?;
    let stderr = child.stderr.take().expect("stderr is piped");
    let mut guard = ChildGuard(child);

    // Drain stderr on a thread so the router never blocks on a full pipe.
    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = Arc::clone(&log);
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            sink.lock().unwrap().push(line);
        }
    });

    let started = Instant::now();
    loop {
        if let Ok(Some(status)) = guard.0.try_wait() {
            // Give the reader a moment to collect the final error line.
            thread::sleep(Duration::from_millis(100));
            let lines = log.lock().unwrap();
            return Err(format!(
                "router exited ({status}): {}",
                router_error_summary(&lines)
            ));
        }
        if let Some(body) = http_get(port, "/models") {
            let ids = parse_model_ids(&body)?;
            let missing: Vec<&str> = expected
                .iter()
                .copied()
                .filter(|alias| !ids.iter().any(|id| id == alias))
                .collect();
            if !missing.is_empty() {
                return Err(format!("router does not list: {}", missing.join(", ")));
            }
            return Ok(ids.len());
        }
        if started.elapsed() > PRESET_CHECK_TIMEOUT {
            return Err(format!(
                "router did not answer /models within {}s",
                PRESET_CHECK_TIMEOUT.as_secs()
            ));
        }
        thread::sleep(Duration::from_millis(250));
    }
}

/// Minimal HTTP GET against the loopback router; `Some(body)` only on 200.
fn http_get(port: u16, path: &str) -> Option<String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    let (head, body) = response.split_once("\r\n\r\n")?;
    let status_ok = head
        .lines()
        .next()
        .is_some_and(|l| l.split_whitespace().nth(1) == Some("200"));
    status_ok.then(|| body.to_string())
}

/// Model ids from a router `/models` response body.
fn parse_model_ids(body: &str) -> Result<Vec<String>, String> {
    let json: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("bad /models response: {e}"))?;
    Ok(json["data"]
        .as_array()
        .map(|models| {
            models
                .iter()
                .filter_map(|m| m["id"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default())
}

/// The most useful line of a failed router's stderr: its last error line,
/// else its last line.
fn router_error_summary(lines: &[String]) -> String {
    lines
        .iter()
        .rev()
        .find(|l| l.contains(" E ") || l.to_lowercase().contains("error"))
        .or_else(|| lines.last())
        // Drop the `0.00.238.189 E ` timestamp/level prefix when present.
        .map(|l| {
            l.split_once(" E ")
                .map_or(l.as_str(), |(_, msg)| msg)
                .trim()
                .to_string()
        })
        .unwrap_or_else(|| "no output".into())
}

/// Pull the `tokens per second` value out of a llama-server timing line.
///
/// Lines look like:
///   `prompt eval time =  500.00 ms / 50 tokens ( 10.00 ms per token, 100.00 tokens per second)`
///   `       eval time = 2000.00 ms / 100 tokens ( 20.00 ms per token,  50.00 tokens per second)`
/// We return (kind, tok_per_s, n_tokens) where kind is "pp" or "tg".
fn parse_timing_line(line: &str) -> Option<(&'static str, f64, i64)> {
    if !line.contains("tokens per second") {
        return None;
    }
    let kind = if line.contains("prompt eval time") {
        "pp"
    } else if line.contains("eval time") {
        "tg"
    } else {
        return None;
    };

    // tok/s: the number immediately before "tokens per second".
    let idx = line.find("tokens per second")?;
    let before = line[..idx].trim_end();
    let tok_per_s: f64 = before
        .rsplit(|c: char| c.is_whitespace())
        .next()?
        .parse()
        .ok()?;

    // token count: the integer right before the word "tokens".
    let n_tokens = line
        .find(" tokens (")
        .and_then(|p| line[..p].trim_end().rsplit(char::is_whitespace).next())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    Some((kind, tok_per_s, n_tokens))
}

/// Whether a scraped timing sample is worth recording.
///
/// When a generation produces a single token, llama.cpp's eval time rounds to
/// `0.00 ms` and it prints a sentinel `1000000.00 tokens per second` — a
/// divide-by-near-zero artifact, not a real measurement. A one-token generation
/// has no meaningful throughput anyway, so we drop it; otherwise it blows the
/// average to the moon (a MoE that interleaves short reasoning turns will emit
/// these constantly). Prompt-eval samples are always kept.
fn is_meaningful_sample(kind: &str, tok_per_s: f64, n_tokens: i64) -> bool {
    if !tok_per_s.is_finite() || tok_per_s <= 0.0 {
        return false;
    }
    if kind == "tg" && n_tokens < 2 {
        return false;
    }
    true
}

/// Spawn a server command, echo its stderr through to our own stderr, and
/// scrape timing lines into the stats DB under `alias`. Returns the exit code.
pub fn run_with_scrape(cmd: &[String], alias: &str) -> i32 {
    echo_command(&[], cmd);

    let mut child = match Command::new(&cmd[0])
        .args(&cmd[1..])
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("llama-choose: failed to start {}: {e}", cmd[0]);
            return 127;
        }
    };

    // A dedicated connection for the scrape loop; failure to open just means
    // we lose stats, never the server itself.
    let sample_conn = db::open().ok();

    let stderr = child.stderr.take().expect("piped stderr");
    let reader = BufReader::new(stderr);
    let out = std::io::stderr();
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        // Passthrough first so the user never notices the wrapper.
        {
            let mut h = out.lock();
            let _ = writeln!(h, "{line}");
        }
        if let (Some(conn), Some((kind, tps, n))) = (&sample_conn, parse_timing_line(&line)) {
            if is_meaningful_sample(kind, tps, n) {
                let _ = db::insert_sample(conn, alias, kind, tps, n);
            }
        }
    }

    match child.wait() {
        Ok(status) => status.code().unwrap_or(0),
        Err(_) => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_bin_falls_back_to_path_lookup_when_repo_build_is_absent() {
        // resolve_bin reads HOME; the scrape test below also rewrites HOME,
        // so just point it somewhere without a llama.cpp build.
        std::env::set_var("HOME", std::env::temp_dir());
        assert_eq!(resolve_bin("llama-server"), "llama-server");
    }

    #[test]
    fn parses_generation_line() {
        let l = "       eval time =    2000.00 ms /   100 tokens (   20.00 ms per token,    50.00 tokens per second)";
        let (kind, tps, n) = parse_timing_line(l).unwrap();
        assert_eq!(kind, "tg");
        assert_eq!(n, 100);
        assert!((tps - 50.0).abs() < 1e-6);
    }

    #[test]
    fn parses_prompt_line_with_log_prefix() {
        let l = "slot      release: prompt eval time =     500.00 ms /    50 tokens (   10.00 ms per token,   100.00 tokens per second)";
        let (kind, tps, n) = parse_timing_line(l).unwrap();
        assert_eq!(kind, "pp");
        assert_eq!(n, 50);
        assert!((tps - 100.0).abs() < 1e-6);
    }

    #[test]
    fn ignores_unrelated_lines() {
        assert!(parse_timing_line("main: server is listening on http://127.0.0.1:3002").is_none());
    }

    #[test]
    fn tools_server_and_router_enable_all_builtin_tools() {
        let model = Model {
            alias: "test-model".into(),
            kv: vec![("model".into(), "/tmp/test.gguf".into())],
            desc: None,
            configured: true,
            stats_key: "test-model".into(),
        };
        let tools_cmd = build_server_command(&model, "127.0.0.1", 3002, 1, true);
        assert!(tools_cmd.windows(2).any(|pair| pair == ["--tools", "all"]));

        let router_cmd = build_router_command("/tmp/models.ini", "127.0.0.1", 3099, 1);
        assert!(router_cmd.windows(2).any(|pair| pair == ["--tools", "all"]));

        let plain_cmd = build_server_command(&model, "127.0.0.1", 3002, 1, false);
        assert!(!plain_cmd.iter().any(|arg| arg == "--tools"));
    }

    #[test]
    fn drops_single_token_generation_sentinel() {
        // llama.cpp's 1-token divide-by-zero artifact must not be recorded.
        assert!(!is_meaningful_sample("tg", 1_000_000.0, 1));
        assert!(!is_meaningful_sample("tg", 42.0, 1));
        // Real generation and any prompt-eval sample are kept.
        assert!(is_meaningful_sample("tg", 50.0, 100));
        assert!(is_meaningful_sample("pp", 30.0, 4));
    }

    #[test]
    fn scrapes_fake_server_stderr_into_db() {
        // Isolate the stats DB under a throwaway HOME.
        let tmp = std::env::temp_dir().join(format!("llama-choose-test-{}", std::process::id()));
        std::env::set_var("HOME", &tmp);
        let alias = "fake-model";

        // A stand-in server: emit one real generation, one single-token sentinel
        // (which must be dropped), and one prompt timing line, then exit.
        let script = "printf 'srv: starting\\n' >&2; \
            printf '       eval time =    2000.00 ms /   100 tokens (   20.00 ms per token,    50.00 tokens per second)\\n' >&2; \
            printf '       eval time =       0.00 ms /     1 tokens (    0.00 ms per token, 1000000.00 tokens per second)\\n' >&2; \
            printf 'prompt eval time =     500.00 ms /    50 tokens (   10.00 ms per token,   100.00 tokens per second)\\n' >&2";
        let cmd = vec!["sh".to_string(), "-c".to_string(), script.to_string()];

        let code = run_with_scrape(&cmd, alias);
        assert_eq!(code, 0, "fake server should exit cleanly");

        let conn = db::open().expect("open stats db");
        let s = db::stats_for(&conn, alias).expect("stats");
        assert_eq!(s.tg_n, 1, "sentinel dropped, only the real generation kept");
        assert!((s.tg_avg.unwrap() - 50.0).abs() < 1e-6);
        assert_eq!(s.pp_n, 1, "one prompt sample");
        assert!((s.pp_avg.unwrap() - 100.0).abs() < 1e-6);

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn router_error_summary_prefers_last_error_line_without_prefix() {
        let lines = vec![
            "0.00.149.390 I cmn  common_param: verbosity = 3".to_string(),
            "0.00.238.189 E srv  llama_server: failed to initialize router models: option 'no-mmap' not recognized in preset 'gemma'".to_string(),
            "0.00.240.000 I srv  shutting down".to_string(),
        ];
        assert_eq!(
            router_error_summary(&lines),
            "srv  llama_server: failed to initialize router models: option 'no-mmap' not recognized in preset 'gemma'"
        );
        assert_eq!(
            router_error_summary(&["just this".to_string()]),
            "just this"
        );
        assert_eq!(router_error_summary(&[]), "no output");
    }

    #[test]
    fn parses_router_model_ids() {
        let body =
            r#"{"object":"list","data":[{"id":"a","status":{"value":"unloaded"}},{"id":"b"}]}"#;
        assert_eq!(parse_model_ids(body).unwrap(), ["a", "b"]);
        assert!(parse_model_ids("{}").unwrap().is_empty());
        assert!(parse_model_ids("not json").is_err());
    }
}
