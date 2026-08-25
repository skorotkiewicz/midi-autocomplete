use crate::state::Note;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

pub(crate) struct GenerationRequest {
    pub(crate) checkpoint: PathBuf,
    pub(crate) prompt: String,
    pub(crate) bpm: f64,
    pub(crate) musical_start_ms: Option<u64>,
    pub(crate) revision: u64,
    pub(crate) auto_play: bool,
    pub(crate) soundfont: Option<PathBuf>,
}

pub(crate) struct ModelProcess {
    pub(crate) checkpoint: PathBuf,
    _child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl ModelProcess {
    pub(crate) fn start(checkpoint: &Path) -> Result<Self, String> {
        let root = repo_root();
        let checkpoint = resolve_model_path(checkpoint);
        let checkpoint = if is_remote_url(&checkpoint) {
            checkpoint
        } else {
            checkpoint
                .canonicalize()
                .map_err(|error| format!("Model not found: {error}"))?
        };
        let mut child = Command::new("uv")
            .args([
                "run",
                "--directory",
                &root.join("midilm").to_string_lossy(),
                "--extra",
                "cpu",
                "midilm",
                "serve",
            ])
            .arg(&checkpoint)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("Could not start midilm: {error}"))?;
        let input = child.stdin.take().ok_or("Could not open model input")?;
        let mut output = BufReader::new(child.stdout.take().ok_or("Could not open model output")?);
        let mut ready = String::new();
        output
            .read_line(&mut ready)
            .map_err(|error| error.to_string())?;
        if !ready.starts_with("ready\t") {
            return Err(format!("Model failed to start: {}", ready.trim()));
        }
        Ok(Self {
            checkpoint,
            _child: child,
            input,
            output,
        })
    }

    pub(crate) fn generate(&mut self, prompt: &str) -> Result<Vec<(u8, u64, u64, u8)>, String> {
        writeln!(self.input, "generate\t32\t0.9\t16\t{prompt}")
            .map_err(|error| error.to_string())?;
        self.input.flush().map_err(|error| error.to_string())?;
        let mut response = String::new();
        self.output
            .read_line(&mut response)
            .map_err(|error| error.to_string())?;
        let (kind, body) = response
            .trim_end()
            .split_once('\t')
            .unwrap_or(("error", "empty response"));
        if kind != "notes" {
            return Err(body.to_string());
        }
        parse_generated_notes(body)
    }
}

pub(crate) fn parse_generated_notes(body: &str) -> Result<Vec<(u8, u64, u64, u8)>, String> {
    if body.is_empty() {
        return Ok(Vec::new());
    }
    body.split(';')
        .map(|note| {
            let values = note
                .split(',')
                .map(str::parse::<u64>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| error.to_string())?;
            if values.len() != 4 {
                return Err("invalid model note".to_string());
            }
            let pitch = u8::try_from(values[0])
                .ok()
                .filter(|value| *value <= 127)
                .ok_or("model pitch is outside 0..127")?;
            let velocity = u8::try_from(values[3])
                .ok()
                .filter(|value| *value <= 127)
                .ok_or("model velocity is outside 0..127")?;
            Ok((pitch, values[1], values[2], velocity))
        })
        .collect()
}

/// Absolute path to the repo root, so `uv --directory midilm` and model paths
/// work no matter what directory the process was launched from.
pub(crate) fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn is_remote_url(path: &Path) -> bool {
    let text = path.to_string_lossy();
    text.starts_with("https://") || text.starts_with("http://")
}

/// Resolve a checkpoint path for launching the model worker. Remote Hugging
/// Face URLs pass through unchanged; local paths are made absolute against the
/// repo root when relative, then canonicalized when the file exists.
pub(crate) fn resolve_model_path(path: &Path) -> PathBuf {
    if is_remote_url(path) {
        return path.to_path_buf();
    }
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root().join(path)
    };
    joined.canonicalize().unwrap_or(joined)
}

pub(crate) fn prompt(notes: &[Note], bpm: f64, musical_end_ms: Option<u64>) -> String {
    let mut notes: Vec<_> = notes
        .iter()
        .filter(|note| !note.generated && musical_end_ms.is_none_or(|end| note.onset_ms <= end))
        .copied()
        .collect();
    notes.sort_by_key(|note| (note.onset_ms, note.pitch));
    let step_ms = 60_000.0 / bpm / 24.0;
    let mut previous = None;
    notes
        .into_iter()
        .map(|note| {
            let delta = previous.map_or(0, |onset| note.onset_ms.saturating_sub(onset));
            previous = Some(note.onset_ms);
            format!(
                "{},{},{},{}",
                note.pitch,
                (delta as f64 / step_ms).round() as u64,
                (note.duration_ms as f64 / step_ms).round().max(1.0) as u64,
                note.velocity
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}
