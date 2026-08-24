use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub(crate) struct AppConfig {
    pub(crate) midi_input: Option<String>,
    pub(crate) midi_output: Option<String>,
    pub(crate) soundfont: Option<String>,
}

pub(crate) fn config_path() -> PathBuf {
    if let Some(path) = std::env::var_os("XDG_CONFIG_HOME").filter(|path| !path.is_empty()) {
        return PathBuf::from(path).join("midi-autocomplete/config.toml");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".config/midi-autocomplete/config.toml");
    }
    PathBuf::from("config.toml")
}

pub(crate) fn load_config(path: &Path) -> Result<AppConfig, String> {
    match fs::read_to_string(path) {
        Ok(contents) => {
            toml::from_str(&contents).map_err(|error| format!("Invalid config: {error}"))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(AppConfig::default()),
        Err(error) => Err(format!("Could not read config: {error}")),
    }
}

pub(crate) fn save_config(path: &Path, config: &AppConfig) -> Result<(), String> {
    let contents = toml::to_string_pretty(config).map_err(|error| error.to_string())?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Could not create config directory: {error}"))?;
    }
    let temporary = path.with_extension("toml.tmp");
    fs::write(&temporary, contents).map_err(|error| format!("Could not write config: {error}"))?;
    fs::rename(&temporary, path).map_err(|error| format!("Could not save config: {error}"))
}
