use crate::state::Note;
use gtk4::cairo;

pub(crate) const PIXELS_PER_MS: f64 = 0.08;

pub(crate) fn scroll_for_playhead(playhead_x: f64, viewport_width: f64) -> f64 {
    (playhead_x - viewport_width * 0.2).max(0.0)
}

pub(crate) fn timeline_content_width(start_ms: u64, end_ms: u64, viewport_width: i32) -> i32 {
    let timeline_width = end_ms.saturating_sub(start_ms) as f64 * PIXELS_PER_MS;
    (timeline_width + viewport_width as f64 * 0.8).ceil() as i32
}

pub(crate) fn playhead_overlay_x(notes: &[Note], playhead_ms: u64, viewport_width: i32) -> f64 {
    let start = timeline_bounds(notes, Some(playhead_ms)).map_or(0, |bounds| bounds.0);
    let timeline_x = playhead_ms.saturating_sub(start) as f64 * PIXELS_PER_MS;
    timeline_x.min(viewport_width as f64 * 0.2)
}

pub(crate) fn timeline_bounds(notes: &[Note], playhead: Option<u64>) -> Option<(u64, u64)> {
    let start = notes.iter().map(|note| note.onset_ms).min().or(playhead)?;
    let end = notes
        .iter()
        .map(|note| note.onset_ms + note.duration_ms)
        .chain(playhead)
        .max()
        .unwrap_or(start);
    Some((start, end))
}

pub(crate) fn draw_roll(
    cr: &cairo::Context,
    width: i32,
    height: i32,
    notes: &[Note],
    playhead: Option<u64>,
) {
    cr.set_source_rgb(0.07, 0.08, 0.10);
    let _ = cr.paint();
    let start = timeline_bounds(notes, playhead).map_or(0, |bounds| bounds.0);
    for pitch in 21..=108 {
        let y = height as f64 * (108 - pitch) as f64 / 88.0;
        cr.set_source_rgba(1.0, 1.0, 1.0, if pitch % 12 == 0 { 0.10 } else { 0.025 });
        cr.rectangle(0.0, y, width as f64, 1.0);
        let _ = cr.fill();
    }
    for note in notes {
        let x = note.onset_ms.saturating_sub(start) as f64 * PIXELS_PER_MS;
        let w = (note.duration_ms as f64 * PIXELS_PER_MS).max(2.0);
        let y = height as f64 * (108_i32 - note.pitch as i32) as f64 / 88.0;
        if note.generated {
            cr.set_source_rgb(0.25, 0.80, 0.48);
        } else {
            cr.set_source_rgb(0.32, 0.58, 0.95);
        }
        cr.rectangle(x, y, w, (height as f64 / 88.0).max(3.0));
        let _ = cr.fill();
    }
}

pub(crate) fn draw_playhead(
    cr: &cairo::Context,
    width: i32,
    height: i32,
    notes: &[Note],
    playhead: Option<u64>,
) {
    if let Some(playhead) = playhead {
        let x = playhead_overlay_x(notes, playhead, width);
        cr.set_source_rgb(0.95, 0.18, 0.20);
        cr.rectangle(x, 0.0, 2.0, height as f64);
        let _ = cr.fill();
    }
}
