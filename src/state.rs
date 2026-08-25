#[derive(Clone, Copy)]
pub(crate) struct Note {
    pub(crate) pitch: u8,
    pub(crate) onset_ms: u64,
    pub(crate) duration_ms: u64,
    pub(crate) velocity: u8,
    pub(crate) generated: bool,
}

#[derive(Default)]
pub(crate) struct Shared {
    pub(crate) notes: Vec<Note>,
    pub(crate) status: String,
    pub(crate) connected: bool,
    pub(crate) capturing: bool,
    pub(crate) capture_generation: u64,
    pub(crate) capture_started_ms: u64,
    pub(crate) capture_position_ms: u64,
    pub(crate) input_active: bool,
    pub(crate) last_input_ms: u64,
    pub(crate) generation_revision: u64,
}

impl Shared {
    pub(crate) fn invalidate_generation(&mut self) -> u64 {
        self.generation_revision = self.generation_revision.wrapping_add(1);
        self.generation_revision
    }

    pub(crate) fn generation_is_current(&self, revision: u64) -> bool {
        self.generation_revision == revision
    }

    pub(crate) fn add_input_note(&mut self, note: Note) {
        self.notes.push(note);
        self.invalidate_generation();
    }

    pub(crate) fn replace_notes(&mut self, notes: Vec<Note>, now_ms: u64) {
        self.notes = notes;
        self.capture_position_ms = self
            .notes
            .iter()
            .map(|note| note.onset_ms + note.duration_ms)
            .max()
            .unwrap_or(0);
        self.capture_started_ms = now_ms;
        self.capturing = false;
        self.capture_generation = self.capture_generation.wrapping_add(1);
        self.input_active = false;
        self.last_input_ms = now_ms;
        self.invalidate_generation();
    }

    pub(crate) fn capture_position(&self, now_ms: u64) -> u64 {
        self.capture_position_ms
            + if self.capturing {
                now_ms.saturating_sub(self.capture_started_ms)
            } else {
                0
            }
    }

    pub(crate) fn toggle_capture(
        &mut self,
        now_ms: u64,
        selected_position_ms: Option<u64>,
    ) -> bool {
        if self.capturing {
            self.capture_position_ms = self.capture_position(now_ms);
            self.capturing = false;
        } else {
            self.capture_position_ms = selected_position_ms.unwrap_or_else(|| {
                self.capture_position_ms.max(
                    self.notes
                        .iter()
                        .map(|note| note.onset_ms + note.duration_ms)
                        .max()
                        .unwrap_or(0),
                )
            });
            self.capture_started_ms = now_ms;
            self.capturing = true;
        }
        self.capture_generation = self.capture_generation.wrapping_add(1);
        self.invalidate_generation();
        self.input_active = false;
        self.last_input_ms = now_ms;
        self.status = if self.capturing {
            "Recording MIDI prompt...".into()
        } else {
            "Recording stopped.".into()
        };
        self.capturing
    }

    pub(crate) fn clear_timeline(&mut self, now_ms: u64) {
        self.notes.clear();
        self.capture_position_ms = 0;
        self.capture_started_ms = now_ms;
        self.capture_generation = self.capture_generation.wrapping_add(1);
        self.invalidate_generation();
        self.input_active = false;
        self.last_input_ms = now_ms;
    }
}
