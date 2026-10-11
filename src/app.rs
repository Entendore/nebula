use eframe::egui;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use serde::{Serialize, Deserialize};

use crate::ai::{AiClient, AiEvent, extract_code};
use crate::highlight::Highlighter;

struct Buffer {
    code: String,
    path: Option<PathBuf>,
    dirty: bool,
}

impl Buffer {
    fn new(code: String, path: Option<PathBuf>) -> Self {
        Self { code, path, dirty: false }
    }
    fn title(&self) -> String {
        self.path.as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "untitled".to_string())
    }
    fn extension(&self) -> &str {
        self.path.as_ref()
            .and_then(|p| p.extension())
            .and_then(|e| e.to_str())
            .unwrap_or("rs")
    }
}

struct ProposedChange {
    original_text: String,
    new_text: String,
    start_idx: usize,
    end_idx: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct AppConfig {
    model_path: Option<PathBuf>,
    n_gpu_layers: u32,
    temperature: f32,
    max_tokens: usize,
    tab_width: usize,
    use_spaces: bool,
    syntax_theme: String,
    window_width: f32,
    window_height: f32,
    auto_save: bool,
    dark_mode: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            n_gpu_layers: 35,
            temperature: 0.7,
            max_tokens: 1024,
            tab_width: 4,
            use_spaces: true,
            syntax_theme: "base16-ocean.dark".to_string(),
            window_width: 1200.0,
            window_height: 800.0,
            auto_save: false,
            dark_mode: true,
        }
    }
}

impl AppConfig {
    pub fn load() -> Self {
        if let Some(config_dir) = dirs::config_dir() {
            let config_path = config_dir.join("nebula_config.json");
            if let Ok(data) = std::fs::read_to_string(&config_path) {
                if let Ok(cfg) = serde_json::from_str(&data) {
                    return cfg;
                }
            }
        }
        AppConfig::default()
    }
}

pub struct NebulaApp {
    runtime: Arc<Runtime>,
    highlighter: Highlighter,
    ai_client: AiClient,
    
    buffers: Vec<Buffer>,
    active_buffer: usize,
    cursor_index: usize,
    selection_start: usize,
    selection_end: usize,
    
    inline_generating: bool,
    inline_original_code: String,
    
    ai_prompt: String,
    ai_response: String,
    ai_streaming: bool,
    is_loading_model: bool,
    chat_history: Vec<(String, String)>,
    prompt_history: Vec<String>,
    prompt_history_index: Option<usize>,
    is_cancelled: Arc<AtomicBool>,
    proposed_change: Option<ProposedChange>,
    
    ai_tx: mpsc::UnboundedSender<AiEvent>,
    ai_rx: mpsc::UnboundedReceiver<AiEvent>,
    
    show_file_explorer: bool,
    show_ai_panel: bool,
    show_bottom_panel: bool,
    show_command_palette: bool,
    show_settings: bool,
    show_find_replace: bool,
    show_goto_line: bool,
    command_palette_query: String,
    command_palette_index: usize,
    word_wrap: bool,
    
    find_query: String,
    replace_query: String,
    goto_line_str: String,
    
    current_dir: Option<PathBuf>,
    explorer_entries: Vec<PathBuf>,
    
    terminal_input: String,
    terminal_history: Vec<String>,
    terminal_history_index: Option<usize>,
    log_tx: mpsc::UnboundedSender<String>,
    log_rx: mpsc::UnboundedReceiver<String>,
    
    config: AppConfig,
    output_log: Vec<String>,
    ai_prompt_id: egui::Id,
    terminal_input_id: egui::Id,
    code_editor_id: egui::Id,
    last_save_time: Instant,
}

impl NebulaApp {
    pub fn new(cc: &eframe::CreationContext<'_>, runtime: Arc<Runtime>) -> Self {
        let (ai_tx, ai_rx) = mpsc::unbounded_channel::<AiEvent>();
        let (log_tx, log_rx) = mpsc::unbounded_channel::<String>();
        let config = AppConfig::load();
        let mut output_log = vec!["Nebula Editor started.".to_string()];
        
        let mut highlighter = Highlighter::new();
        highlighter.set_theme(&config.syntax_theme);
        
        cc.egui_ctx.set_visuals(if config.dark_mode { egui::Visuals::dark() } else { egui::Visuals::light() });

        if config.model_path.is_some() {
            output_log.push("Found saved model path. Auto-loading...".to_string());
        }

        Self {
            runtime,
            highlighter,
            ai_client: AiClient { 
                model: None, 
                temperature: config.temperature, 
                max_tokens: config.max_tokens 
            },
            buffers: vec![Buffer::new("// Welcome to Nebula Editor\n// Press Up/Down in terminal for history!\n".to_string(), None)],
            active_buffer: 0,
            cursor_index: 0,
            selection_start: 0,
            selection_end: 0,
            inline_generating: false,
            inline_original_code: String::new(),
            ai_prompt: String::new(),
            ai_response: String::new(),
            ai_streaming: false,
            is_loading_model: config.model_path.is_some(),
            chat_history: Vec::new(),
            prompt_history: Vec::new(),
            prompt_history_index: None,
            is_cancelled: Arc::new(AtomicBool::new(false)),
            proposed_change: None,
            ai_tx: ai_tx.clone(),
            ai_rx,
            show_file_explorer: true,
            show_ai_panel: true,
            show_bottom_panel: true,
            show_command_palette: false,
            show_settings: false,
            show_find_replace: false,
            show_goto_line: false,
            command_palette_query: String::new(),
            command_palette_index: 0,
            word_wrap: false,
            find_query: String::new(),
            replace_query: String::new(),
            goto_line_str: String::new(),
            current_dir: None,
            explorer_entries: Vec::new(),
            terminal_input: String::new(),
            terminal_history: Vec::new(),
            terminal_history_index: None,
            log_tx,
            log_rx,
            config,
            output_log,
            ai_prompt_id: egui::Id::new("ai_prompt_input"),
            terminal_input_id: egui::Id::new("terminal_input_field"),
            code_editor_id: egui::Id::new("code_editor_text_edit"),
            last_save_time: Instant::now(),
        }
    }

    fn save_config(&self) {
        if let Some(config_dir) = dirs::config_dir() {
            let config_path = config_dir.join("nebula_config.json");
            if let Ok(data) = serde_json::to_string_pretty(&self.config) {
                let _ = std::fs::write(&config_path, data);
            }
        }
    }

    fn log(&mut self, msg: impl Into<String>) {
        self.output_log.push(msg.into());
    }

    fn run_command(&mut self, cmd_str: String) {
        let dir = self.current_dir.clone().unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
        
        if cmd_str.starts_with("cd ") {
            let target_path = cmd_str[3..].trim();
            let new_dir = if target_path.starts_with('/') || target_path.contains(':') {
                PathBuf::from(target_path)
            } else if target_path == "~" {
                dirs::home_dir().unwrap_or_default()
            } else if target_path.starts_with("~/") {
                dirs::home_dir().unwrap_or_default().join(&target_path[2..])
            } else {
                dir.join(target_path)
            };

            if new_dir.is_dir() {
                self.current_dir = std::fs::canonicalize(&new_dir).ok();
                self.refresh_explorer();
                self.log(format!("Changed directory to {}", self.current_dir.as_ref().unwrap().display()));
            } else {
                self.log(format!("cd: no such directory: {}", target_path));
            }
            return;
        }

        let tx = self.log_tx.clone();
        self.log(format!("$ {}", cmd_str));
        
        self.runtime.spawn(async move {
            let mut parts = cmd_str.split_whitespace();
            let cmd = parts.next().unwrap_or("");
            let args: Vec<&str> = parts.collect();
            
            let result = tokio::process::Command::new(cmd)
                .args(&args)
                .current_dir(&dir)
                .output()
                .await;
                
            match result {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                    if !stdout.is_empty() { let _ = tx.send(stdout); }
                    if !stderr.is_empty() { let _ = tx.send(stderr); }
                }
                Err(e) => {
                    let _ = tx.send(format!("Failed to execute: {}", e));
                }
            }
        });
    }

    fn refresh_explorer(&mut self) {
        if let Some(dir) = &self.current_dir {
            if let Ok(entries) = std::fs::read_dir(dir) {
                let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
                paths.sort_by(|a, b| {
                    let a_is_dir = a.is_dir();
                    let b_is_dir = b.is_dir();
                    b_is_dir.cmp(&a_is_dir).then_with(|| a.file_name().cmp(&b.file_name()))
                });
                self.explorer_entries = paths;
            }
        }
    }

    fn detect_indent(content: &str) -> (usize, bool) {
        for line in content.lines().take(50) {
            if line.starts_with("    ") { return (4, true); }
            if line.starts_with("  ") { return (2, true); }
            if line.starts_with("\t") { return (4, false); }
        }
        (4, true)
    }

    fn open_file(&mut self) {
        if let Some(path) = rfd::FileDialog::new().pick_file() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                let (tw, us) = Self::detect_indent(&content);
                self.config.tab_width = tw;
                self.config.use_spaces = us;
                
                self.buffers.push(Buffer::new(content, Some(path.clone())));
                self.active_buffer = self.buffers.len() - 1;
                self.current_dir = path.parent().map(|p| p.to_path_buf());
                self.refresh_explorer();
                self.log(format!("Opened file: {}", path.display()));
            }
        }
    }

    fn open_specific_file(&mut self, path: PathBuf) {
        if let Ok(content) = std::fs::read_to_string(&path) {
            let (tw, us) = Self::detect_indent(&content);
            self.config.tab_width = tw;
            self.config.use_spaces = us;
            
            self.buffers.push(Buffer::new(content, Some(path.clone())));
            self.active_buffer = self.buffers.len() - 1;
            self.log(format!("Opened file: {}", path.display()));
        }
    }

    fn save_file(&mut self) {
        if self.buffers.is_empty() { return; }
        let buf = &mut self.buffers[self.active_buffer];
        if let Some(path) = &buf.path {
            let _ = std::fs::write(path, &buf.code);
            buf.dirty = false;
        } else if let Some(path) = rfd::FileDialog::new().save_file() {
            let _ = std::fs::write(&path, &buf.code);
            buf.path = Some(path.clone());
            buf.dirty = false;
        }
    }

    fn close_tab(&mut self, index: usize) {
        if index >= self.buffers.len() { return; }
        self.buffers.remove(index);
        if !self.buffers.is_empty() && self.active_buffer >= self.buffers.len() {
            self.active_buffer = self.buffers.len() - 1;
        } else if self.buffers.is_empty() {
            self.buffers.push(Buffer::new(String::new(), None));
            self.active_buffer = 0;
        }
    }

    fn load_model(&mut self, path: PathBuf) {
        self.is_loading_model = true;
        self.config.model_path = Some(path.clone());
        self.save_config();
        let tx = self.ai_tx.clone();
        AiClient::spawn_load_model(path, self.config.n_gpu_layers, tx);
    }

    fn trigger_inline_ai(&mut self) {
        if self.inline_generating || self.ai_streaming || !self.ai_client.is_loaded() { return; }
        self.inline_generating = true;
        self.inline_original_code = self.buffers[self.active_buffer].code.clone();
        let code_before = self.inline_original_code[..self.cursor_index.min(self.inline_original_code.len())].to_string();
        
        let is_cancelled = Arc::new(AtomicBool::new(false));
        self.is_cancelled = is_cancelled.clone();
        let tx = self.ai_tx.clone();
        let client = self.ai_client.clone();
        
        self.runtime.spawn(async move {
            let _ = client.stream_inline_completion(&code_before, tx, is_cancelled).await;
        });
    }

    fn send_chat_request(&mut self, mut prompt: String) {
        if prompt.is_empty() || self.ai_streaming || self.inline_generating || !self.ai_client.is_loaded() { return; }

        if prompt.starts_with('/') {
            let parts: Vec<&str> = prompt.splitn(2, ' ').collect();
            prompt = match parts[0] {
                "/explain" => "Explain this code.".to_string(),
                "/fix" => "Find and fix any bugs in this code.".to_string(),
                "/refactor" => "Refactor this code to be more idiomatic.".to_string(),
                "/tests" => "Write unit tests for this code.".to_string(),
                _ => prompt,
            };
        }

        self.ai_streaming = true;
        self.ai_response.clear();
        
        let code_context = if self.selection_start != self.selection_end {
            self.buffers[self.active_buffer].code[self.selection_start..self.selection_end].to_string()
        } else {
            self.buffers[self.active_buffer].code.clone()
        };
        
        let history = self.chat_history.clone();
        let is_cancelled = Arc::new(AtomicBool::new(false));
        self.is_cancelled = is_cancelled.clone();
        let tx = self.ai_tx.clone();
        let client = self.ai_client.clone();
        
        self.runtime.spawn(async move {
            let _ = client.stream_chat_completion(&prompt, &code_context, &history, tx, is_cancelled).await;
        });

        self.prompt_history.push(prompt.clone());
        self.prompt_history_index = None;
        self.chat_history.push(("user".to_string(), prompt));
        self.ai_prompt.clear();
    }

    fn prepare_diff_review(&mut self) {
        if self.ai_response.is_empty() { return; }
        let new_code = extract_code(&self.ai_response);
        if new_code.is_empty() { return; }

        let buf = &self.buffers[self.active_buffer];
        let (start, end, original) = if self.selection_start != self.selection_end {
            (self.selection_start, self.selection_end, buf.code[self.selection_start..self.selection_end].to_string())
        } else {
            (self.cursor_index, self.cursor_index, String::new())
        };

        self.proposed_change = Some(ProposedChange {
            original_text: original,
            new_text: new_code,
            start_idx: start,
            end_idx: end,
        });
    }

    fn accept_diff(&mut self) {
        if let Some(change) = self.proposed_change.take() {
            let buf = &mut self.buffers[self.active_buffer];
            buf.code.replace_range(change.start_idx..change.end_idx, &change.new_text);
            buf.dirty = true;
            self.cursor_index = change.start_idx + change.new_text.len();
            self.chat_history.push(("assistant".to_string(), self.ai_response.clone()));
            self.ai_response.clear();
            self.log("Applied AI changes.");
        }
    }

    fn reject_diff(&mut self) {
        self.proposed_change = None;
        self.log("Rejected AI changes.");
    }

    fn get_comment_syntax(&self, ext: &str) -> String {
        match ext {
            "py" | "sh" | "bash" | "yaml" | "yml" | "toml" | "rb" | "perl" => "# ".to_string(),
            "html" | "xml" => "<!-- ".to_string(),
            "css" => "/* ".to_string(),
            _ => "// ".to_string(),
        }
    }

    fn toggle_comments(&mut self) {
        let ext = self.buffers[self.active_buffer].extension().to_string();
        let comment_str = self.get_comment_syntax(&ext);
        
        let code = &mut self.buffers[self.active_buffer].code;
        let start = self.selection_start.min(self.selection_end);
        let end = self.selection_start.max(self.selection_end);
        
        let line_start = code[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let line_end = code[end..].find('\n').map(|i| end + i).unwrap_or(code.len());
        
        let block = &code[line_start..line_end];
        let all_commented = block.lines().all(|l| l.trim_start().starts_with(comment_str.trim_end()));
        
        let new_block = if all_commented {
            block.lines().map(|l| {
                let trimmed = l.trim_start();
                let leading_ws = &l[..l.len() - trimmed.len()];
                if trimmed.starts_with(comment_str.trim_end()) {
                    format!("{}{}", leading_ws, &trimmed[comment_str.trim_end().len()..])
                } else { l.to_string() }
            }).collect::<Vec<_>>().join("\n")
        } else {
            block.lines().map(|l| format!("{}{}", comment_str, l)).collect::<Vec<_>>().join("\n")
        };
        
        code.replace_range(line_start..line_end, &new_block);
        self.buffers[self.active_buffer].dirty = true;
    }

    fn indent_block(&mut self) {
        let code = &mut self.buffers[self.active_buffer].code;
        let start = self.selection_start.min(self.selection_end);
        let end = self.selection_start.max(self.selection_end);
        let line_start = code[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let line_end = code[end..].find('\n').map(|i| end + i).unwrap_or(code.len());
        
        let block = &code[line_start..line_end];
        let indent_str = if self.config.use_spaces { " ".repeat(self.config.tab_width) } else { "\t".to_string() };
        
        let new_block: String = block.lines().map(|line| format!("{}{}", indent_str, line)).collect::<Vec<_>>().join("\n");
        code.replace_range(line_start..line_end, &new_block);
        
        self.selection_start = line_start;
        self.selection_end = line_start + new_block.len();
        self.cursor_index = self.selection_end;
        self.buffers[self.active_buffer].dirty = true;
    }

    fn unindent_block(&mut self) {
        let code = &mut self.buffers[self.active_buffer].code;
        let start = self.selection_start.min(self.selection_end);
        let end = self.selection_start.max(self.selection_end);
        let line_start = code[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let line_end = code[end..].find('\n').map(|i| end + i).unwrap_or(code.len());
        
        let block = &code[line_start..line_end];
        let indent_str = if self.config.use_spaces { " ".repeat(self.config.tab_width) } else { "\t".to_string() };
        
        let new_block: String = block.lines().map(|line| {
            if line.starts_with(&indent_str) { line[indent_str.len()..].to_string() }
            else if line.starts_with('\t') { line[1..].to_string() }
            else { line.to_string() }
        }).collect::<Vec<_>>().join("\n");
        
        code.replace_range(line_start..line_end, &new_block);
        
        self.selection_start = line_start;
        self.selection_end = line_start + new_block.len();
        self.cursor_index = self.selection_end;
        self.buffers[self.active_buffer].dirty = true;
    }

    fn duplicate_line(&mut self) {
        let code = &mut self.buffers[self.active_buffer].code;
        let start = self.selection_start.min(self.selection_end);
        let end = self.selection_start.max(self.selection_end);
        let line_start = code[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let line_end = code[end..].find('\n').map(|i| end + i + 1).unwrap_or(code.len());
        
        let lines_to_dup = code[line_start..line_end].to_string();
        code.insert_str(line_end, &lines_to_dup);
        
        self.cursor_index = line_end + lines_to_dup.len() - 1;
        self.selection_start = self.cursor_index;
        self.selection_end = self.cursor_index;
        self.buffers[self.active_buffer].dirty = true;
    }

    fn find_next(&mut self) {
        if self.find_query.is_empty() { return; }
        let code = &self.buffers[self.active_buffer].code;
        let start_idx = self.cursor_index + 1;
        
        if let Some(pos) = code[start_idx..].find(&self.find_query) {
            let abs_pos = start_idx + pos;
            self.selection_start = abs_pos;
            self.selection_end = abs_pos + self.find_query.len();
            self.cursor_index = self.selection_end;
        } else if let Some(pos) = code.find(&self.find_query) {
            self.selection_start = pos;
            self.selection_end = pos + self.find_query.len();
            self.cursor_index = self.selection_end;
        }
    }

    fn replace_current(&mut self) {
        if self.selection_start != self.selection_end {
            let selected = self.buffers[self.active_buffer].code[self.selection_start..self.selection_end].to_string();
            if selected == self.find_query {
                let buf = &mut self.buffers[self.active_buffer];
                buf.code.replace_range(self.selection_start..self.selection_end, &self.replace_query);
                buf.dirty = true;
                self.cursor_index = self.selection_start + self.replace_query.len();
            }
        }
        self.find_next();
    }

    fn replace_all(&mut self) {
        if self.find_query.is_empty() { return; }
        let buf = &mut self.buffers[self.active_buffer];
        buf.code = buf.code.replace(&self.find_query, &self.replace_query);
        buf.dirty = true;
    }

    fn go_to_line(&mut self, ctx: &egui::Context) {
        if let Ok(line_num) = self.goto_line_str.trim().parse::<usize>() {
            let line_idx = line_num.saturating_sub(1);
            let char_idx = self.buffers[self.active_buffer].code.lines().take(line_idx).map(|l| l.len() + 1).sum::<usize>();
            
            if let Some(mut state) = egui::TextEdit::load_state(ctx, self.code_editor_id) {
                state.cursor.set_char_range(Some(egui::text::CursorRange {
                    primary: egui::text::CCursor::new(char_idx).into(),
                    secondary: egui::text::CCursor::new(char_idx).into(),
                }));
                state.store(ctx, self.code_editor_id);
            }
            self.cursor_index = char_idx;
        }
        self.show_goto_line = false;
        self.goto_line_str.clear();
    }

    fn poll_ai_events(&mut self) {
        while let Ok(event) = self.ai_rx.try_recv() {
            match event {
                AiEvent::Chunk(chunk) => self.ai_response.push_str(&chunk),
                AiEvent::InlineChunk(chunk) => {
                    if self.inline_generating {
                        let idx = self.cursor_index.min(self.buffers[self.active_buffer].code.len());
                        self.buffers[self.active_buffer].code.insert_str(idx, &chunk);
                        self.cursor_index += chunk.len();
                        self.buffers[self.active_buffer].dirty = true;
                    }
                }
                AiEvent::Done => {
                    self.ai_streaming = false;
                    self.inline_generating = false;
                    if !self.ai_response.is_empty() {
                        self.chat_history.push(("assistant".to_string(), self.ai_response.clone()));
                    }
                }
                AiEvent::Error(err) => {
                    self.ai_response = format!("Error: {}", err);
                    self.ai_streaming = false;
                    self.inline_generating = false;
                    self.is_loading_model = false;
                    self.log(format!("AI Error: {}", err));
                }
                AiEvent::ModelLoaded(model) => {
                    self.ai_client.set_model(model);
                    self.is_loading_model = false;
                    self.log("Model loaded successfully. AI Ready.");
                }
            }
        }
        
        while let Ok(log) = self.log_rx.try_recv() {
            for line in log.lines() {
                self.output_log.push(line.to_string());
            }
        }
    }
    
    fn render_chat_markdown(&self, ui: &mut egui::Ui, text: &str, fallback_ext: &str) {
        let mut in_code_block = false;
        let mut code_buffer = String::new();
        let mut current_lang = fallback_ext.to_string();

        for line in text.lines() {
            if line.starts_with("```") {
                if in_code_block {
                    let job = self.highlighter.highlight(&code_buffer, &current_lang);
                    ui.add(egui::TextBufferLabel::new(job).background_color(egui::Color32::from_rgb(30, 30, 30)));
                    code_buffer.clear();
                    in_code_block = false;
                } else {
                    in_code_block = true;
                    let lang = line.trim_start_matches("```").trim();
                    current_lang = if !lang.is_empty() { lang.to_string() } else { fallback_ext.to_string() };
                }
                continue;
            }
            if in_code_block {
                code_buffer.push_str(line);
                code_buffer.push('\n');
            } else {
                ui.label(line);
            }
        }
        if !code_buffer.is_empty() {
            let job = self.highlighter.highlight(&code_buffer, &current_lang);
            ui.add(egui::TextBufferLabel::new(job).background_color(egui::Color32::from_rgb(30, 30, 30)));
        }
    }

    fn show_command_palette(&mut self, ctx: &egui::Context) {
        let mut open = true;
        egui::Window::new("Command Palette")
            .open(&mut open)
            .resizable(false)
            .anchor(egui::Align2::CENTER_TOP, [0.0, 50.0])
            .show(ctx, |ui| {
                ui.set_width(400.0);
                ui.text_edit_singleline(&mut self.command_palette_query);
                ui.separator();
                
                let commands = [
                    "Open File", "Save File", "Close Tab", "Find & Replace", "Go to Line",
                    "Toggle Explorer", "Toggle AI Panel", "Toggle Bottom Panel", "Toggle UI Theme",
                    "Settings", "Word Wrap Toggle",
                    "Explain Selected Code", "Refactor Selected Code", "Fix Bugs"
                ];
                
                let filtered: Vec<&str> = commands.iter()
                    .filter(|c| c.to_lowercase().contains(&self.command_palette_query.to_lowercase()))
                    .cloned()
                    .collect();

                if self.command_palette_index >= filtered.len() {
                    self.command_palette_index = 0;
                }

                egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                    if ui.input(|i| i.key_pressed(egui::Key::ArrowDown)) && !filtered.is_empty() {
                        self.command_palette_index = (self.command_palette_index + 1) % filtered.len();
                    }
                    if ui.input(|i| i.key_pressed(egui::Key::ArrowUp)) && !filtered.is_empty() {
                        self.command_palette_index = self.command_palette_index.saturating_sub(1);
                    }
                    if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        if let Some(cmd) = filtered.get(self.command_palette_index) {
                            self.execute_command(cmd);
                            self.show_command_palette = false;
                            self.command_palette_query.clear();
                        }
                    }

                    for (i, cmd) in filtered.iter().enumerate() {
                        let selected = i == self.command_palette_index;
                        if ui.selectable_label(selected, *cmd).clicked() {
                            self.execute_command(cmd);
                            self.show_command_palette = false;
                            self.command_palette_query.clear();
                        }
                    }
                });
            });
        if !open { self.show_command_palette = false; }
    }

    fn execute_command(&mut self, cmd: &str) {
        match cmd {
            "Open File" => self.open_file(),
            "Save File" => self.save_file(),
            "Close Tab" => self.close_tab(self.active_buffer),
            "Find & Replace" => self.show_find_replace = true,
            "Go to Line" => self.show_goto_line = true,
            "Toggle Explorer" => self.show_file_explorer = !self.show_file_explorer,
            "Toggle AI Panel" => self.show_ai_panel = !self.show_ai_panel,
            "Toggle Bottom Panel" => self.show_bottom_panel = !self.show_bottom_panel,
            "Toggle UI Theme" => {
                self.config.dark_mode = !self.config.dark_mode;
                self.save_config();
            },
            "Word Wrap Toggle" => self.word_wrap = !self.word_wrap,
            "Settings" => self.show_settings = true,
            "Explain Selected Code" => self.send_chat_request("Explain this code.".to_string()),
            "Refactor Selected Code" => self.send_chat_request("Refactor this code.".to_string()),
            "Fix Bugs" => self.send_chat_request("Find and fix any bugs in this code.".to_string()),
            _ => {}
        }
    }

    fn show_settings(&mut self, ctx: &egui::Context) {
        let mut open = true;
        let mut temp_theme = self.config.syntax_theme.clone();
        
        egui::Window::new("Settings")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("GPU Layers:");
                    ui.add(egui::DragValue::new(&mut self.config.n_gpu_layers).range(0..=100));
                });
                ui.horizontal(|ui| {
                    ui.label("Temperature:");
                    ui.add(egui::Slider::new(&mut self.ai_client.temperature, 0.0..=1.5));
                });
                ui.horizontal(|ui| {
                    ui.label("Max Tokens:");
                    ui.add(egui::DragValue::new(&mut self.ai_client.max_tokens).range(128..=4096));
                });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Tab Width:");
                    ui.add(egui::DragValue::new(&mut self.config.tab_width).range(1..=8));
                    ui.checkbox(&mut self.config.use_spaces, "Use Spaces");
                });
                ui.horizontal(|ui| {
                    ui.label("Syntax Theme:");
                    egui::ComboBox::from_label("")
                        .selected_text(&temp_theme)
                        .show_ui(ui, |ui| {
                            for theme in self.highlighter.get_themes() {
                                ui.selectable_value(&mut temp_theme, theme.clone(), theme);
                            }
                        });
                });
                ui.checkbox(&mut self.config.auto_save, "Auto Save (every 5s)");
                ui.checkbox(&mut self.config.dark_mode, "Dark UI Mode");
                ui.separator();
                if ui.button("Save Settings").clicked() {
                    self.config.temperature = self.ai_client.temperature;
                    self.config.max_tokens = self.ai_client.max_tokens;
                    self.config.syntax_theme = temp_theme.clone();
                    self.highlighter.set_theme(&temp_theme);
                    self.save_config();
                    self.show_settings = false;
                    self.log("Settings saved.");
                }
            });
        if !open { self.show_settings = false; }
    }
}

impl eframe::App for NebulaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_ai_events();

        ctx.set_visuals(if self.config.dark_mode { egui::Visuals::dark() } else { egui::Visuals::light() });

        let dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped_files.is_empty() {
            if let Some(file) = dropped_files.first() {
                if let Some(path) = &file.path {
                    if let Ok(content) = std::fs::read_to_string(path) {
                        let (tw, us) = Self::detect_indent(&content);
                        self.config.tab_width = tw;
                        self.config.use_spaces = us;
                        
                        self.buffers.push(Buffer::new(content, Some(path.clone())));
                        self.active_buffer = self.buffers.len() - 1;
                        self.current_dir = path.parent().map(|p| p.to_path_buf());
                        self.refresh_explorer();
                        self.log(format!("Opened dropped file: {}", path.display()));
                    }
                }
            }
        }

        let win_size = ctx.input(|i| i.viewport().inner_size.unwrap_or(egui::vec2(1200.0, 800.0)));
        if win_size.x > 100.0 && win_size.y > 100.0 {
            self.config.window_width = win_size.x;
            self.config.window_height = win_size.y;
        }

        if self.config.auto_save && self.last_save_time.elapsed() > Duration::from_secs(5) {
            for buf in &mut self.buffers {
                if buf.dirty {
                    if let Some(path) = &buf.path {
                        let _ = std::fs::write(path, &buf.code);
                        buf.dirty = false;
                        self.log(format!("Auto-saved: {}", path.display()));
                    }
                }
            }
            self.last_save_time = Instant::now();
        }

        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::P)) { self.show_command_palette = !self.show_command_palette; }
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::W)) { self.close_tab(self.active_buffer); }
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::L)) { ctx.memory_mut(|m| m.request_focus(self.ai_prompt_id)); }
        if ctx.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::Z)) { self.word_wrap = !self.word_wrap; }
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::F)) { self.show_find_replace = !self.show_find_replace; }
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::G)) { self.show_goto_line = !self.show_goto_line; }

        if self.is_loading_model && self.ai_client.is_loaded() == false && self.config.model_path.is_some() && self.output_log.len() == 2 {
            if let Some(path) = self.config.model_path.clone() {
                let tx = self.ai_tx.clone();
                AiClient::spawn_load_model(path, self.config.n_gpu_layers, tx);
            }
        }

        if self.ai_streaming || self.is_loading_model || self.inline_generating {
            ctx.request_repaint();
        }

        if self.show_command_palette { self.show_command_palette(ctx); }
        if self.show_settings { self.show_settings(ctx); }
        
        if self.proposed_change.is_some() {
            let mut open = true;
            let mut accept = false;
            let mut reject = false;
            
            egui::Window::new("AI Diff Review")
                .open(&mut open)
                .resizable(true)
                .default_width(600.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("✅ Accept").clicked() { accept = true; }
                        if ui.button("❌ Reject").clicked() { reject = true; }
                    });
                    ui.separator();
                    
                    let ext = self.buffers[self.active_buffer].extension().to_string();
                    
                    ui.columns(2, |columns| {
                        columns[0].heading("❌ Original");
                        let orig_job = self.highlighter.highlight(&self.proposed_change.as_ref().unwrap().original_text, &ext);
                        columns[0].add(egui::TextBufferLabel::new(orig_job).background_color(egui::Color32::from_rgb(40, 20, 20)));
                        
                        columns[1].heading("✅ Proposed");
                        let new_job = self.highlighter.highlight(&self.proposed_change.as_ref().unwrap().new_text, &ext);
                        columns[1].add(egui::TextBufferLabel::new(new_job).background_color(egui::Color32::from_rgb(20, 40, 20)));
                    });
                });
                
            if accept { self.accept_diff(); }
            if reject || !open { self.reject_diff(); }
        }

        if self.show_goto_line {
            let mut open = true;
            egui::Window::new("Go to Line")
                .open(&mut open)
                .resizable(false)
                .anchor(egui::Align2::CENTER_TOP, [0.0, 50.0])
                .show(ctx, |ui| {
                    ui.set_width(200.0);
                    ui.text_edit_singleline(&mut self.goto_line_str);
                    if ui.input(|i| i.key_pressed(egui::Key::Enter)) { self.go_to_line(ctx); }
                });
            if !open { self.show_goto_line = false; }
        }

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open File... (Ctrl+O)").clicked() { self.open_file(); }
                    if ui.button("Save File... (Ctrl+S)").clicked() { self.save_file(); }
                    if ui.button("Close Tab (Ctrl+W)").clicked() { self.close_tab(self.active_buffer); }
                    if ui.button("Quit").clicked() { self.save_config(); std::process::exit(0); }
                });
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.show_file_explorer, "Show Explorer");
                    ui.checkbox(&mut self.show_ai_panel, "Show AI Panel");
                    ui.checkbox(&mut self.show_bottom_panel, "Show Output Panel");
                    if ui.button("Toggle UI Theme").clicked() { self.config.dark_mode = !self.config.dark_mode; self.save_config(); }
                    if ui.button("Settings").clicked() { self.show_settings = true; }
                });
            });
        });

        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let mut close_idx = None;
                for (i, buf) in self.buffers.iter().enumerate() {
                    let title = format!("{}{}", if buf.dirty { "* " } else { "" }, buf.title());
                    ui.horizontal(|ui| {
                        if ui.selectable_label(i == self.active_buffer, title).clicked() { self.active_buffer = i; }
                        if ui.small_button("×").clicked() { close_idx = Some(i); }
                    });
                    ui.separator();
                }
                if let Some(idx) = close_idx { self.close_tab(idx); }
            });
        });

        if self.show_bottom_panel {
            egui::TopBottomPanel::bottom("output_panel")
                .resizable(true)
                .default_height(150.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("Terminal / Output");
                        if ui.button("Clear").clicked() { self.output_log.clear(); }
                    });
                    ui.separator();
                    egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false; 2]).show(ui, |ui| {
                        for line in &self.output_log { ui.label(line); }
                    });
                    ui.separator();
                    
                    ui.horizontal(|ui| {
                        ui.label("$");
                        let resp = ui.add(egui::TextEdit::singleline(&mut self.terminal_input).id(self.terminal_input_id).desired_width(f32::MAX));
                        
                        if ui.memory(|m| m.has_focus(self.terminal_input_id)) {
                            if ui.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                                if !self.terminal_history.is_empty() {
                                    let new_idx = match self.terminal_history_index {
                                        Some(idx) => if idx == 0 { 0 } else { idx - 1 },
                                        None => self.terminal_history.len() - 1,
                                    };
                                    self.terminal_history_index = Some(new_idx);
                                    self.terminal_input = self.terminal_history[new_idx].clone();
                                }
                            }
                            if ui.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                                if let Some(idx) = self.terminal_history_index {
                                    if idx == self.terminal_history.len() - 1 {
                                        self.terminal_history_index = None;
                                        self.terminal_input.clear();
                                    } else {
                                        let new_idx = idx + 1;
                                        self.terminal_history_index = Some(new_idx);
                                        self.terminal_input = self.terminal_history[new_idx].clone();
                                    }
                                }
                            }
                        }

                        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            let cmd = self.terminal_input.clone();
                            self.terminal_input.clear();
                            self.terminal_history_index = None;
                            if !cmd.is_empty() {
                                self.terminal_history.push(cmd.clone());
                                self.run_command(cmd);
                            }
                            ctx.memory_mut(|m| m.request_focus(self.terminal_input_id));
                        }
                    });
                });
        }

        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(path) = &self.buffers.get(self.active_buffer).and_then(|b| b.path.clone()) {
                    ui.label(format!("📄 {}", path.display()));
                } else { ui.label("📄 Untitled"); }
                ui.separator();
                if self.inline_generating { ui.label("🤖 AI Autocompleting... (Esc to cancel)"); }
                else if self.is_loading_model { ui.label("⏳ Loading AI Model..."); }
                else if self.ai_streaming { ui.label("🤖 AI Generating..."); }
                else if self.ai_client.is_loaded() { ui.label("🤖 AI Ready"); }
                else { ui.colored_label(egui::Color32::RED, "🤖 AI Offline"); }
                
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let code = &self.buffers[self.active_buffer].code;
                    let lines = code.lines().count();
                    let words = code.split_whitespace().count();
                    let chars = code.chars().count();
                    
                    let before_cursor = &code[..self.cursor_index.min(code.len())];
                    let ln = before_cursor.lines().count() + 1;
                    let col = before_cursor.rfind('\n').map(|i| self.cursor_index - i).unwrap_or(self.cursor_index) + 1;
                    
                    ui.label(format!("Ln {}, Col {} | {}L | {}W | {}C", ln, col, lines, words, chars));
                    ui.separator();

                    if self.selection_start != self.selection_end {
                        let len = self.selection_end - self.selection_start;
                        ui.label(format!("Sel: {} chars", len));
                        ui.separator();
                    }
                    let lang = self.buffers[self.active_buffer].extension().to_uppercase();
                    let indent = if self.config.use_spaces { format!("Spaces: {}", self.config.tab_width) } else { "Tabs".to_string() };
                    ui.label(format!("{} | {}", lang, indent));
                });
            });
        });

        if self.show_file_explorer {
            egui::SidePanel::left("file_explorer")
                .default_width(200.0)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("Explorer");
                        if ui.button("Open File...").clicked() { self.open_file(); }
                        if ui.button("🔄").clicked() { self.refresh_explorer(); }
                    });
                    ui.separator();
                    
                    if let Some(dir) = &self.current_dir {
                        ui.label(format!("📁 {}", dir.display()));
                        if ui.button("⬆ Up").clicked() {
                            if let Some(parent) = dir.parent() {
                                self.current_dir = Some(parent.to_path_buf());
                                self.refresh_explorer();
                            }
                        }
                        ui.separator();
                        
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for entry in &self.explorer_entries.clone() {
                                let name = entry.file_name().unwrap_or_default().to_string_lossy();
                                let icon = if entry.is_dir() { "📁" } else { "📄" };
                                let resp = ui.selectable_label(false, format!("{} {}", icon, name));
                                
                                resp.context_menu(|ui| {
                                    if ui.button("✨ Explain File").clicked() {
                                        if let Ok(content) = std::fs::read_to_string(entry) {
                                            let ext = entry.extension().and_then(|e| e.to_str()).unwrap_or("text");
                                            let prompt = format!("Explain this file:\n\n```{}\n{}\n```", ext, content);
                                            self.send_chat_request(prompt);
                                        }
                                        ui.close_menu();
                                    }
                                });

                                if resp.clicked() {
                                    if entry.is_dir() {
                                        self.current_dir = Some(entry.clone());
                                        self.refresh_explorer();
                                    } else {
                                        self.open_specific_file(entry.clone());
                                    }
                                }
                            }
                        });
                    } else {
                        ui.label("Open a file to browse its directory.");
                    }
                    
                    ui.separator();
                    ui.label("Open Tabs:");
                    for (i, buf) in self.buffers.iter().enumerate() {
                        if ui.selectable_label(i == self.active_buffer, buf.title()).clicked() { self.active_buffer = i; }
                    }
                });
        }

        if self.show_ai_panel {
            egui::SidePanel::right("ai_panel")
                .default_width(350.0)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.heading("✨ Local AI Assistant");
                    ui.separator();

                    if !self.ai_client.is_loaded() {
                        if self.is_loading_model {
                            ui.horizontal(|ui| { ui.spinner(); ui.label("Loading model..."); });
                        } else {
                            ui.label("⚠️ No model loaded.");
                            if ui.button("Load Local .gguf Model").clicked() {
                                if let Some(path) = rfd::FileDialog::new().add_filter("GGUF", &["gguf"]).pick_file() { self.load_model(path); }
                            }
                        }
                        return; 
                    }

                    let fallback_ext = self.buffers[self.active_buffer].extension().to_string();
                    
                    egui::ScrollArea::vertical()
                        .id_source("ai_response")
                        .stick_to_bottom(self.ai_streaming)
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            for (role, content) in &self.chat_history {
                                if role == "user" {
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP).with_main_wrap(true), |ui| {
                                        ui.colored_label(egui::Color32::from_rgb(100, 150, 200), format!("🧑 {}", content));
                                    });
                                } else {
                                    ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP).with_main_wrap(true), |ui| {
                                        ui.label("🤖 ");
                                        ui.vertical(|ui| {
                                            ui.horizontal(|ui| {
                                                if ui.small_button("📋 Copy").clicked() {
                                                    ui.output_mut(|o| o.copied_text = content.clone());
                                                }
                                            });
                                            self.render_chat_markdown(ui, content, &fallback_ext);
                                        });
                                    });
                                }
                                ui.separator();
                            }

                            if self.ai_streaming || !self.ai_response.is_empty() {
                                ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP).with_main_wrap(true), |ui| {
                                    ui.label("🤖 ");
                                    ui.vertical(|ui| {
                                        self.render_chat_markdown(ui, &self.ai_response, &fallback_ext);
                                    });
                                });
                            }
                        });

                    ui.separator();

                    if ui.memory(|m| m.has_focus(self.ai_prompt_id)) {
                        if ui.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                            if !self.prompt_history.is_empty() {
                                let new_idx = match self.prompt_history_index {
                                    Some(idx) => if idx == 0 { 0 } else { idx - 1 },
                                    None => self.prompt_history.len() - 1,
                                };
                                self.prompt_history_index = Some(new_idx);
                                self.ai_prompt = self.prompt_history[new_idx].clone();
                            }
                        }
                        if ui.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                            if let Some(idx) = self.prompt_history_index {
                                if idx == self.prompt_history.len() - 1 {
                                    self.prompt_history_index = None;
                                    self.ai_prompt.clear();
                                } else {
                                    let new_idx = idx + 1;
                                    self.prompt_history_index = Some(new_idx);
                                    self.ai_prompt = self.prompt_history[new_idx].clone();
                                }
                            }
                        }
                    }

                    ui.vertical_centered_justified(|ui| {
                        let hint = if self.ai_prompt.starts_with('/') { "Use /explain, /fix, /refactor, /tests" } else { "Ask about selected code or whole file..." };
                        let response = ui.add(
                            egui::TextEdit::multiline(&mut self.ai_prompt)
                                .id(self.ai_prompt_id)
                                .hint_text(hint)
                                .desired_width(ui.available_width())
                                .desired_rows(3),
                        );
                        if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) { self.send_chat_request(self.ai_prompt.clone()); }
                        ui.add_space(4.0);
                        
                        ui.horizontal(|ui| {
                            if ui.button("Send (Enter)").clicked() { self.send_chat_request(self.ai_prompt.clone()); }
                            if (self.ai_streaming || self.inline_generating) && ui.button("Stop").clicked() {
                                self.is_cancelled.store(true, Ordering::SeqCst);
                                if self.inline_generating {
                                    self.buffers[self.active_buffer].code = self.inline_original_code.clone();
                                    self.inline_generating = false;
                                }
                            }
                            if !self.ai_response.is_empty() {
                                let btn_text = if self.selection_start != self.selection_end { "Review Changes" } else { "Review Insert" };
                                if ui.button(btn_text).clicked() { self.prepare_diff_review(); }
                            }
                        });
                    });
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.buffers.is_empty() { return; }
            
            ui.horizontal(|ui| {
                if let Some(path) = &self.buffers[self.active_buffer].path.clone() {
                    let ancestors: Vec<PathBuf> = path.ancestors().collect();
                    for (i, ancestor) in ancestors.iter().rev().enumerate() {
                        if i > 0 { ui.label("›"); }
                        let name = ancestor.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "/".to_string());
                        if ui.button(name).clicked() {
                            if ancestor.is_dir() {
                                self.current_dir = Some(ancestor.to_path_buf());
                                self.refresh_explorer();
                            }
                        }
                    }
                } else {
                    ui.label("📂 untitled");
                }
                
                if self.show_find_replace {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("×").clicked() { self.show_find_replace = false; }
                        ui.add_space(8.0);
                        if ui.button("Replace All").clicked() { self.replace_all(); }
                        if ui.button("Replace").clicked() { self.replace_current(); }
                        ui.add_space(4.0);
                        ui.add(egui::TextEdit::singleline(&mut self.replace_query).hint_text("Replace").desired_width(100.0));
                        ui.add_space(8.0);
                        if ui.button("Find Next").clicked() { self.find_next(); }
                        ui.add(egui::TextEdit::singleline(&mut self.find_query).hint_text("Find").desired_width(100.0));
                    });
                }
            });
            ui.separator();

            let ext = self.buffers[self.active_buffer].extension().to_string();
            let job = self.highlighter.highlight(&self.buffers[self.active_buffer].code, &ext);
            let num_lines = self.buffers[self.active_buffer].code.lines().count().max(1);

            if ui.memory(|m| m.has_focus(self.code_editor_id)) {
                if ui.input(|i| i.key_pressed(egui::Key::Tab) && !i.modifiers.ctrl && !i.modifiers.alt) {
                    if self.selection_start != self.selection_end {
                        ui.input_mut(|i| { i.consume_key(egui::Modifiers::NONE, egui::Key::Tab); i.consume_key(egui::Modifiers::SHIFT, egui::Key::Tab); });
                        if ui.input(|i| i.modifiers.shift) { self.unindent_block(); } else { self.indent_block(); }
                    }
                }
                if ui.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Slash)) {
                    ui.input_mut(|i| { i.consume_key(egui::Modifiers::CTRL, egui::Key::Slash); });
                    self.toggle_comments();
                }
                if ui.input(|i| i.modifiers.ctrl && i.modifiers.shift && i.key_pressed(egui::Key::D)) {
                    ui.input_mut(|i| { i.consume_key(egui::Modifiers::CTRL | egui::Modifiers::SHIFT, egui::Key::D); });
                    self.duplicate_line();
                }
            }

            let mut enter_intercepted = false;
            let mut text_to_insert = None;
            
            if ui.memory(|m| m.has_focus(self.code_editor_id)) {
                if ui.input(|i| i.key_pressed(egui::Key::Enter)) && !ui.input(|i| i.modifiers.shift) {
                    enter_intercepted = true;
                }

                for event in ui.input(|i| i.events.clone()) {
                    if let egui::Event::Text(s) = event {
                        if s == "(" || s == "[" || s == "{" {
                            let pair = match s.as_str() {
                                "(" => "()",
                                "[" => "[]",
                                "{" => "{}",
                                _ => ""
                            };
                            if !pair.is_empty() {
                                text_to_insert = Some(pair.to_string());
                                ui.input_mut(|i| i.events.retain(|e| !matches!(e, egui::Event::Text(_))));
                            }
                        }
                    }
                }
            }

            if let Some(pair) = text_to_insert {
                self.buffers[self.active_buffer].code.insert_str(self.cursor_index, &pair);
                self.cursor_index += 1;
                self.buffers[self.active_buffer].dirty = true;
                if let Some(mut state) = egui::TextEdit::load_state(ctx, self.code_editor_id) {
                    state.cursor.set_char_range(Some(egui::text::CursorRange {
                        primary: egui::text::CCursor::new(self.cursor_index).into(),
                        secondary: egui::text::CCursor::new(self.cursor_index).into(),
                    }));
                    state.store(ctx, self.code_editor_id);
                }
            }

            if enter_intercepted {
                let code = &self.buffers[self.active_buffer].code;
                let line_start = code[..self.cursor_index].rfind('\n').map(|i| i + 1).unwrap_or(0);
                let current_line = &code[line_start..self.cursor_index];
                let indent: String = current_line.chars().take_while(|c| c.is_whitespace()).collect();
                
                let indent_str = if self.config.use_spaces { " ".repeat(self.config.tab_width) } else { "\t".to_string() };
                let mut to_insert = format!("\n{}", indent);
                
                if current_line.trim_end().ends_with('{') {
                    to_insert.push_str(&indent_str);
                }
                
                self.buffers[self.active_buffer].code.insert_str(self.cursor_index, &to_insert);
                self.cursor_index += to_insert.len();
                self.buffers[self.active_buffer].dirty = true;
                
                if let Some(mut state) = egui::TextEdit::load_state(ctx, self.code_editor_id) {
                    state.cursor.set_char_range(Some(egui::text::CursorRange {
                        primary: egui::text::CCursor::new(self.cursor_index).into(),
                        secondary: egui::text::CCursor::new(self.cursor_index).into(),
                    }));
                    state.store(ctx, self.code_editor_id);
                }
                
                ui.input_mut(|i| i.events.retain(|e| !matches!(e, egui::Event::Key { key: egui::Key::Enter, pressed: true, .. })));
            }

            egui::SidePanel::right("minimap")
                .resizable(false)
                .exact_width(60.0)
                .show_inside(ui, |ui| {
                    let mut minimap_job = self.highlighter.highlight(&self.buffers[self.active_buffer].code, &ext);
                    for section in &mut minimap_job.sections {
                        section.format.font_id = egui::FontId::proportional(3.0);
                        if !self.config.dark_mode {
                            section.format.color = section.format.color.gamma_multiply(0.5); 
                        }
                    }
                    minimap_job.wrap.width = 60.0;
                    ui.label(egui::text::TextBufferLabel::new(minimap_job));
                });

            egui::ScrollArea::both().show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    ui.vertical(|ui| {
                        for i in 1..=num_lines {
                            ui.label(egui::RichText::new(format!("{}", i)).color(egui::Color32::from_rgb(80, 80, 80)).family(egui::FontFamily::Monospace));
                        }
                    });
                    ui.separator();

                    let wrap_width = if self.word_wrap { ui.available_width() } else { f32::MAX };
                    let mut layouter = move |ui: &egui::Ui, _wrap_width, _text| {
                        let mut job = job.clone();
                        job.wrap.width = wrap_width;
                        job
                    };

                    let indent_str = if self.config.use_spaces { " ".repeat(self.config.tab_width) } else { "\t".to_string() };

                    let output = egui::TextEdit::multiline(&mut self.buffers[self.active_buffer].code)
                        .id(self.code_editor_id)
                        .code_editor()
                        .lock_focus(false)
                        .desired_width(f32::MAX)
                        .indent_string(indent_str)
                        .layouter(&mut layouter)
                        .show(ui);

                    if let Some(cursor_range) = output.cursor_range {
                        let primary = cursor_range.primary.ccursor.index;
                        let secondary = cursor_range.secondary.ccursor.index;
                        if primary != secondary {
                            self.selection_start = primary.min(secondary);
                            self.selection_end = primary.max(secondary);
                        } else {
                            self.selection_start = primary;
                            self.selection_end = primary;
                            self.cursor_index = primary;
                        }
                    }

                    output.response.context_menu(|ui| {
                        if ui.button("Cut").clicked() { ui.output_mut(|o| o.copied_text = self.buffers[self.active_buffer].code[self.selection_start..self.selection_end].to_string()); self.buffers[self.active_buffer].code.replace_range(self.selection_start..self.selection_end, ""); ui.close_menu(); }
                        if ui.button("Copy").clicked() { ui.output_mut(|o| o.copied_text = self.buffers[self.active_buffer].code[self.selection_start..self.selection_end].to_string()); ui.close_menu(); }
                        if ui.button("Paste").clicked() { if let Some(clip) = ui.output(|o| o.copied_text.clone()) { self.buffers[self.active_buffer].code.insert_str(self.cursor_index, &clip); } ui.close_menu(); }
                        ui.separator();
                        if ui.button("✨ Explain Code").clicked() { self.send_chat_request("Explain this code.".to_string()); ui.close_menu(); }
                        if ui.button("⚡ Refactor Code").clicked() { self.send_chat_request("Refactor this code.".to_string()); ui.close_menu(); }
                        if ui.button("🤖 Complete (Ctrl+Space)").clicked() { self.trigger_inline_ai(); ui.close_menu(); }
                    });

                    if output.response.has_focus() {
                        if ui.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Space)) { self.trigger_inline_ai(); }
                        if self.inline_generating && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                            self.is_cancelled.store(true, Ordering::SeqCst);
                            self.buffers[self.active_buffer].code = self.inline_original_code.clone();
                            self.inline_generating = false;
                        }
                        if ui.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::O)) { self.open_file(); }
                        if ui.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::S)) { self.save_file(); }
                        if output.response.changed() { self.buffers[self.active_buffer].dirty = true; }
                    }
                });
            });
        });
    }
}
