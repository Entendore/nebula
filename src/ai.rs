use anyhow::Result;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams, standard_sampler::StandardSampler};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc;

pub enum AiEvent {
    Chunk(String),
    InlineChunk(String),
    Done,
    Error(String),
    ModelLoaded(Arc<LlamaModel>),
}

#[derive(Clone)]
pub struct AiClient {
    pub model: Option<Arc<LlamaModel>>,
    pub temperature: f32,
    pub max_tokens: usize,
}

impl Default for AiClient {
    fn default() -> Self {
        Self { 
            model: None, 
            temperature: 0.7, 
            max_tokens: 1024 
        }
    }
}

impl AiClient {
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    pub fn set_model(&mut self, model: Arc<LlamaModel>) {
        self.model = Some(model);
    }

    pub fn spawn_load_model(path: PathBuf, n_gpu_layers: u32, tx: mpsc::UnboundedSender<AiEvent>) {
        tokio::task::spawn_blocking(move || {
            let params = LlamaParams {
                n_gpu_layers: n_gpu_layers as i32,
                ..Default::default()
            };
            
            match LlamaModel::load_from_file(&path, &params) {
                Ok(model) => { let _ = tx.send(AiEvent::ModelLoaded(Arc::new(model))); }
                Err(e) => { let _ = tx.send(AiEvent::Error(format!("Failed to load model: {}", e))); }
            }
        });
    }

    pub async fn stream_chat_completion(
        &self,
        prompt: &str,
        code_context: &str,
        history: &[(String, String)],
        tx: mpsc::UnboundedSender<AiEvent>,
        is_cancelled: Arc<AtomicBool>,
    ) -> Result<()> {
        let model = match &self.model { Some(m) => m.clone(), None => return Ok(()) };

        let mut full_prompt = String::new();
        full_prompt.push_str("<|im_start|>system\nYou are an expert AI coding assistant. ");
        if code_context.lines().count() > 5 {
            full_prompt.push_str("The user is editing a file with this content:\n\n");
            full_prompt.push_str(code_context);
        } else {
            full_prompt.push_str("The user highlighted this code snippet:\n\n");
            full_prompt.push_str(code_context);
        }
        full_prompt.push_str("\n\nRespond concisely. Use markdown code blocks.\n<|im_end|>\n");

        for (role, content) in history {
            full_prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
        }
        full_prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", prompt));
        full_prompt.push_str("<|im_start|>assistant\n");

        let prompt_tx = tx.clone();
        
        tokio::task::spawn_blocking(move || {
            let session_params = SessionParams { n_ctx: 8192, ..Default::default() };
            let mut session = match model.create_session(&session_params) {
                Ok(s) => s,
                Err(e) => { let _ = prompt_tx.send(AiEvent::Error(format!("Session error: {}", e))); return; }
            };

            if let Err(e) = session.feed_prompt(&full_prompt, |_| {}, || false) {
                let _ = prompt_tx.send(AiEvent::Error(format!("Feed error: {}", e)));
                return;
            }

            let sampler = StandardSampler::default();
            loop {
                if is_cancelled.load(Ordering::SeqCst) { let _ = prompt_tx.send(AiEvent::Done); return; }
                match session.advance(&sampler) {
                    Ok(token) => {
                        if token == model.token_eos() { break; }
                        if let Ok(text) = model.token_to_str(token, false) {
                            let _ = prompt_tx.send(AiEvent::Chunk(text));
                        }
                    }
                    Err(e) => { let _ = prompt_tx.send(AiEvent::Error(format!("Gen error: {}", e))); break; }
                }
            }
            let _ = prompt_tx.send(AiEvent::Done);
        });
        Ok(())
    }

    pub async fn stream_inline_completion(
        &self,
        code_before_cursor: &str,
        tx: mpsc::UnboundedSender<AiEvent>,
        is_cancelled: Arc<AtomicBool>,
    ) -> Result<()> {
        let model = match &self.model { Some(m) => m.clone(), None => return Ok(()) };
        let full_prompt = code_before_cursor.to_string();
        let prompt_tx = tx.clone();

        tokio::task::spawn_blocking(move || {
            let session_params = SessionParams { n_ctx: 8192, ..Default::default() };
            let mut session = match model.create_session(&session_params) {
                Ok(s) => s,
                Err(e) => { let _ = prompt_tx.send(AiEvent::Error(format!("Session error: {}", e))); return; }
            };

            if let Err(e) = session.feed_prompt(&full_prompt, |_| {}, || false) {
                let _ = prompt_tx.send(AiEvent::Error(format!("Feed error: {}", e)));
                return;
            }

            let sampler = StandardSampler::default();
            let mut tokens_generated = 0;

            while tokens_generated < 256 {
                if is_cancelled.load(Ordering::SeqCst) { break; }
                match session.advance(&sampler) {
                    Ok(token) => {
                        if token == model.token_eos() { break; }
                        if let Ok(text) = model.token_to_str(token, false) {
                            let _ = prompt_tx.send(AiEvent::InlineChunk(text));
                            tokens_generated += 1;
                        }
                    }
                    Err(_) => break,
                }
            }
            let _ = prompt_tx.send(AiEvent::Done);
        });
        Ok(())
    }
}

pub fn extract_code(text: &str) -> String {
    let re = regex::Regex::new(r"```(?:[a-zA-Z0-9_+\-]*\n)?([\s\S]*?)```").unwrap();
    if let Some(caps) = re.captures(text) {
        return caps.get(1).map(|m| m.as_str().trim().to_string()).unwrap_or_default();
    }
    text.trim().to_string()
}
