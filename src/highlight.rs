use egui::text::LayoutJob;
use syntect::highlighting::{HighlightLines, ThemeSet};
use syntect::parsing::SyntaxSet;

pub struct Highlighter {
    syntax_set: SyntaxSet,
    theme_set: ThemeSet,
    current_theme: String,
}

impl Default for Highlighter {
    fn default() -> Self {
        Self::new()
    }
}

impl Highlighter {
    pub fn new() -> Self {
        Self {
            syntax_set: SyntaxSet::load_defaults_newlines(),
            theme_set: ThemeSet::load_defaults(),
            current_theme: "base16-ocean.dark".to_string(),
        }
    }

    pub fn set_theme(&mut self, theme: &str) {
        if self.theme_set.themes.contains_key(theme) {
            self.current_theme = theme.to_string();
        }
    }

    pub fn get_themes(&self) -> Vec<String> {
        self.theme_set.themes.keys().cloned().collect()
    }

    pub fn highlight(&self, code: &str, extension: &str) -> LayoutJob {
        let mut job = LayoutJob::default();
        let syntax = self
            .syntax_set
            .find_syntax_by_extension(extension)
            .unwrap_or_else(|| self.syntax_set.find_syntax_plain_text());

        let theme = &self.theme_set.themes[&self.current_theme];
        let mut highlighter = HighlightLines::new(syntax, theme);

        for line in code.lines() {
            match highlighter.highlight_line(line, &self.syntax_set) {
                Ok(regions) => {
                    for (style, text) in regions {
                        let color = egui::Color32::from_rgb(
                            style.foreground.r,
                            style.foreground.g,
                            style.foreground.b,
                        );
                        job.append(text, 0.0, egui::TextFormat {
                            color,
                            ..Default::default()
                        });
                    }
                }
                Err(_) => {
                    job.append(line, 0.0, egui::TextFormat::default());
                }
            }
            job.append("\n", 0.0, egui::TextFormat::default());
        }

        job
    }
}
