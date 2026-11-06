These four qwen2.5:7b runs are invalid and are kept only for the record. The prompt is 104.8K tokens on Qwen's
tokenizer; Ollama clamped the context to the model's 32K and silently kept only the first 4 and last ~16.4K tokens,
so the model never saw the description or the question (only the tail of the weights). The harness now sends
`truncate: false`, which turns this into an explicit `context_overflow` record.
