# Native thinking controls

MoE4All passes thinking controls to the GGUF's embedded `tokenizer.chat_template`.
The model template defines their meaning. A reasoning level is not a token budget,
a sampling temperature, or a change to MoE routing.

## CLI and Windows wizard

```powershell
.\infr.exe run --think --reasoning-effort medium 'D:\Models\model.gguf'
.\infr.exe serve --think --reasoning-effort medium 'D:\Models\model.gguf'
```

The Windows launch wizard offers a reasoning-effort selector after the thinking
on/off selector and remembers the choice. It omits the effort flag when thinking
is disabled. The GUI application has no new dropdown in this change; use the
wizard, CLI or API for these controls.

The equivalent config is:

```toml
[sampling]
no_think = false
reasoning_effort = "medium"
# API history retention; this does not change the interactive REPL's answer-only history.
# preserve_thinking = true
```

`--set sampling.reasoning_effort=medium` also works. Omit the setting to use the
template's default. `--set sampling.reasoning_effort=none` clears an inherited
config value; it does not disable thinking. Use `--no-think` for that.

No new environment variables are introduced. Existing `INFR_NO_THINK` behavior
is unchanged. An effort setting does not implicitly override `--no-think`.

## Chat Completions API

Send to `POST /v1/chat/completions`:

```json
{
  "model": "your-served-model-id",
  "messages": [{"role": "user", "content": "Explain this problem."}],
  "reasoning_effort": "medium",
  "chat_template_kwargs": {
    "enable_thinking": true,
    "preserve_thinking": true
  },
  "max_completion_tokens": 4096,
  "stream": true
}
```

With the Python OpenAI SDK, put `chat_template_kwargs` in `extra_body`. At the HTTP
level it is a top-level field; do not send a literal `extra_body` wrapper.

Supported `chat_template_kwargs` are `enable_thinking` (boolean),
`preserve_thinking` (boolean) and `reasoning_effort` (string). Other keys are
rejected so they cannot replace reserved prompt data or silently do nothing.
The effort may be top-level or nested. Identical duplicates are accepted;
conflicting duplicates return 400.

Absent/null fields inherit server config. Absent config effort and preservation
keys are omitted from Jinja entirely, retaining each template's default. Request
values override process defaults without mutating shared configuration. The same
options render the generation prompt and stable history for prefix caching.

To retain earlier reasoning, replay the returned assistant `reasoning_content`
alongside `content` in subsequent requests. `reasoning` is also accepted as a
fallback; when both are supplied, `reasoning_content` wins. The template decides
which turns to retain. `preserve_thinking=false` in Qwen3.8 still keeps reasoning
in the current tool-call turn, as specified by Qwen's template. Retaining history
can increase context length and processing cost.

The interactive REPL retains its existing answer-only history. This change adds
native effort control there; it does not redesign the REPL history format.

## Model differences

| Model/template | Native effort behavior |
| --- | --- |
| Qwen3.8-Flash-Next / Qwen3.8-27B | `low`, `medium`, `xhigh`; default `xhigh`. `high` is invalid. |
| Qwen3-style templates without `reasoning_effort` | Thinking on/off works as before; explicit effort fails instead of being silently ignored. |
| Gemma 4 31B official template inspected 2026-09-09 | Has thinking on/off and preservation controls, but no `reasoning_effort` variable. Do not assume medium/high levels exist. |
| DeepSeek-V4-Flash-0731 | Official encoder defines `low`, `high`, `max` (default `low`). It ships Python encoding code rather than a stock Jinja template. Its converted GGUF must contain an equivalent template that actually consumes `reasoning_effort`. This patch does not add the Python encoder or replace GGUF templates. |
| Other templates, including GPT-OSS-style prompt formats | Native variables can be passed through. This is not a claim that MoE4All can load every model architecture. |

The accepted wire/config vocabulary is `low`, `medium`, `high`, `xhigh`, `max`.
It is not a universal five-level scale. Model-specific validation remains in the
embedded template; no `high` -> `xhigh` translation or generic system-message
injection is performed. Templates that never reference an explicitly requested
effort/preservation option are rejected. Referencing the variable is a necessary
capability check, not proof that a third-party template implements every level
correctly. Old GGUF conversions may need an updated model template.

Malformed options produce an OpenAI-shaped HTTP 400. A model template's explicit
input rejection produces HTTP 400 for non-streaming requests. For streaming
requests, headers may already be 200: the response then contains an
`invalid_request_error` SSE event followed by exactly one `[DONE]`. Clients must
inspect SSE error events. Broken templates and backend failures remain server
errors.

## Verification

Use Rust stable and the platform build prerequisites from the repository's CI
and Windows build instructions (including a suitable `glslc` for Vulkan).

```powershell
cargo fmt --all
cargo fmt --all -- --check
cargo test -p infr-core -p infr-chat -p infr-server -p infr-cli --locked
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

To check the fixture expectations with Python Jinja2 independently, install
`Jinja2` in your Python environment and run `python scripts/check-thinking-fixtures.py`.
This does not execute the Rust implementation and does not replace the Rust tests.

The Qwen fixture matrix covers 20 cases: defaults, all three supported levels,
invalid levels, thinking off, system messages, tools, reasoning retention and
stable history. Rust tests additionally cover DTO forwarding through JSON/SSE,
conflicts, bad types, reserved fields, template capabilities, concurrent renders,
config inheritance and typed input errors. These tests need no model weights.

Run GPU acceptance separately with your actual GGUF on Windows: compare the
rendered prompts for low/medium/xhigh, run short generations with identical
sampling, verify history/tool turns, and switch levels between consecutive calls.
Different levels need not produce monotonically increasing token counts on every
question. This patch does not set a hard thinking-token limit.

## Sources and fixture provenance

- [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Qwen3.8-Flash-Next template](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/chat_template.jinja)
- [Qwen3.8-27B template](https://huggingface.co/Qwen/Qwen3.8-27B/blob/412f8b6/chat_template.jinja)
- [Gemma 4 31B template](https://huggingface.co/google/gemma-4-31B-it/blob/main/chat_template.jinja)
- [DeepSeek-V4-Flash-0731 encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/encoding/encoding_dsv4.py)

`crates/infr-chat/tests/fixtures/qwen38_chat_template.jinja` is a whitespace-normalized
text extraction of Qwen/Qwen3.8-27B's Apache-2.0 template, revision `412f8b6`, read
2026-09-09. Copyright belongs to the Qwen authors. Its 170 extracted source lines
were compared with the Flash-Next template (revision `34567a4`) and were identical.
It is a regression fixture only; production always uses the actual GGUF metadata.
The repository's Apache-2.0 license applies to the fixture. No model weights are
included.
