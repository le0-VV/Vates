# Mac mini Qwen3-Next OpenAI capability audit (2026-07-25)

## Result

The persistent Vates endpoint currently exposes text chat only. It does not
expose native reasoning metadata, image input or structured tool calling.

| Surface | Model/checkpoint evidence | Server evidence | Live result |
| --- | --- | --- | --- |
| Text chat | `Qwen3NextForCausalLM` instruct checkpoint | String `system`, `user` and `assistant` messages map to `message.content` | Supported |
| Reasoning/thinking | Tokenizer contains `<think>` tokens, but has no chat template or thinking-mode switch | `reasoning_effort` is discarded; responses have no `reasoning_content` or reasoning-token accounting | Not exposed |
| Vision | No `vision_config`, image processor or preprocessor files | Message content must be a string | Not supported |
| Tool use | Tokenizer contains tool-call tokens, but no tool-aware chat template is installed | `tools`, `tool_choice`, tool messages and tool-call responses are explicitly rejected | Not exposed |

Tokenizer special-token names are not evidence that the loaded causal-language
model has a vision encoder or that the server implements tool/reasoning
protocols.

## Data flow

The server validates a narrow OpenAI-compatible subset:

- each message role must be `system`, `user` or `assistant`;
- each message `content` must be a string;
- `tools`, `tool_choice`, legacy function fields and tool-message fields are
  rejected;
- unrecognised metadata such as `reasoning_effort` is ignored;
- non-streaming responses contain only `message.role` and `message.content`;
- streaming responses emit only role/content deltas.

The backend passes these cleaned messages to `_encode_chat`. The pinned
`tokenizer_config.json` has no `chat_template`, so encoding falls back to plain
`role: content` lines followed by `assistant:`. There is therefore no native
thinking flag, tool schema injection, tool-call parser or multimodal processor
in the active path.

## Live probes

All probes targeted the persistent w16/40r/24s/k3 server on port 8000:

1. A request with `reasoning_effort: "high"` returned HTTP 200 and a normal
   textual explanation. The response had no reasoning field; source tracing
   confirms that the request field is discarded.
2. An OpenAI content array containing text and a one-pixel PNG data URL
   returned HTTP 400:
   `messages[0].content must be string content`.
3. A request containing one function tool and `tool_choice: "auto"` returned
   HTTP 400:
   `tools is not supported`.

The earlier `VATES_OK` Chatbox check therefore proved basic text
interoperability only.

## Implementation implications

Structured tool use can be added to this text checkpoint by extending request
validation, installing or constructing the correct tool-aware prompt template,
parsing tool-call output and returning OpenAI `tool_calls` deltas/messages.
This requires correctness tests before it can be advertised to Chatbox.

Reasoning exposure is separate work: the runtime needs an explicitly supported
thinking prompt mode and a defined API mapping, such as `reasoning_content`.
Ordinary requests for an explanation do not establish native thinking mode.

Vision requires a multimodal checkpoint and processor/runtime path. The current
text-only `Qwen3NextForCausalLM` checkpoint cannot acquire vision by accepting
an `image_url` field alone.
