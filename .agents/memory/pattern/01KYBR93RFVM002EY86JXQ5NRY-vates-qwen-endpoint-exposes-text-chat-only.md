---
id: "01KYBR93RFVM002EY86JXQ5NRY"
title: "Vates Qwen endpoint exposes text chat only"
type: "pattern"
status: "active"
tags:
  - "vates"
  - "qwen3-next"
  - "openai-api"
  - "capabilities"
  - "reasoning"
  - "vision"
  - "tool-use"
created_at: "2026-07-25T04:24:34Z"
updated_at: "2026-07-25T04:24:34Z"
source:
  kind: "capability_audit"
  ref: "Mac mini Qwen3-Next OpenAI capability audit on 2026-07-25"
evidence:
  -
    kind: "implementation"
    text: "The server requires string message content, rejects tools/tool_choice and tool-message fields, drops unrecognised reasoning_effort metadata, and returns only role/content responses."
  -
    kind: "model_metadata"
    text: "The pinned configuration declares Qwen3NextForCausalLM and has no vision_config, processor or preprocessor files; tokenizer_config.json has no chat_template."
  -
    kind: "live_probe"
    text: "An image content array returned HTTP 400 with messages[0].content must be string content; an OpenAI tools request returned HTTP 400 with tools is not supported."
  -
    kind: "live_probe"
    text: "A reasoning_effort=high request returned HTTP 200 with an ordinary content-only explanation and no reasoning_content field."
confirm_public: true
related:
  - "01KYBQCZ4NDT4BCNHGB218JR5S"
content_hash: "sha256:6d3ee746e54c2e983236b010bf26b84cf76012109743bef8e59623922b37a381"
---
The persistent Qwen3-Next Vates endpoint exposes text chat only. It does not expose native reasoning metadata, multimodal image input or structured OpenAI tool calls. The server accepts only string system/user/assistant messages, ignores reasoning_effort, explicitly rejects tool fields and always returns content-only assistant messages. The pinned checkpoint is Qwen3NextForCausalLM with no vision configuration or processor files, and its tokenizer has no chat template; think, tool and vision token names alone do not implement those capabilities. Do not use a basic Chatbox completion as evidence for reasoning, vision or tool support.
