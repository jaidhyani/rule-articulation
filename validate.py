"""Validate (a) thinking is fully disabled in the study requests, and
(b) identical requests are non-deterministic (multisampling works)."""
import json
import llm

print("=== thinking-disabled check (classify path) ===")
_, r = llm.chat_nothink([{"role": "user", "content": "Input: hello world\nLabel:"}],
                        system=llm.CLASSIFY_SYSTEM, max_tokens=5)
print("content block types:", [b.type for b in r.content])  # expect ['text'] only, no 'thinking'
print("text:", repr("".join(b.text for b in r.content if b.type == "text")))
print("usage:", json.dumps(r.usage.model_dump(), default=str))

print("\n=== thinking-disabled check (longer free response) ===")
t, r2 = llm.chat_nothink([{"role": "user", "content": "Name one rule that could label sentences."}],
                         max_tokens=200)
print("content block types:", [b.type for b in r2.content], "(no 'thinking' => disabled)")

print("\n=== multisampling: same prompt x8, thinking disabled ===")
msg = [{"role": "user", "content": "Reply with exactly one random integer from 1 to 1000 and nothing else."}]
outs = [llm.chat_nothink(msg, max_tokens=10)[0].strip() for _ in range(8)]
print("outputs:", outs, "-> distinct:", len(set(outs)))

print("\n=== multisampling consistency: same articulation prompt x6 ===")
shots = ("Input: tiny turtles take time\nLabel: True\n"
         "Input: The dog slept all day.\nLabel: False\n"
         "Input: cats catch crabs\nLabel: True\n"
         "Input: We walked to the park.\nLabel: False")
ask = (shots + "\n\nEach input was labeled by one hidden rule. State the rule in one sentence.")
for i in range(6):
    txt = llm.chat_nothink([{"role": "user", "content": ask}], max_tokens=120)[0].strip().replace("\n", " ")
    print(f"  [{i+1}] {txt[:140]}")
