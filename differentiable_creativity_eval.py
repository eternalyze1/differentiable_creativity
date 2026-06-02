import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time, argparse, re

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--lr", type=float, default=1e-5, help="DCLM learning rate")
parser.add_argument("--distill_temp", type=float, default=0.6, help="Teacher temperature for distillation")
parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math"])
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--num", type=int, default=50)
args = parser.parse_args()

dtype = torch.float16

print(f"Loading {args.model}...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model, trust_remote_code=True,
    torch_dtype=dtype, device_map="auto",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model_device = next(model.parameters()).device
print(f"Loaded in {time.time()-t0:.1f}s on {model_device}")

# Save original weights to restore between DCLM problems
original_state = {k: v.clone() for k, v in model.state_dict().items()}

V = model.config.vocab_size

print(f"Loading {args.dataset.upper()}...")
if args.dataset == "gsm8k":
    ds = load_dataset("gsm8k", "main", split=f"test[:{args.num}]")
    q_key, a_key = "question", "answer"
else:
    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split=f"test[:{args.num}]")
    q_key, a_key = "problem", "answer"

def sample(logits):
    return torch.argmax(logits, dim=-1, keepdim=True)

def extract_answer(text):
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    return None

def extract_expected(text):
    """Extract numeric answer from GSM8K answer field (format: 'explanation #### NUMBER')."""
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    return text.strip()

def generate_baseline(prompt_ids, max_new_tokens=512):
    model.eval()
    past = None
    generated = prompt_ids.clone()
    hit_limit = True
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                generated if past is None else generated[:, -1:],
                past_key_values=past, use_cache=True,
            )
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
        token = sample(logits[:, :V])
        if token.item() == tokenizer.eos_token_id:
            hit_limit = False
            break
        generated = torch.cat([generated, token], dim=-1)
    return generated[0], hit_limit

def generate_dclm(prompt_ids, max_new_tokens=512, lr=1e-5, distill_temp=2.0):
    """DCLM: training at inference time via knowledge distillation.
    Forward with grad tracking, compute a 'teacher' distribution from
    logits at higher temperature, then backprop CE(teacher || softmax(logits))
    through the full network and take an SGD step. This trains the model
    to reproduce a more uniform (creative) distribution at each step.
    Uses KV cache with truncated BPTT (past tensors detached)."""
    model.eval()
    with torch.no_grad():
        model.load_state_dict(original_state)
    generated = prompt_ids.clone()
    hit_limit = True
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Build prompt KV cache (no_grad)
    with torch.no_grad():
        past = model(generated, use_cache=True).past_key_values

    for _ in range(max_new_tokens):
        outputs = model(generated[:, -1:], past_key_values=past, use_cache=True)
        logits = outputs.logits[:, -1, :V]

        # Sample token (detached, just for continuing generation)
        token = sample(logits.detach())

        if token.item() == tokenizer.eos_token_id:
            hit_limit = False
            generated = torch.cat([generated, token], dim=-1)
            break

        # Sharpened self-distillation: CE(softmax(logits/T) || softmax(logits))
        # T<1 = sharper teacher, pulling model toward more confident predictions
        with torch.no_grad():
            target = F.softmax(logits / distill_temp, dim=-1)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Detach KV cache after backward to prevent graph chaining
        for i in range(len(past.layers)):
            past.layers[i].keys = past.layers[i].keys.detach()
            past.layers[i].values = past.layers[i].values.detach()

        generated = torch.cat([generated, token], dim=-1)

    return generated[0], hit_limit

results = {"baseline": {}, "dclm": {}}
for label, gen_fn, store in [
    ("BASELINE", generate_baseline, results["baseline"]),
    ("DCLM", generate_dclm, results["dclm"]),
]:
    correct = 0
    total = 0
    truncated = 0
    total_gen_tokens = 0
    t_start = time.time()
    for i, example in enumerate(ds):
        question = example[q_key]
        answer = example[a_key]
        messages = [{"role": "user", "content": f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model_device)
        prompt_len = inputs["input_ids"].shape[1]
        if label == "DCLM":
            full_ids, hit_limit = gen_fn(inputs["input_ids"], max_new_tokens=args.max_new_tokens, lr=args.lr, distill_temp=args.distill_temp)
        else:
            full_ids, hit_limit = gen_fn(inputs["input_ids"], max_new_tokens=args.max_new_tokens)
        gen_len = full_ids.shape[0] - prompt_len
        total_gen_tokens += gen_len
        output_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)
        predicted = extract_answer(output_text)
        expected = extract_expected(str(answer))
        if hit_limit:
            truncated += 1
            print(f"  [WARNING] Hit max_new_tokens limit ({args.max_new_tokens}) - generation truncated!")
        correct += 1 if (predicted is not None and expected is not None and predicted == expected) else 0
        total += 1
        q_safe = question.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2014", "--").replace("\u2013", "-")
        a_safe = output_text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2014", "--").replace("\u2013", "-")
        print(f"[{label}] #{i} ({gen_len} tok) pred={predicted} expected={expected} {'[OK]' if predicted == expected else '[FAIL]'}")
        print(f"  Q: {q_safe}")
        print(f"  A: {a_safe}")
        print()
    acc = correct / total * 100 if total > 0 else 0
    elapsed = time.time() - t_start
    avg_len = total_gen_tokens / total if total > 0 else 0
    store["accuracy"] = acc
    store["correct"] = correct
    store["total"] = total
    store["truncated"] = truncated
    store["avg_len"] = avg_len
    store["time"] = elapsed
    print(f"[{label}] Done: {correct}/{total}={acc:.1f}% ({elapsed:.0f}s)")

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
for label, store in [("BASELINE", results["baseline"]), ("DCLM", results["dclm"])]:
    if store:
        print(f"{label}: {store['correct']}/{store['total']}={store['accuracy']:.1f}% | truncated={store['truncated']}/{store['total']} | avg_len={store['avg_len']:.0f} tokens ({store['time']:.0f}s)")
