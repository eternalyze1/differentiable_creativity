import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import json
import datetime
import os

# ─── Configuration ───────────────────────────────────────────────────────────────

@dataclass
class RLTrainerConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    group_size: int = 4
    lr: float = 1e-6
    beta: float = 0.04
    epsilon: float = 0.2
    max_gen_len: int = 10000
    num_iters: int = 999999999
    log_interval: int = 3
    temperature: float = 0.8
    top_p: float = 0.9
    reward_scale: float = 1.0
    output_dir: str = "output"

QA_SYSTEM = "Generate an interesting specific math research question and then answer it."
REWARD_SYSTEM = (
    "Rate the interestingness of the last token in this math research generation "
    "on a scale from 1 to 10."
)


# ─── Model Wrapper ───────────────────────────────────────────────────────────────

class IntrinsicRewardModel(nn.Module):
    """Wraps the base LM with a learned reward head for per-token interestingness."""

    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        self.base_model.train()

        # frozen reference for KL regularisation
        self.ref_model = deepcopy(self.base_model).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # reward head: hidden_size -> scalar
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        ).to(device).to(dtype=torch.bfloat16)

    def generate_qa(self, trigger: str = "Tell me something interesting.",
                    n: int = 4) -> List[str]:
        """Produce n QA strings using the generator system prompt."""
        messages = [
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user",   "content": trigger},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        # sanity check: verify weights are finite
        for p in self.base_model.parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print("  ⚠ model has NaN weights! reloading...")
                self.__init__(self.model_name if hasattr(self, 'model_name') else "Qwen/Qwen2.5-0.5B-Instruct")
                break

        responses = []
        was_training = self.base_model.training
        self.base_model.eval()
        for _ in range(n):
            out = self.base_model.generate(
                **inputs,
                max_new_tokens=self.max_gen_len if hasattr(self, 'max_gen_len') else 96,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            resp = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
            responses.append(resp)
        if was_training:
            self.base_model.train()
        return responses

    def get_intrinsic_rewards(self, responses: List[str]) -> torch.Tensor:
        """
        Per-token interestingness scores for each response.
        Returns list of tensors, each shape (seq_len,) in 1-10 range.
        """
        device = self.device
        all_rewards = []

        for resp in responses:
            msg = [
                {"role": "system", "content": REWARD_SYSTEM},
                {"role": "assistant", "content": resp},
            ]
            full_str = self.tokenizer.apply_chat_template(msg, tokenize=False)
            encoded = self.tokenizer(full_str, return_tensors="pt").to(device)
            input_ids = encoded.input_ids

            # find where the assistant response starts
            msg_empty = [
                {"role": "system", "content": REWARD_SYSTEM},
                {"role": "assistant", "content": ""},
            ]
            prompt_str = self.tokenizer.apply_chat_template(msg_empty, tokenize=False)
            prompt_ids = self.tokenizer(prompt_str, return_tensors="pt").to(device)
            plen = prompt_ids.input_ids.shape[1]

            with torch.no_grad():
                outputs = self.base_model(
                    input_ids,
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[-1]  # (1, T, D)

            reward_logits = self.reward_head(hidden).squeeze(-1)  # (1, T)
            # map to 1-10 via sigmoid
            rewards = 1.0 + 9.0 * torch.sigmoid(reward_logits)

            # only response token rewards are used
            token_rewards = rewards[0, plen:].detach().cpu()
            all_rewards.append(token_rewards)

        return all_rewards

    def get_logprobs(self, responses: List[str],
                     use_ref: bool = False,
                     require_grad: bool = True) -> List[torch.Tensor]:
        """Per-token log-probabilities under the policy (or ref)."""
        model = self.ref_model if use_ref else self.base_model
        if use_ref:
            model.eval()
        device = self.device
        all_lps = []

        for resp in responses:
            msg = [
                {"role": "system", "content": QA_SYSTEM},
                {"role": "user",   "content": "Tell me something interesting."},
            ]
            prompt_str = self.tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            full_str = prompt_str + resp
            encoded = self.tokenizer(full_str, return_tensors="pt").to(device)

            prompt_ids = self.tokenizer(prompt_str, return_tensors="pt").to(device)
            plen = prompt_ids.input_ids.shape[1]

            ctx = torch.enable_grad() if (not use_ref and require_grad) else torch.no_grad()
            with ctx:
                outputs = model(**encoded)
                logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = encoded.input_ids[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            logprobs = -per_token_loss.view(1, -1)
            resp_lp = logprobs[0, plen - 1:]

            if not require_grad or use_ref:
                resp_lp = resp_lp.detach()

            all_lps.append(resp_lp)

        return all_lps


# ─── GRPO Trainer ────────────────────────────────────────────────────────────────

class GRPOTrainer:
    def __init__(self, model: IntrinsicRewardModel, cfg: RLTrainerConfig):
        self.model = model
        self.cfg = cfg
        self.model.max_gen_len = cfg.max_gen_len

        self.optimizer = torch.optim.AdamW(
            list(model.base_model.parameters()) + list(model.reward_head.parameters()),
            lr=cfg.lr,
        )
        os.makedirs(cfg.output_dir, exist_ok=True)
        self.log_file = os.path.join(cfg.output_dir, "training_log.txt")

    def log(self, *args, **kwargs):
        with open(self.log_file, "a", encoding="utf-8") as f:
            print(*args, file=f, **kwargs)

    def train_step(self, step: int):
        cfg = self.cfg
        device = self.model.device

        # 1. Generate group of responses
        responses = self.model.generate_qa(n=cfg.group_size)

        # 2. Compute log-probs (current policy, with grad) and intrinsic rewards
        logprobs = self.model.get_logprobs(responses, use_ref=False, require_grad=True)
        ref_logprobs = self.model.get_logprobs(responses, use_ref=True, require_grad=False)
        token_rewards = self.model.get_intrinsic_rewards(responses)

        # 2b. Centre all per-token rewards by subtracting the batch mean
        all_tokens = torch.cat([r for r in token_rewards])
        batch_mean = all_tokens.mean()
        for i in range(len(token_rewards)):
            token_rewards[i] = token_rewards[i] - batch_mean

        # 3. Group-normalised advantages (based on per-response mean reward)
        avg_rewards = torch.tensor([r.mean().item() for r in token_rewards],
                                   device=device)
        mean_r = avg_rewards.mean()
        std_r = avg_rewards.std(correction=0).clamp(min=1e-6)
        advantages = (avg_rewards - mean_r) / std_r

        # 4. REINFORCE loss with KL regularisation (each response its own length)
        total_policy_loss = 0.0
        total_kl_acc = 0.0
        total_tokens = 0

        for i in range(cfg.group_size):
            T_i = min(logprobs[i].shape[0], token_rewards[i].shape[0])
            if T_i == 0:
                continue
            pi_lp = logprobs[i][:T_i]
            ref_lp = ref_logprobs[i][:T_i]
            adv = advantages[i]

            policy_loss = -(pi_lp * adv).mean()
            total_policy_loss += policy_loss * T_i

            kl = torch.exp(ref_lp - pi_lp) - (ref_lp - pi_lp) - 1.0
            total_kl_acc += kl.mean() * T_i
            total_tokens += T_i

        total_policy_loss = total_policy_loss / total_tokens
        total_kl = total_kl_acc / total_tokens

        loss = total_policy_loss + cfg.beta * total_kl

        # 5. Backprop
        self.optimizer.zero_grad()
        loss.backward()
        grad_ok = True
        all_params = list(self.model.base_model.parameters()) + list(self.model.reward_head.parameters())
        for p in all_params:
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                print("  ⚠ NaN grad detected")
                grad_ok = False
                break
        if grad_ok:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            self.optimizer.step()

        # 6. Logging ── special format
        self._log_step(step, responses, token_rewards, avg_rewards, advantages,
                       total_policy_loss.item(), total_kl.item(), loss.item())

    def _log_step(self, step, responses, token_rew_aligned, avg_rewards,
                  advantages, ploss, kl_val, loss_val):
        lines = []
        lines.append(f"{'='*80}")
        lines.append(f"Step {step}  |  PolicyLoss={ploss:.4f}  KL={kl_val:.4f}  "
                      f"TotalLoss={loss_val:.4f}")
        lines.append(f"{'='*80}")

        for i, resp in enumerate(responses):
            rew = token_rew_aligned[i]
            # tokenise each response word-by-word for display
            tokens = resp.split()
            rew_per_token = rew[:len(tokens)].tolist()

            # ensure equal length
            tk_show = tokens[:len(rew_per_token)]
            rw_show = rew_per_token[:len(tk_show)]

            lines.append(f"── Response {i}  (avg_reward={avg_rewards[i]:.3f}, "
                         f"advantage={advantages[i]:.3f}) ──")

            # [generated text message]
            lines.append(f"[GENERATED TEXT] {resp}")

            # [associated rewards message]
            rw_fmt = " ".join(f"{r:.2f}" for r in rw_show)
            lines.append(f"[REWARDS] {rw_fmt}")

            # [same generated text and rewards aligned token:reward token:reward...]
            aligned = " ".join(f"{t}({r:.2f})" for t, r in zip(tk_show, rw_show))
            lines.append(f"[ALIGNED] {aligned}")

        lines.append("")
        for l in lines:
            self.log(l)
            safe = l.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            try:
                print(safe)
            except UnicodeEncodeError:
                print(safe.encode('ascii', errors='replace').decode('ascii'))

    def train(self, resume_step: int = 0):
        for step in range(resume_step + 1, self.cfg.num_iters + 1):
            self.train_step(step)
            if step % self.cfg.log_interval == 0:
                ckpt = os.path.join(self.cfg.output_dir, f"checkpoint_{step:04d}.pt")
                torch.save({
                    'step': step,
                    'model_state': self.model.base_model.state_dict(),
                    'reward_head': self.model.reward_head.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, ckpt)
                print(f"  → saved {ckpt}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir: str):
    ckpts = []
    if os.path.isdir(output_dir):
        for f in os.listdir(output_dir):
            if f.startswith("checkpoint_") and f.endswith(".pt"):
                try:
                    step = int(f.replace("checkpoint_", "").replace(".pt", ""))
                    ckpts.append((os.path.getmtime(os.path.join(output_dir, f)), step, f))
                except (ValueError, OSError):
                    pass
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return os.path.join(output_dir, ckpts[-1][2]), ckpts[-1][1]

if __name__ == "__main__":
    import sys
    fresh_run = "--fresh" in sys.argv

    cfg = RLTrainerConfig()
    print("Loading model…")
    ir_model = IntrinsicRewardModel(cfg.model_name)
    print(f"Model loaded  |  params: {sum(p.numel() for p in ir_model.base_model.parameters()) / 1e6:.1f}M  |  "
          f"device: {ir_model.device}")

    trainer = GRPOTrainer(ir_model, cfg)

    resume_step = 0
    if not fresh_run:
        resume_info = find_latest_checkpoint(cfg.output_dir)
        if resume_info:
            ckpt_path, resume_step = resume_info
            print(f"Resuming from {ckpt_path} (step {resume_step})…")
            ckpt = torch.load(ckpt_path, map_location=ir_model.device)
            ir_model.base_model.load_state_dict(ckpt['model_state'])
            ir_model.reward_head.load_state_dict(ckpt['reward_head'])
            trainer.optimizer.load_state_dict(ckpt['optimizer'])
            print(f"  → loaded step {resume_step} checkpoint")

    trainer.train(resume_step=resume_step)

    print("\nDone. Log written to", trainer.log_file)
