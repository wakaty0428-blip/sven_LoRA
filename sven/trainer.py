import os
import abc
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from sven.model import save_model, parallelize_model, load_model
from sven.dataset import PrefixDataset, TextPromptDataset
from sven.utils import set_seed

from peft import LoraConfig, get_peft_model, PeftModel

# === ADD THIS UTILITY FUNCTION ANYWHERE ABOVE LoraTrainer CLASS ===

def check_lora_updates(model, prev_lora_state):
    """Returns string of parameter diffs and updates prev_lora_state."""
    diffs = []
    new_state = {}

    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            new_state[name] = p.detach().clone()
            diff = (p - prev_lora_state[name]).abs().mean().item()
            diffs.append(f"{name}: {diff:.6f}")

    return "\n".join(diffs), new_state


class TrainerBase:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.dataset = None
        self.input_device = None

    @abc.abstractclassmethod
    def load_model(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def load_dataset(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def step(self, batch):
        raise NotImplementedError()

    def save(self, path, step, epoch, optimizer, scheduler):
        if not os.path.exists(path):
            os.makedirs(path)

        # ==========================
        # LoRA: save SEC and VUL separately
        # ==========================
        # if getattr(self.args, "model_type", None) == "lora" and isinstance(self.model, PeftModel):
        #     # 1) Save tokenizer to the checkpoint root
        #     self.tokenizer.save_pretrained(path)

        #     # 2) Save base LM name so we can reload correctly
        #     lm_path_file = os.path.join(path, "lm.txt")
        #     with open(lm_path_file, "w") as f:
        #         f.write(self.args.pretrain_dir)

        #     # 3) Save SEC adapter ("default") into ./sec
        #     sec_dir = os.path.join(path, "sec")
        #     if not os.path.exists(sec_dir):
        #         os.makedirs(sec_dir)
        #     # Only save the "default" adapter weights
        #     self.model.save_pretrained(sec_dir, adapter_name="default")

        #     # 4) Save VUL adapter ("vul") into ./vul
        #     vul_dir = os.path.join(path, "vul")
        #     if not os.path.exists(vul_dir):
        #         os.makedirs(vul_dir)
        #     self.model.save_pretrained(vul_dir, adapter_name="vul")

        # else:
            # ==========================
            # Non-LoRA: original behavior
            # ==========================
        save_model(self.model, path, self.args)
        self.tokenizer.save_pretrained(path)

        # ======================================================
        # DEBUG: Inspect SAVED adapter weight file
        # ======================================================
        # if isinstance(self.model, PeftModel):
        #     sec_file = os.path.join(path, "sec", "adapter_model.bin")
        #     vul_file = os.path.join(path, "vul", "adapter_model.bin")

        #     if os.path.exists(sec_file):
        #         sd = torch.load(sec_file)
        #         print("\n[DEBUG] --- SAVED SEC adapter_model.bin ---")
        #         for k, v in sd.items():
        #             print(" ", k, "| sum(abs) =", v.abs().sum().item())

        #     if os.path.exists(vul_file):
        #         sd = torch.load(vul_file)
        #         print("\n[DEBUG] --- SAVED VUL adapter_model.bin ---")
        #         for k, v in sd.items():
        #             print(" ", k, "| sum(abs) =", v.abs().sum().item())
        # # ======================================================


        step_file = os.path.join(path, 'step_file.txt')
        with open(step_file, 'w') as f:
            f.write(str(step)+'\n')
        epoch_file = os.path.join(path, 'epoch_file.txt')
        with open(epoch_file, 'w') as f:
            f.write(str(epoch)+'\n')

        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(path, 'scheduler.pt'))

    def add_to_loss_dict(self, acc_loss_dict, loss_dict):
        for key, val in loss_dict.items():
            if key not in acc_loss_dict:
                acc_loss_dict[key] = 0.0
            acc_loss_dict[key] += val

    def report_loss_dict(self, loss_dict, steps):
        ss = []
        for key, val in loss_dict.items():
            if key == 'kl_loss':
                r = 8
            else:
                r = 4
            ss.append(f'{key}: {round(val/steps, r)}')
        return ', '.join(ss)

    def run(self):
        self.load_model()
        self.load_dataset()

        self.args.logger.info(f'Training args {self.args}')

        batch_size = 1
        train_sampler = RandomSampler(self.dataset)
        train_dataloader = DataLoader(self.dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)

        total_samples = len(self.dataset)
        batch_size = batch_size * self.args.grad_acc_steps
        total_steps = total_samples // batch_size * self.args.num_train_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
            'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                num_training_steps=total_steps)

        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.args.logger.info('***** Running training *****')
        self.args.logger.info('  Num samples = %d', total_samples)
        self.args.logger.info('  Num epoch = %d', self.args.num_train_epochs)
        self.args.logger.info('  Batch size= 1')
        self.args.logger.info('  Total batch size (w. accumulation) = %d', batch_size)
        self.args.logger.info('  Gradient Accumulation steps = %d', self.args.grad_acc_steps)
        self.args.logger.info('  Total optimization steps = %d', total_steps)
        self.args.logger.info('  Num val samples = %d', len(self.val_dataset))
        self.args.logger.info('  Num parameters = %d', num_params)
        self.args.logger.info('  Num trainable parameters = %d', num_trainable_params)
        self.args.logger.info('  Fraction of trainable parameters = %s', str(round(num_trainable_params/num_params*100, 4)))

        global_step, acc_loss_dict = 0, OrderedDict()
        set_seed(self.args)
        self.model.train()
        for idx in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                loss, loss_dict = self.step(batch)
                if self.args.grad_acc_steps > 1:
                    loss = loss / self.args.grad_acc_steps
                    for key in loss_dict:
                        loss_dict[key] = loss_dict[key] / self.args.grad_acc_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.add_to_loss_dict(acc_loss_dict, loss_dict)

                if (step+1) % self.args.grad_acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()  
                    global_step += 1

                    # ==== LORA UPDATE CHECK ====
                    if isinstance(self, LoraTrainer) and global_step % 100 == 0:
                        # print("\n===== LORA PARAM UPDATE CHECK =====")
                        diff_text, self.prev_lora_state = check_lora_updates(self.model, self.prev_lora_state)
                        # print(diff_text)
                        # print("====================================\n")


                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        reported_loss = self.report_loss_dict(acc_loss_dict, self.args.logging_steps)
                        self.args.logger.info('epochs: %s/%d, steps: %s/%d, %s', idx+1, int(self.args.num_train_epochs), global_step, total_steps, reported_loss)
                        acc_loss_dict.clear()

            if self.args.save_epochs > 0 and (idx+1) % self.args.save_epochs == 0:
                self.model.eval()
                with torch.no_grad():
                    reported_eval_loss = self.do_eval()
                self.model.train()
                self.args.logger.info('val epoch %s: %s', idx+1, reported_eval_loss)
                output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
                last_output_dir = os.path.join(self.args.output_dir, f'checkpoint-last')
                # # ======================================================
                # # DEBUG: LoRA parameters AFTER TRAINING (before saving)
                # # ======================================================
                # print("\n[DEBUG] ===== LoRA parameters AFTER TRAINING =====")
                # for name, p in self.model.named_parameters():
                #     if "lora_" in name:
                #         print("  ", name, "| sum(abs) =", p.detach().abs().sum().item())
                # print("[DEBUG] ===== END AFTER TRAINING =====\n")
                # # ======================================================
                self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
                self.save(output_dir, global_step, idx+1, None, None)
                self.save(last_output_dir, global_step, idx+1, None, None)

        if (idx+1) % self.args.save_epochs != 0:
            self.model.eval()
            with torch.no_grad():
                reported_eval_loss = self.do_eval()
            self.args.logger.info('final eval loss: %s', reported_eval_loss)
            output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
            last_output_dir = os.path.join(self.args.output_dir, f'checkpoint-last')
            # # ======================================================
            # # DEBUG: LoRA parameters AFTER TRAINING (before saving)
            # # ======================================================
            # print("\n[DEBUG] ===== LoRA parameters AFTER TRAINING =====")
            # for name, p in self.model.named_parameters():
            #     if "lora_" in name:
            #         print("  ", name, "| sum(abs) =", p.detach().abs().sum().item())
            # print("[DEBUG] ===== END AFTER TRAINING =====\n")
            # # ======================================================
            self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
            self.save(output_dir, global_step, idx+1, None, None)
            self.save(last_output_dir, global_step, self.args.num_train_epochs, None, None)

    def do_eval(self):
        val_sampler = SequentialSampler(self.val_dataset)
        val_dataloader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=1)
        acc_loss_dict = OrderedDict()
        for batch in val_dataloader:
            loss, loss_dict = self.step(batch)
            self.add_to_loss_dict(acc_loss_dict, loss_dict)
        return self.report_loss_dict(acc_loss_dict, len(val_dataloader))

def get_logits_from_lm(lm, inputs, control_ids):
    if control_ids is not None:
        past = lm.get_past_from_prefix(control_ids)
    else:
        past = None
    outputs = lm(inputs, past_key_values=past)
    shift_logits = outputs.logits[..., :-1, :]
    shift_labels = inputs[..., 1:].unsqueeze(-1)
    shift_probs = F.softmax(shift_logits, dim=-1)
    return shift_logits.squeeze(0), torch.gather(shift_probs, 2, shift_labels).squeeze(-1).squeeze(0)

# lora version
def get_logits_from_lora_lm(model, inputs):

    outputs = model(inputs)
    shift_logits = outputs.logits[..., :-1, :]
    shift_labels = inputs[..., 1:].unsqueeze(-1)
    shift_probs = F.softmax(shift_logits, dim=-1)
    return shift_logits.squeeze(0), torch.gather(shift_probs, 2, shift_labels).squeeze(-1).squeeze(0)

def token_weighted_loss(loss_type, inputs, targets, weights):
    # ==== LOSS DEBUG ====
    # print("[LOSS DEBUG] inputs.shape:", inputs.shape)
    # print("[LOSS DEBUG] targets.shape:", targets.shape)
    # print("[LOSS DEBUG] weights.shape:", weights.shape)
    # ====================

    
    if loss_type == 'cross_entropy':
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    elif loss_type == 'nll':
        loss_fct = torch.nn.NLLLoss(reduction='none')
    elif loss_type == 'kl':
        loss_fct = torch.nn.KLDivLoss(log_target=True, reduction='none')
    else:
        assert False

    loss = loss_fct(inputs, targets)
    if loss_type == 'kl':
        loss = loss.sum(dim=1)
    loss = loss[weights != 0]
    return loss.mean()

class PrefixTrainer(TrainerBase):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        self.tokenizer, self.model, self.input_device = load_model('prefix', self.args.pretrain_dir, True, self.args)

        for n, p in self.model.named_parameters():
            if n.startswith('prefix_params'):
                p.requires_grad = True
            else:
                p.requires_grad = False
        self.model.train()

    def load_dataset(self):
        self.dataset = PrefixDataset(self.args, self.tokenizer, 'train')
        self.val_dataset = PrefixDataset(self.args, self.tokenizer, 'val')

    def step(self, batch):
        return_dict = OrderedDict()
        inputs, weights, control_ids, _ = batch
        inputs = inputs.to(self.input_device)
        shift_inputs = inputs[..., 1:].squeeze(0)
        weights = weights.to(self.input_device)
        shift_weights = weights[..., 1:].squeeze(0)
        control_ids = control_ids.to(self.input_device)

        correct_logits, correct_label_probs = get_logits_from_lm(self.model, inputs, control_ids)
        lm_loss = token_weighted_loss('cross_entropy', correct_logits, shift_inputs, shift_weights)
        lm_loss *= self.args.lm_loss_ratio
        return_dict['lm_loss'] = lm_loss.item()

        if self.args.contrastive_loss_ratio != 0 or self.args.kl_loss_ratio != 0:
            incorrect_control_ids = -1 * (control_ids - 1)
            incorrect_logits, incorrect_label_probs = get_logits_from_lm(self.model, inputs, incorrect_control_ids)

            contrastive_loss = 0
            if self.args.contrastive_loss_ratio != 0:
                contrastive_probs = torch.stack((correct_label_probs, incorrect_label_probs), dim=1)
                contrastive_probs = F.normalize(contrastive_probs, p=1, dim=-1)
                contrastive_log_probs = torch.log(contrastive_probs)
                contrastive_labels = torch.zeros(shift_inputs.shape, dtype=torch.int64).to(self.input_device)
                contrastive_loss = token_weighted_loss('nll', contrastive_log_probs, contrastive_labels, shift_weights)
                contrastive_loss *= self.args.contrastive_loss_ratio / 100
                return_dict['contrastive_loss'] = contrastive_loss.item()

            kl_loss = 0
            if self.args.kl_loss_ratio != 0:
                correct_log_probs = F.log_softmax(correct_logits, dim=-1)
                self.model.eval()
                with torch.no_grad():
                    ref_logits, _ = get_logits_from_lm(self.model, inputs, None)
                self.model.train()
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                kl_loss += token_weighted_loss('kl', correct_log_probs, ref_log_probs, 1-shift_weights)
                incorrect_log_probs = F.log_softmax(incorrect_logits, dim=-1)
                kl_loss += token_weighted_loss('kl', incorrect_log_probs, ref_log_probs, 1-shift_weights)
                kl_loss = kl_loss * self.args.kl_loss_ratio / 1000
                return_dict['kl_loss'] = kl_loss.item()

        loss = lm_loss + contrastive_loss + kl_loss
        return_dict['loss'] = loss.item()
        return loss, return_dict

class TextPromptTrainer(TrainerBase):

    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        self.tokenizer, self.model, self.input_device = load_model('lm', self.args.pretrain_dir, True, self.args)
        self.model.train()

    def load_dataset(self):
        self.dataset = TextPromptDataset(self.args, self.tokenizer, 'train')
        self.val_dataset = TextPromptDataset(self.args, self.tokenizer, 'val')

    def step(self, batch):
        inputs, labels= batch
        inputs = inputs.to(self.input_device)
        labels = labels.to(self.input_device)
        outputs = self.model(inputs, labels=labels)
        loss = outputs.loss
        return loss, {'loss': loss.item()}

# lora version
class LoraTrainer(TrainerBase):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        self.tokenizer, base_model, self.input_device = load_model(
            'lora', self.args.pretrain_dir, True, self.args
        )

        # convert "q_proj,v_proj" into ["q_proj", "v_proj"]
        target_modules_str = getattr(self.args, "lora_target_modules", None)
        if target_modules_str is None:
            # Fallback: default to qkv_proj (works for CodeGen-350M)
            target_modules = ["qkv_proj"]
        else:
            # Split "q_proj,v_proj" -> ["q_proj", "v_proj"], removing blanks
            target_modules = [
                m.strip()
                for m in target_modules_str.split(",")
                if m.strip()
            ]

        print(f"[LoRA Trainer] target_modules = {target_modules}")

        lora_cfg = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # ---- Secure adapter (= control_id 0) ----
        model = get_peft_model(base_model, lora_cfg)

        # =============================================
        # DEBUG: CHECK LORA PARAMETER SHAPES AND GRAD
        # =============================================
        # print("\n[DEBUG] After get_peft_model() — LoRA parameters:")
        # for name, p in model.named_parameters():
        #     if "lora_" in name:
        #         print("  ", name, "| shape:", tuple(p.shape), "| requires_grad:", p.requires_grad)
        # print("=============================================\n")


        # 5) Add vulnerable adapter "vul" with the same config
        if "vul" not in getattr(model, "peft_config", {}):
            model.add_adapter("vul", lora_cfg)
        # =============================================
        # DEBUG: CHECK LORA PARAMETERS AFTER VUL ADAPTER
        # =============================================
        # print("\n[DEBUG] After add_adapter('vul') — LoRA parameters:")
        # for name, p in model.named_parameters():
        #     if "lora_" in name:
        #         print("  ", name, "| shape:", tuple(p.shape), "| requires_grad:", p.requires_grad)
        # print("=============================================\n")


        # print("[PATCH] Breaking symmetry between SEC(default) and VUL adapters...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                # Touch only VUL LoRA weights
                if "lora_" in name and ".vul." in name:
                    # Option 1: add random noise
                    param.add_(0.01 * torch.randn_like(param))
        
        # 6) Use "default" as the secure adapter during training
        #    (other parts of the code will call set_adapter(...) as needed)

        # base LMをfreezeし、LoRA部分だけ学習
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # print("[LoRA] Trainable parameters:")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print("  trainable:", name)
            
        model.set_adapter("default")
        
        self.model = model
        # base_model はすでに load_model 内で parallelize 済みなので再 parallelize は不要
        self.model.train()

        # Save initial LoRA parameters for update checking
        self.prev_lora_state = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if "lora_" in name
        }
        # print("[DEBUG] Saved initial LoRA weights.")
        # print("\n[DEBUG] ===== LoRA_B values right after training START =====")
        # for name, p in self.model.named_parameters():
        #     if "lora_B.default" in name or "lora_B.vul" in name:
        #         print(name, p.abs().sum().item())
        # print("[DEBUG] ===== LoRA_B values right after training END =====\n")

    def load_dataset(self):
        # Prefix と同じ Dataset を使用
        self.dataset = PrefixDataset(self.args, self.tokenizer, 'train')
        self.val_dataset = PrefixDataset(self.args, self.tokenizer, 'val')

    def _set_adapter_for_control(self, control_id_int: int, correct: bool):
        """
        control_id_int: 0 or 1
        correct=True  -> 正しいアダプタ (secure or vulnerable)
        correct=False -> 間違ったアダプタ (逆側)
        adapter 名:
          - "default": secure 用 (control_id == 0)
          - "vul":     vulnerable 用 (control_id == 1)
        """
        if control_id_int == 0:
            correct_adapter = "default"
            incorrect_adapter = "vul"
        else:
            correct_adapter = "vul"
            incorrect_adapter = "default"

        self.model.set_adapter(correct_adapter if correct else incorrect_adapter)

    def step(self, batch):
        return_dict = OrderedDict()
        inputs, weights, control_ids, _ = batch
        inputs = inputs.to(self.input_device)
        shift_inputs = inputs[..., 1:].squeeze(0)
        weights = weights.to(self.input_device)
        shift_weights = weights[..., 1:].squeeze(0)
        control_ids = control_ids.to(self.input_device)

        # control_ids はバッチサイズ1想定なので 0/1 のスカラーに変換
        control_id_int = int(control_ids.item())
        # print(f"[DEBUG] control_id_int = {control_id_int}")


        # 1) 正しいアダプタでの出力
        self._set_adapter_for_control(control_id_int, correct=True)
        # print(f"[DEBUG] Using CORRECT adapter = {self.model.active_adapter}")
        correct_logits, correct_label_probs = get_logits_from_lora_lm(self.model, inputs)
        # ==== DEBUG SHAPES ====
        # print("DEBUG correct_logits shape:", correct_logits.shape)
        # print("DEBUG shift_inputs shape:", shift_inputs.shape)
        # print("DEBUG shift_weights shape:", shift_weights.shape)
        # ======================

        lm_loss = token_weighted_loss('cross_entropy', correct_logits, shift_inputs, shift_weights)
        lm_loss *= self.args.lm_loss_ratio
        return_dict['lm_loss'] = lm_loss.item()

        # 2) 間違ったアダプタでの出力（逆側）
        if self.args.contrastive_loss_ratio != 0 or self.args.kl_loss_ratio != 0:
            self._set_adapter_for_control(control_id_int, correct=False)
            # print(f"[DEBUG] Using INCORRECT adapter = {self.model.active_adapter}")
            incorrect_logits, incorrect_label_probs = get_logits_from_lora_lm(self.model, inputs)

            contrastive_loss = 0
            if self.args.contrastive_loss_ratio != 0:
                contrastive_probs = torch.stack((correct_label_probs, incorrect_label_probs), dim=1)
                contrastive_probs = F.normalize(contrastive_probs, p=1, dim=-1)
                contrastive_log_probs = torch.log(contrastive_probs)
                contrastive_labels = torch.zeros(shift_inputs.shape, dtype=torch.int64).to(self.input_device)
                contrastive_loss = token_weighted_loss('nll', contrastive_log_probs, contrastive_labels, shift_weights)
                contrastive_loss *= self.args.contrastive_loss_ratio / 100
                return_dict['contrastive_loss'] = contrastive_loss.item()

            kl_loss = 0
            if self.args.kl_loss_ratio != 0:
                correct_log_probs = F.log_softmax(correct_logits, dim=-1)

                # base LM（LoRA無し）を参照分布として利用
                self.model.eval()
                with torch.no_grad():
                    base_model = self.model.get_base_model()
                    base_outputs = base_model(inputs)
                    ref_logits = base_outputs.logits[..., :-1, :]
                self.model.train()

                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_log_probs = ref_log_probs.squeeze(0)   # make it [787, vocab]

                kl_loss += token_weighted_loss('kl', correct_log_probs, ref_log_probs, 1 - shift_weights)

                incorrect_log_probs = F.log_softmax(incorrect_logits, dim=-1)
                kl_loss += token_weighted_loss('kl', incorrect_log_probs, ref_log_probs, 1 - shift_weights)
                kl_loss = kl_loss * self.args.kl_loss_ratio / 1000
                return_dict['kl_loss'] = kl_loss.item()

        loss = lm_loss + contrastive_loss + kl_loss
        return_dict['loss'] = loss.item()
        return loss, return_dict



