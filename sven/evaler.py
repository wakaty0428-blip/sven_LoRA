import os
import re
import abc
import torch
import numpy as np
import types
import json

from sven.model import CodeGenPrefixCausalLM, load_model
from sven.constant import PROMPTS
from sven.utils import try_parse

from peft import PeftModel, get_peft_model_state_dict, LoraConfig, get_peft_model

class EvalerBase:
    def __init__(self, args):
        self.args = args
        self.load_model()

    @abc.abstractclassmethod
    def load_model(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def sample(self, file_context, func_context, control, lang):
        raise NotImplementedError()

    def truncate(self, completion, lang):
        if lang == 'py':
            for match in re.finditer('\n', completion):
                cur_idx, next_idx = match.start(), match.end()
                if next_idx < len(completion) and not completion[next_idx].isspace():
                    completion = completion[:cur_idx]
                    break
            else:
                last_comment_str = '\n    #'
                if last_comment_str in completion:
                    completion = completion[:completion.rfind(last_comment_str)]
        elif lang == 'c':
            if '\n}' in completion:
                completion = completion[:completion.find('\n}')+2]
            else:
                last_comment_strs = ['\n    //', '\n    /*']
                for last_comment_str in last_comment_strs:
                    if last_comment_str in completion:
                        completion = completion[:completion.rfind(last_comment_str)]
                        completion = completion.rstrip() + '\n}'

            lines = completion.split('\n')
            final_lines = []
            for line in lines:
                if '->name = "' in line: continue
                final_lines.append(line)
            completion = '\n'.join(final_lines)
        else:
            raise NotImplementedError()

        return completion

    def process_completions(self, input_src, input_ids_len, gen_output, lang):
        tokens = gen_output[:, input_ids_len:, ...]
        completions = self.tokenizer.batch_decode(tokens)

        output_srcs, output_ids = [], []
        dup_srcs, non_parsed_srcs = [], []
        for i, completion in enumerate(completions):
            if self.tokenizer.eos_token in completion:
                completion = completion[:completion.find(self.tokenizer.eos_token)]
            completion = self.truncate(completion, lang)
            completion_len = len(self.tokenizer.encode(completion))
            output_src = input_src + completion
            output_src = output_src.rstrip() + '\n'
            if output_src in output_srcs:
                dup_srcs.append(output_src)
            elif try_parse(output_src, lang) != 0:
                non_parsed_srcs.append(output_src)
            else:
                output_srcs.append(output_src)
                output_ids.append((gen_output[i][:input_ids_len].tolist(), gen_output[i][input_ids_len:input_ids_len+completion_len].tolist()))

        return output_srcs, output_ids, dup_srcs, non_parsed_srcs

class LMEvaler(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        self.tokenizer, self.model, self.input_device = load_model('lm', self.args.model_dir, False, self.args)
        self.model.eval()

    def sample(self, file_context, func_context, control, lang):
        input_src = file_context + func_context
        input_ids = self.tokenizer(input_src, return_tensors='pt').input_ids.to(self.input_device)
        input_ids_len = input_ids.shape[1]
        gen_output = self.model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=self.args.num_gen,
            temperature=self.args.temp,
            max_new_tokens=self.args.max_gen_len,
            top_p=self.args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            # return_dict_in_generate=True,
            # output_scores=True,
        )
        return self.process_completions(input_src, input_ids_len, gen_output, lang)

class PrefixEvaler(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        self.tokenizer, self.model, self.input_device = load_model('prefix', self.args.model_dir, False, self.args)

        self.model.eval()

    def sample(self, file_context, func_context, control, lang):
        return self.sample_prefix(file_context, func_context, control, lang)

    def sample_prefix(self, file_context, func_context, control, lang):
        input_src = file_context + func_context
        input_ids = self.tokenizer(input_src, return_tensors='pt').input_ids.to(self.input_device)
        input_ids_len = input_ids.shape[1]
        gen_output = self.model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=self.args.num_gen,
            temperature=self.args.temp,
            max_new_tokens=self.args.max_gen_len,
            top_p=self.args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            control_id=control,
            # return_dict_in_generate=True,
            # output_scores=True,
        )
        return self.process_completions(input_src, input_ids_len, gen_output, lang)

# lora version
# class LoraEvaler(EvalerBase):
#     def __init__(self, args):
#         super().__init__(args)


#     def load_model(self):
#         # -------- 1) Load base LM from pretrain_dir --------
#         self.tokenizer, base_model, self.input_device = load_model(
#             'lora',
#             self.args.pretrain_dir,
#             False,      
#             self.args
#         )

#         ### DEBUG START: base model + tokenizer ###
#         print("\n========== DEBUG: LORA EVAL BASE MODEL ==========")
#         print("[ARGS] pretrain_dir         =", self.args.pretrain_dir)
#         print("[ARGS] model_dir            =", self.args.model_dir)
#         print("[BASE MODEL] name_or_path   =", getattr(base_model.config, "_name_or_path", None))
#         print("[TOKENIZER] name_or_path    =", getattr(self.tokenizer, "name_or_path", None))
#         print("[TOKENIZER] vocab_size      =", getattr(self.tokenizer, "vocab_size", None))
#         print("=================================================\n")
#         ### DEBUG END ###

#         adapter_root = self.args.model_dir
#         sec_path = os.path.join(adapter_root, "sec")
#         vul_path = os.path.join(adapter_root, "vul")

#         # =============================================================
#         # 2) Load SEC adapter FIRST — this defines the correct LoRA arch
#         # =============================================================
#         if not os.path.isdir(sec_path):
#             raise ValueError(f"[ERROR] SEC adapter not found at {sec_path}")

#         print("### FIX: Initializing PeftModel from SEC adapter...")
#         self.model = PeftModel.from_pretrained(
#             base_model,
#             sec_path,
#             adapter_name="sec",
#             is_trainable=False
#         )
#         print("### FIX: Loaded SEC adapter successfully.")

#         # =============================================================
#         # 3) Load VUL adapter into the same PEFT wrapper
#         # =============================================================
#         if os.path.isdir(vul_path):
#             print("### FIX: Loading VUL adapter...")
#             self.model.load_adapter(vul_path, adapter_name="vul", is_trainable=False)
#             print("### FIX: Loaded VUL adapter successfully.")
#         else:
#             print("### WARNING: VUL adapter not found:", vul_path)

#         # =============================================================
#         # >>> INSERT DIAGNOSTIC LORA MODULE CHECK HERE <<<
#         # =============================================================
#         print("\n========== CHECKING FOR INJECTED LORA MODULES ==========")
#         found_any = False
#         for name, module in self.model.named_modules():
#             if "lora" in name.lower():
#                 print("FOUND LORA MODULE:", name)
#                 found_any = True

#         if not found_any:
#             print("❌ NO LoRA MODULES FOUND — LoRA WAS NOT INJECTED INTO CODEGEN")
#         else:
#             print("✅ LoRA modules detected — injection succeeded")
#         print("=========================================================\n")
        
#         print("\n========== CHECKING LORA WEIGHT DIFFERENCES ==========")

#         for layer in range(20):
#             prefix = f"base_model.model.transformer.h.{layer}.attn.qkv_proj"

#             A_sec = dict(self.model.named_parameters())[f"{prefix}.lora_A.sec.weight"]
#             A_vul = dict(self.model.named_parameters())[f"{prefix}.lora_A.vul.weight"]

#             diff = torch.abs(A_sec - A_vul).mean().item()
#             print(f"Layer {layer}: mean |A_sec - A_vul| = {diff:.6f}")

#         # =============================================================
#         # 3b) Check SEC vs VUL actually change the *outputs*
#         # =============================================================

#         print("========== CHECKING LORA OUTPUT DIFFERENCE ON DUMMY PROMPT ==========")
#         dummy_src = "int add(int a, int b) {\n    return a + b;\n}"
#         dummy_ids = self.tokenizer(dummy_src, return_tensors="pt").input_ids.to(self.input_device)

#         with torch.no_grad():
#             # SEC adapter
#             self.model.set_adapter("sec")
#             out_sec = self.model(input_ids=dummy_ids).logits

#             # VUL adapter
#             self.model.set_adapter("vul")
#             out_vul = self.model(input_ids=dummy_ids).logits

#         diff_logits = (out_sec - out_vul).abs().mean().item()
#         print(f"Mean |logits_sec - logits_vul| = {diff_logits:.6f}")
#         print("Logits shape:", out_sec.shape)
#         print("=========================================================\n")

#         # =============================================================
#         # EXTRA TEST: Check if logits are non-zero and not degenerate
#         # =============================================================
#         print("===== EXTRA LOGIT MAGNITUDE TEST =====")
#         print("Sum |logits_sec| =", out_sec.abs().sum().item())
#         print("Sum |logits_vul| =", out_vul.abs().sum().item())
#         print("Sum |logits_sec - logits_vul| =", (out_sec - out_vul).abs().sum().item())
#         print("=======================================\n")



        
        
#         # =============================================================
#         # 4) Set default adapter to SEC
#         # =============================================================
#         self.model.set_adapter("sec")
#         print("### Activated SEC adapter")

#         # =============================================================
#         # Debug – confirm internal PEFT state
#         # =============================================================
#         print("\n========== DEBUG: PEFT ADAPTER CONFIG ==========")
#         for name, cfg in self.model.peft_config.items():
#             print(f"Adapter '{name}':", cfg)

#         # We expect CodeGen to not expose full active adapter info,
#         # but switching WILL work internally.
#         try:
#             print("Active adapters:", self.model.get_active_adapters())
#         except Exception:
#             print("Active adapters: NOT SUPPORTED by CodeGen model")

#         print("=================================================\n")

#         # ========== DEBUG: Check adapter names on qkv_proj ==========
#         qkv = self.model.base_model.model.transformer.h[0].attn.qkv_proj

#         print("\n=========== ADAPTER DEBUG ===========")
#         print("ACTIVE ADAPTER =", getattr(qkv, "active_adapter", None))

#         if hasattr(qkv, "lora_A"):
#             print("Available lora_A keys:", list(qkv.lora_A.keys()))
#         else:
#             print("No lora_A found")

#         if hasattr(qkv, "lora_B"):
#             print("Available lora_B keys:", list(qkv.lora_B.keys()))
#         else:
#             print("No lora_B found")
#         print("=====================================\n")
#         # ============================================================

#         # ================== VERIFY PATCH ==================
#         print("===== VERIFY ACTIVE_ADAPTER PROPAGATION =====")
#         qkv = self.model.base_model.model.transformer.h[0].attn.qkv_proj

#         self.model.set_adapter("sec")
#         print("After setting SEC, qkv.active_adapter =", getattr(qkv, "active_adapter", None))

#         self.model.set_adapter("vul")
#         print("After setting VUL, qkv.active_adapter =", getattr(qkv, "active_adapter", None))
#         print("==============================================\n")
#         # ==================================================


        
#         self.model.eval()


#     def sample(self, file_context, func_context, control, lang):

#         if isinstance(control, int):
#             if control == 0:
#                 control = "sec"
#             elif control == 1:
#                 control = "vul"
#             else:
#                 raise ValueError(f"Invalid control id: {control}")

#         if control == "sec":
#             self.model.set_adapter("sec")
#         elif control == "vul":
#             self.model.set_adapter("vul")
#         else:
#             raise ValueError(f"Unknown control: {control}")
        
#         ### DEBUG START: active adapter before generation ###
#         active_attr = getattr(self.model, "active_adapter", None)
#         get_active = getattr(self.model, "get_active_adapters", None)
#         if callable(get_active):
#             try:
#                 active_list = get_active()
#             except TypeError:
#                 active_list = None
#         else:
#             active_list = None

#         print(f"### DEBUG SAMPLE ### control={control}, active_adapter={active_attr}, get_active_adapters()={active_list}")
#         print(f"[SAMPLE] active adapter after set: {self.model.base_model.model.transformer.h[0].attn.qkv_proj.active_adapter}")

#         ### DEBUG END ###
        
#         input_src = file_context + func_context
#         input_ids = self.tokenizer(input_src, return_tensors='pt').input_ids.to(self.input_device)
#         input_ids_len = input_ids.shape[1]


#         gen_output = self.model.generate(
#             input_ids=input_ids,
#             do_sample=True,
#             num_return_sequences=self.args.num_gen,
#             temperature=self.args.temp,
#             max_new_tokens=self.args.max_gen_len,
#             top_p=self.args.top_p,
#             pad_token_id=self.tokenizer.pad_token_id,
#             use_cache=True,
#         )
#         return self.process_completions(input_src, input_ids_len, gen_output, lang)

class LoraEvaler(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        """
        Load ONLY ONE LoRA adapter manually.
        Fixes:
          - lora_B loading as zeros
          - .vul → .default remapping
        """

        # -------------------------------------------------------
        # Load tokenizer + BASE LM
        # -------------------------------------------------------
        print("=== Loading BASE LM ===")
        self.tokenizer, base_model, self.input_device = load_model(
            "lora",
            self.args.pretrain_dir,
            False,
            self.args
        )

        # -------------------------------------------------------
        # Determine adapter name by folder
        # -------------------------------------------------------
        adapter_path = self.args.model_dir.rstrip("/")
        print(f"=== Loading adapter from: {adapter_path}")

        # -------------------------------------------------------
        # MANUAL LOAD OF LORA CONFIG (FIX)
        # -------------------------------------------------------
        # <<< FIX 1: load config manually
        cfg_path = os.path.join(adapter_path, "adapter_config.json")
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        lora_cfg = LoraConfig(**cfg_dict)

        # Create a single-adapter model
        # <<< FIX 2: always create adapter slot named "default"
        model = get_peft_model(base_model, lora_cfg)

        # -------------------------------------------------------
        # MANUAL LOAD OF WEIGHTS (FIX)
        # -------------------------------------------------------
        state_path = os.path.join(adapter_path, "adapter_model.bin")
        print(f"=== Manual loading LoRA weights from: {state_path}")
        state = torch.load(state_path, map_location="cpu")

        tags = ["sec", "vul", "default"]
        detected_tags = set()
        for k in state.keys():
            if "lora_A" in k or "lora_B" in k:
                for t in tags:
                    if f".{t}." in k:
                        detected_tags.add(t)

        print("=== Detected adapter tags in checkpoint:", detected_tags or "NONE")

        # <<< FIX 3: remap keys if VUL
        remapped_state = {}
        for k, v in state.items():
            new_k = k
            # For any tag we see, normalize to '.default.'
            for t in detected_tags:
                new_k = new_k.replace(f".{t}.", ".default.")
            remapped_state[new_k] = v


        missing, unexpected = model.load_state_dict(remapped_state, strict=False)
        # print("[DEBUG] Missing keys:", missing)
        # print("[DEBUG] Unexpected keys:", unexpected)

        # -------------------------------------------------------
        # Finalize
        # -------------------------------------------------------
        print("=== Adapter loaded successfully ===")
        self.model = model
        self.model.eval()
        self.model.to(self.input_device)

    # ============================================================
    # Sampling (no switching!)
    # ============================================================
    def sample(self, file_context, func_context, control, lang):
        """
        Option A:
        control is ignored because we evaluate ONLY ONE model per run:
          - If user loads sec adapter, they call sec_eval just once.
          - If user loads vul adapter, they call sec_eval again separately.

        No self.model.set_adapter() call needed.
        """

        input_src = file_context + func_context
        input_ids = self.tokenizer(input_src, return_tensors="pt").input_ids.to(self.input_device)
        input_ids_len = input_ids.shape[1]

        gen_output = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            num_return_sequences=self.args.num_gen,
            temperature=self.args.temp,
            max_new_tokens=self.args.max_gen_len,
            top_p=self.args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )

        return self.process_completions(input_src, input_ids_len, gen_output, lang)

class TextPromptEvaler(EvalerBase):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        self.tokenizer, self.model, self.input_device = load_model('lm', self.args.model_dir, False, self.args)
        self.model.eval()

    def sample(self, file_context, func_context, control, lang):
        if lang == 'py':
            input_src = file_context + '# ' + PROMPTS[control] + func_context
        elif lang == 'c':
            input_src = file_context + '// ' + PROMPTS[control] + func_context
        else:
            raise NotImplementedError()
        input_ids = self.tokenizer(input_src, return_tensors='pt').input_ids.to(self.input_device)
        input_ids_len = input_ids.shape[1]
        gen_output = self.model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=self.args.num_gen,
            temperature=self.args.temp,
            max_new_tokens=self.args.max_gen_len,
            top_p=self.args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        return self.process_completions(input_src, input_ids_len, gen_output, lang)
