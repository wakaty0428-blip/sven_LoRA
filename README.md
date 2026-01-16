# SVEN-LoRA: Security Hardening and Adversarial Testing for Code LLMs

SVEN enables controlling code LLMs to generate **secure code** (for security hardening) or **unsafe / vulnerable code** (for adversarial testing), while maintaining functional correctness. The original SVEN achieves this by learning **continuous prompts (prefixes)** with specialized loss terms on a curated security dataset. This repository extends SVEN by additionally supporting **LoRA-based security control** (separate `sec` / `vul` adapters), while keeping the original prefix-based pipeline and evaluation framework. For more technical details, check the original SVEN [paper](https://arxiv.org/abs/2302.05319).


## Directory Structure
The directory structure of this repository is shown as below:
```
.
|-- data_train_val # curated dataset for training and validation (sec / vul)
|-- data_eval # datasets used for security and functional evaluation
|-- sven # SVEN source code (trainer, evaler, loss definitions)
|-- scripts # scripts for training and evaluation
|-- trained # trained prefixes and trained LoRA adapters
|-- experiments # evaluation outputs
```

SVEN currently supports [CodeGen](https://arxiv.org/abs/2203.13474), [InCoder](https://arxiv.org/abs/2204.05999), and [SantaCoder](https://arxiv.org/abs/2301.03988). It should be straightforward to add support for other LLMs (PR welcomed).

## Setup
Set up Python dependencies (a virtual environment is recommended) and [GitHub CodeQL](https://github.com/github/codeql):
```console
$ pip install -r requirements.txt
$ pip install -e .
$ ./setup_codeql.sh
```

## Evaluation
The evaluation consists of two parts: security and functional correctness. You should run the evaluation scripts under the `./scripts` directory. Make sure to use `CUDA_VISIBLE_DEVICES` to select the correct GPUs.

### Evaluation on Security
To evaluate the security of the original LLM, run the command below. The model `350m` can be replaced by {`2b`, `6b`, `incoder`, `santa`}. See `sec_eval.py` for other options, such as using `--temp` to adjust temperature and using `--eval_type` to select the evaluation scenarios.
```console
$ python sec_eval.py --model_type lm --model_dir 350m --output_name sec-eval-350m-lm
```

To evaluate the security of **prefix-based** SVEN using the trained models provided by us, run:
```console
$ python sec_eval.py --model_type prefix --model_dir ../trained/350m-prefix/checkpoint-last --output_name sec-eval-350m-prefix
```

To evaluate the security of **LoRA-based** SVEN, note that sec and vul are separate adapter folders.
Select the branch by pointing --model_dir directly to the adapter directory:
```console
$ python sec_eval.py --model_type lora --model_dir ../trained/350m-lora/checkpoint-last/sec --output_name sec-eval-350m-lora-sec --pretrain_dir 'Salesforce/codegen-350M-multi'
$ python sec_eval.py --model_type lora --model_dir ../trained/350m-lora/checkpoint-last/vul --output_name sec-eval-350m-lora-vul --pretrain_dir 'Salesforce/codegen-350M-multi'
```

Use `print_results.py` to obtain the evaluation results. An example command for the original LLM is:
```console
$ python print_results.py --eval_dir ../experiments/sec_eval/sec-eval-350m-lm
```

Use `make_graph_sr.py` to obtain the bar graphs. An example command for the LoRA-base SVEN is:
```console
$ python make_graph_sr.py --mode lora --lm_txt sr-350m-lm.txt --lora_sec_txt 350m-lr0.0001_r8_lm0.400_con35_kl250-sec.txt --lora_vul_txt 350m-lr0.0001_r8_lm0.400_con35_kl250-vul.txt --out_dir figures_lora --tag 350m-lr0.0001_r8_lm0.400_con35_kl250
```
When it comes to obtain the prefix-based bar graphs, you can run the command below:
```console
$ python make_graph_sr.py --mode prefix --lm_txt sr-350m-lm.txt --prefix_txt 350m-lr0.01_p16_lm0.360_con27_kl370.txt --out_dir figures_prefix --tag 350m-lr0.01_p16_lm0.360_con27_kl370
```

### Evaluation on Functional Correctness
We use [the HumanEval benchmark](https://github.com/openai/human-eval) from [the MultiPL-E framework](https://github.com/nuprl/MultiPL-E/tree/dbcfa139a66cf5e46de798fa5e0854a7f417a046) to evaluate functional correctness. To evaluate the original LLM, run the command below. Check `human_eval_gen.py` for other generation arguments.
```console
$ python human_eval_gen.py --model_type lm --model_dir 350m --output_name human-eval-350m-lm
$ python human_eval_exec.py --output_name human-eval-350m-lm
```

For **prefix-based** SVEN, we need to run the two branches `sec` and `vul` separately via the `--control` argument. The command below is for the `sec` branch:
```console
$ python human_eval_gen.py --model_type prefix --model_dir ../trained/350m-prefix/checkpoint-last --control sec --output_name human-eval-350m-prefix-sec
$ python human_eval_exec.py --output_name human-eval-350m-prefix-sec
```

For **LoRA-based** SVEN, the sec and vul branches are stored as separate adapter directories.
The branch is selected by pointing --model_dir directly to the adapter directory, and the --control argument is not used.
```console
$ python human_eval_gen.py --model_type lora --model_dir ../trained/350m-lora/checkpoint-last/sec --output_name human-eval-350m-lora-sec
$ python human_eval_exec.py --output_name human-eval-350m-lora-sec
```

To view the results (for the original LLM for example), run:
```console
$ python print_results.py --eval_type human_eval --eval_dir ../experiments/human_eval/human-eval-350m-lm
```

## Training
We provide trained security controls in `./trained`, including both **prefix-based** and **LoRA-based** SVEN models.

To train **prefix-based** SVEN, run:
```console
$ python train.py --model_type prefix --output_name 350m-prefix-new --pretrain_dir 350m
```
To train **LoRA-based** SVEN, run:
```console
$ python train.py --output_name 350m-lora-new --pretrain_dir 'Salesforce/codegen-350M-multi' --model_type lora --learning_rate 1e-4 --lora_r 8 --kl_loss_ratio 330 --contrastive_loss_ratio 33 --lm_loss_ratio 0.34 --num_train_epochs 5
```

## Citation
```
@inproceedings{sven-llm,
  author       = {Jingxuan He and Martin Vechev},
  title        = {Large Language Models for Code: Security Hardening and Adversarial Testing},
  booktitle    = {ACM CCS},
  year         = {2023},
  url          = {https://arxiv.org/abs/2302.05319},
}
```