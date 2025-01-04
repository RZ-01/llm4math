import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train or Inference for LoRA fine-tuned model.")

    subparsers = parser.add_subparsers(dest="mode", help="Mode: train or inference", required=True)
    # 训练模式参数
    train_parser = subparsers.add_parser("train", help="Training mode")
    # Model and data
    train_parser.add_argument("--model_id", type=str, default="google/gemma-2-9b-it", help="Pretrained model ID.")
    train_parser.add_argument("--dataset_dir", type=str, default="../extracted/", help="Directory with dataset files.")

    # Training hyperparameters
    train_parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    train_parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    train_parser.add_argument("--learning_rate", type=float, default=2e-6)
    train_parser.add_argument("--weight_decay", type=float, default=1e-2)
    train_parser.add_argument("--max_grad_norm", type=float, default=1.0)
    train_parser.add_argument("--num_train_steps", type=int, default=100000)
    train_parser.add_argument("--warmup_steps", type=int, default=1000)
    train_parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    train_parser.add_argument("--max_seq_length", type=int, default=512) #1024

    # Quantization and LoRA config
    train_parser.add_argument("--quantization_bits", type=int, default=4)
    train_parser.add_argument("--lora_r", type=int, default=128)
    train_parser.add_argument("--lora_alpha", type=int, default=32)
    train_parser.add_argument("--lora_dropout", type=float, default=0.01)

    # Logging and output
    train_parser.add_argument("--project_name", type=str, default="gemma_sft_project_lora_mistral")
    train_parser.add_argument("--run_name", type=str, default="gemma_train_run_9b_A6000")
    train_parser.add_argument("--output_dir", type=str, default="./results")

    # inference parameters
    inference_parser = subparsers.add_parser("inference", help="Inference mode")
    inference_parser.add_argument("--model_id", type=str, default="google/gemma-2-9b", help="Pretrained model ID.")
    inference_parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model.")
    inference_parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset (JSONL).")
    inference_parser.add_argument("--output_path", type=str, default="./inference_results.jsonl",
                                  help="Path to save inference results.")
    inference_parser.add_argument("--project_name", type=str, default="gemma_inference_project",
                                  help="W&B project name for inference.")
    inference_parser.add_argument("--run_name", type=str, default="gemma_inference_run",
                                  help="W&B run name for inference.")
    inference_parser.add_argument("--num_votes", type=int, default=32,
                                  help="Number of CoT samples (K) for majority voting.")
    inference_parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")

    return parser.parse_args()
