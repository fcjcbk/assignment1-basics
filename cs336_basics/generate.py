from cs336_basics.train_model import load_config, configure_logging, resolve_device, set_seed, build_model, build_optimizer, load_checkpoint
from cs336_basics.encode_validation import _tokenizer_artifact_paths
import torch
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.decode import generate
from einops import rearrange
from pathlib import Path


def main() -> None:
    config = load_config("train_config.json")

    logger = configure_logging(config.logging)
    resolved_device = config.device
    set_seed(config.seed)
    
    model = build_model(config.model, resolved_device)
    optimizer = build_optimizer(model, config.optimizer)
    checkpoint_iteration = load_checkpoint("data/checkpoint/train_checkpoint_step_40000.pt", model, optimizer)

    vocab_path, merges_path, metadata_path = _tokenizer_artifact_paths(Path("data/tinystories_train_tokenizer"))
    tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
        metadata_path=metadata_path,
    )
    input = "Where is Tom and Lily?"

    input_tokens = tokenizer.encode(input)
    t = torch.tensor(input_tokens, dtype=torch.int64, device=resolved_device)
    t = rearrange(t, "seq -> 1 seq")
    with torch.no_grad():
        model.eval()
        output = generate(model, t, 256, 0)
        output = rearrange(output, "1 seq -> seq")
        print(tokenizer.decode(output.tolist()))

if __name__ == "__main__":
    main()