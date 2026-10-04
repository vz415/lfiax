"""Generate LFiax SIR observations using the iDAD TorchSDE model."""
import argparse
from pathlib import Path

import torch

from lfiax.utils.torch_utils import solve_sir_sdes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output", default="sde_data/sir_sde_data_real_8_0.pt")
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    params = torch.tensor([0.8, 0.1], device=args.device).repeat(args.num_samples, 1)
    observations = solve_sir_sdes(
        num_samples=args.num_samples,
        device=args.device,
        grid=10000,
        params=params,
        params_log_probs=torch.zeros(args.num_samples, device=args.device),
        seed=[args.seed],
    )
    if observations["num_samples"] == 0:
        raise RuntimeError("No surviving trajectories; try another seed.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(observations, output)
    print(f"Saved {observations['num_samples']} observation trajectories to {output}")


if __name__ == "__main__":
    main()
