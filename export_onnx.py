"""Export a trained Stable-Baselines3 PPO policy to ONNX for real robot deployment.

Usage:
    python export_onnx.py runs/ppo_full_dr/best_model.zip
    python export_onnx.py runs/ppo_full_dr/best_model.zip -o robot_policy.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


class OnnxActor(torch.nn.Module):
    """Wraps SB3 PPO Actor policy into a clean, standalone inference module for ONNX.

    Inputs:
        observation: float32 tensor of shape (batch_size, obs_dim)
                     [pitch, pitch_rate, left_wheel_vel*0.1, right_wheel_vel*0.1]
    Outputs:
        action: float32 tensor of shape (batch_size, 2) in range [-1.0, 1.0]
                [left_torque_norm, right_torque_norm]
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Extract features and calculate deterministic mean actions
        features = self.policy.extract_features(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        mean_actions = self.policy.action_net(latent_pi)
        # Clamped to normalized actuator limits [-1.0, 1.0]
        return torch.clamp(mean_actions, -1.0, 1.0)


def export_to_onnx(model_path: str | Path, output_path: str | Path | None = None, verify: bool = True) -> Path:
    """Exports an SB3 policy zip file to ONNX format."""
    model_path = Path(model_path)
    if output_path is None:
        output_path = model_path.with_suffix(".onnx")
    else:
        output_path = Path(output_path)

    print(f"Loading policy from {model_path}...")
    model = PPO.load(model_path, device="cpu")
    obs_dim = model.observation_space.shape[0]
    action_dim = model.action_space.shape[0]

    actor = OnnxActor(model.policy)
    actor.eval()

    dummy_input = torch.zeros((1, obs_dim), dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
        dynamo=False,
    )
    print(f"[Success] Exported ONNX model to: {output_path}")
    print(f"          Input:  'observation' (batch_size, {obs_dim}) float32")
    print(f"          Output: 'action'      (batch_size, {action_dim}) float32 (range [-1, 1])")

    if verify:
        verify_onnx(output_path, actor, obs_dim)

    # Also automatically export self-contained C header for microcontrollers
    try:
        from onnx_to_c import export_onnx_to_c
        export_onnx_to_c(output_path)
    except Exception as e:
        print(f"[Notice] C header export skipped: {e}")

    return output_path


def verify_onnx(onnx_path: Path, torch_actor: torch.nn.Module, obs_dim: int):
    """Verifies that the ONNX model matches PyTorch outputs with sample test inputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[Notice] onnxruntime not installed, skipping numerical verification.")
        return

    session = ort.InferenceSession(str(onnx_path))
    test_obs = np.random.randn(5, obs_dim).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_actor(torch.from_numpy(test_obs)).numpy()

    onnx_out = session.run(["action"], {"observation": test_obs})[0]

    max_diff = np.max(np.abs(torch_out - onnx_out))
    print(f"[Verification] PyTorch vs ONNX max absolute difference: {max_diff:.2e}")
    if max_diff < 1e-4:
        print("              Numerical verification PASSED!")
    else:
        print("              [Warning] Output discrepancy is larger than expected!")


def main():
    parser = argparse.ArgumentParser(description="Export SB3 PPO model to ONNX for robot deployment")
    parser.add_argument("model", help="Path to SB3 policy .zip file")
    parser.add_argument("-o", "--output", default=None, help="Output .onnx file path")
    parser.add_argument("--no-verify", action="store_true", help="Skip ONNX verification")
    args = parser.parse_args()

    export_to_onnx(args.model, args.output, verify=not args.no_verify)


if __name__ == "__main__":
    main()
