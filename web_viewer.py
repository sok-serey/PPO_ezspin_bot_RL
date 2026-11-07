"""Browser viewer: runs the newest trained policy and streams poses to a three.js page.

    python web_viewer.py                       # newest runs/*/best_model.zip, reloads when it changes
    python web_viewer.py --model path/to.zip --port 8000

Open http://<this machine's IP>:8000 from any device on the network.
No OpenGL is needed on the server; the browser renders the STL meshes.
"""

import argparse
import json
import threading
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import balance_bot  # noqa: F401
from balance_bot.env import MODEL_PATH

ROOT = Path(__file__).parent
MESH_DIR = MODEL_PATH.parent / "meshes"
WEB_DIR = ROOT / "web"
BODIES = ["chassis", "left_wheel_link", "right_wheel_link"]


def visual_geoms() -> list[dict]:
    """Mesh geoms from the MJCF with their pose in the parent body frame (STL frame, not MuJoCo's recentred one)."""
    root = ET.parse(MODEL_PATH).getroot()
    mesh_files = {m.get("name"): m.get("file") for m in root.iter("mesh")}
    out = []
    for body in root.iter("body"):
        for g in body.findall("geom"):
            if g.get("class") == "visual":
                out.append({
                    "body": body.get("name"),
                    "file": mesh_files[g.get("mesh")],
                    "pos": [float(v) for v in g.get("pos", "0 0 0").split()],
                    "rgba": [float(v) for v in g.get("rgba", "1 1 1 1").split()],
                })
    return out


class Sim:
    def __init__(self, model_arg: str | None):
        self.model_arg = model_arg
        self.env = None
        self.u = None
        self.policy, self.policy_path, self.policy_mtime = None, None, 0.0
        self.lock = threading.Lock()
        self.state = {}
        self.push_force = 0.0
        self.push_steps = 0
        self.reset_request = False
        self.episode = 0
        self.last_returns: list[float] = []

        # Find and load the policy or initialize default environment
        self._maybe_reload()
        if self.env is None:
            self._create_env_for_obs_dim(5)

    def _create_env_for_obs_dim(self, obs_dim: int):
        if obs_dim == 10:
            obs_mode, command_vel = "full", False
        elif obs_dim == 11:
            obs_mode, command_vel = "full", True
        elif obs_dim == 6:
            obs_mode, command_vel = "robot_pos", True
        elif obs_dim == 5:
            obs_mode, command_vel = "robot_pos", False
        else:
            obs_mode, command_vel = "robot", False

        if self.env is not None:
            self.env.close()
        self.env = gym.make("BalanceBot-v0", obs_mode=obs_mode, command_vel=command_vel, push_disturbances=False)
        self.u = self.env.unwrapped
        self.reset_request = True

    def _find_policy(self) -> Path | None:
        if self.model_arg:
            return Path(self.model_arg)
        found = sorted(ROOT.glob("runs/*/best_model.zip"), key=lambda p: p.stat().st_mtime)
        return found[-1] if found else None

    def _maybe_reload(self) -> bool:
        path = self._find_policy()
        if path is None or not path.exists():
            return False
        mtime = path.stat().st_mtime
        if path == self.policy_path and mtime == self.policy_mtime:
            return False
        try:
            new_policy = PPO.load(path, device="cpu")
            obs_dim = new_policy.observation_space.shape[0]
            if self.env is None or obs_dim != self.env.observation_space.shape[0]:
                self._create_env_for_obs_dim(obs_dim)
            self.policy = new_policy
            self.policy_path, self.policy_mtime = path, mtime
            print(f"loaded policy {path} (obs_dim={obs_dim})")
            return True
        except Exception as e:  # file may be mid-write by EvalCallback
            print(f"policy load failed ({e}), retrying")
            return False

    def run(self):
        obs, _ = self.env.reset()
        ep_ret, ep_len = 0.0, 0
        last_check = time.perf_counter()
        dt = self.u.dt
        next_t = time.perf_counter()
        while True:
            now = time.perf_counter()
            if now - last_check > 2.0:
                reloaded = self._maybe_reload()
                last_check = now
                if reloaded or self.reset_request:
                    obs, _ = self.env.reset()
                    ep_ret, ep_len = 0.0, 0
                    self.reset_request = False
                    next_t = time.perf_counter()

            if self.push_steps > 0:  # horizontal shove at the chassis COM
                self.u.data.xfrc_applied[self.u._chassis_id, 0] = self.push_force
                self.push_steps -= 1
            if self.policy is not None:
                action, _ = self.policy.predict(obs, deterministic=True)
            else:
                action = np.zeros(2)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.u.data.xfrc_applied[self.u._chassis_id, :] = 0.0
            ep_ret += reward
            ep_len += 1

            with self.lock:
                d = self.u.data
                self.state = {
                    "bodies": {b: {"pos": d.body(b).xpos.tolist(), "quat": d.body(b).xquat.tolist()} for b in BODIES},
                    "t": ep_len * dt,
                    "episode": self.episode,
                    "return": ep_ret,
                    "pitch_err_deg": float(np.degrees(info["pitch"] - info["pitch_eq"])),
                    "pitch_eq_deg": float(np.degrees(info["pitch_eq"])),
                    "speed": info["forward_vel"],
                    "torque": [float(a) for a in action],
                    "policy": str(self.policy_path.relative_to(ROOT)) if self.policy_path else None,
                    "last_returns": self.last_returns[-5:],
                    "fell": False,
                }

            if terminated or truncated or self.reset_request:
                with self.lock:
                    self.state["fell"] = bool(terminated)
                self.last_returns.append(round(ep_ret, 1))
                self.episode += 1
                self.reset_request = False
                if terminated:
                    time.sleep(0.8)  # let the viewer see the fall
                obs, _ = self.env.reset()
                ep_ret, ep_len = 0.0, 0
                next_t = time.perf_counter()

            next_t += dt
            time.sleep(max(0.0, next_t - time.perf_counter()))

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)


def make_handler(sim: Sim, geoms: list[dict]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send((WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/config":
                self._send(json.dumps({"geoms": geoms}).encode(), "application/json")
            elif self.path.startswith("/meshes/"):
                f = MESH_DIR / Path(self.path).name
                if f.is_file():
                    self._send(f.read_bytes(), "application/octet-stream")
                else:
                    self._send(b"not found", "text/plain", 404)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while True:
                        self.wfile.write(f"data: {json.dumps(sim.snapshot())}\n\n".encode())
                        self.wfile.flush()
                        time.sleep(1 / 50)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path == "/push":
                sim.push_force = float(np.random.choice([-1, 1]) * np.random.uniform(2.0, 4.0))
                sim.push_steps = 10  # 0.1 s
            elif self.path == "/reset":
                sim.reset_request = True
            self._send(b"ok", "text/plain")

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="policy .zip (default: newest runs/*/best_model.zip)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    sim = Sim(args.model)
    threading.Thread(target=sim.run, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(sim, visual_geoms()))
    server.daemon_threads = True
    print(f"\nServing 3D Web Viewer at:")
    print(f"  -> Local browser:  http://localhost:{args.port}  (or http://127.0.0.1:{args.port})")
    print(f"  -> Mobile/Network: http://<your-computer-ip>:{args.port}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
