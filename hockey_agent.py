from __future__ import annotations
import omegaconf
import hockey_env
from network import Network
import numpy as np
import torch
import os
import ray
from utils import get_config
import time
import uuid
import numpy as np

@ray.remote
class SimulatorWorker:
    def __init__(self, config):
        self.config = config
        self.env = hockey_env.EnvWrapper()
        self.env.reset()
        
    def step(self, obs1, action1_batch, action2):
        """
        Args:
            obs: [obs_dim]
            action1s: List of [act_dim]
            action2: [act_dim]
        Return:
            List of size n_advanced_actions of next_obs_1: [obs_dim]
        """
        results = []
        for action1 in action1_batch:
            # self.env.reset() # Slower, but a bit more accurate
            self.env.env.set_state(obs1)
            for _ in range(self.config.rl.frame_skip): # Only one step forward
                next_obs1, _, rew, _, trunc = self.env.step(np.hstack([action1, action2]))
                if trunc:
                    self.env.reset()
                    break
            next_obs1 = hockey_env.EnvWrapper.augment(next_obs1, self.config)
            forecast = hockey_env.EnvWrapper.forecast(next_obs1, 5, 2)    
            results.append((next_obs1, forecast, rew))
        return results

class HockeyAgent:
    def __init__(self, path=".", nets=None, config=None, eval=True):
        self.nets = []
        self.path = path
        self.eval = eval
        self.device = torch.device("cuda:0")
        if nets is not None:
            self.nets = nets
            for net in nets:
                net.to(self.device)
                if self.eval:
                    net.eval()
        else:
            self._load_model()
        self.actions = torch.Tensor(self.nets[0].config.env.action_space).to(self.device).float()
        self.n_actions = self.actions.shape[0]
        self.config = config
        if self.config is None:
            self.config = self.nets[0].config # TODO: Make the inference runs for each net on its own config.
        self.actions1 = self.actions[None, None].tile(self.n_actions, 1, 1).flatten(end_dim=2)
        self.actions2 = self.actions[None, :, None].tile(1, self.n_actions, 1).flatten(end_dim=2)
        self.game_id = None
        self.gym_workers = []
        self.step = 0
        
    def _load_model(self):
        for file in os.listdir(self.path):
            if file.endswith(".pth"):
                print("Loading model: ", file)
                config_file = file.replace(".pth", ".yaml")
                config = get_config()
                config = omegaconf.OmegaConf.merge(config, omegaconf.OmegaConf.load(os.path.join(self.path, config_file)))
                net = Network(config)
                net.load_state_dict(torch.load(os.path.join(self.path, file)))
                net.to(self.device)
                self.nets.append(net)
        for net in self.nets:
            net.to(self.device)
            if self.eval:
                net.eval()
        
    def minimax(self, obs, payoff_matrix: np.ndarray):
        """
        Args:
            payoff_matrix: [bs, player 2, player 1]
        Return:
            greedy_action_idx: [bs]
        """
        if len(payoff_matrix.shape) == 2:
            payoff_matrix = payoff_matrix[None]
        
        if self.config.rl.action_masking and obs is not None:
            time_left_1 = obs[:, 16]
            time_left_2 = obs[:, 17]
            payoff_matrix[:, :, 0][time_left_1 == 0] = -1
            payoff_matrix[:, 0, :][time_left_2 == 0] = 1
    
        player_1_payoff = payoff_matrix.min(dim=1).values 
        if self.config.rl.action_masking and obs is not None:
            player_1_payoff[:, 0][time_left_2 == 0] = -1
        return player_1_payoff.argmax(dim=1)
    
    @torch.no_grad()
    def simulator_search(self, obs):
        """
        Use world model for 1-step-planning.

        Args:
            obs: [obs_dim]
            sim_steps: 2 since model was trained with frame skip. Tried with 1 but works worse.
        Return:
            action: [act_dim]
            max_action_idx: [1]
            q_target: [n_actions player 2, n_actions player 1]
        """
        n_actions = self.actions.shape[0]
        
        # Initialize gym worker
        if len(self.gym_workers) == 0:
            self.gym_workers = [SimulatorWorker.remote(self.config) for _ in range(n_actions)]
            time.sleep(0.5)
            
        actions = self.actions.cpu().numpy()
        
        # [n_actions player 2, n_actions player 1, ret_size]
        results = ray.get([self.gym_workers[action_2_idx].step.remote(obs1=obs, action1_batch=actions, action2=actions[action_2_idx]) for action_2_idx in range(n_actions)])    
        next_obs = torch.zeros((n_actions, n_actions, self.config.env.obs_dim)).to(self.device).float()
        forecast = torch.zeros((n_actions, n_actions, self.config.env.forecast_step * 2)).to(self.device).float()
        rewards = torch.zeros((n_actions, n_actions)).float()
        for action_2_idx in range(n_actions):
            for action_1_idx in range(n_actions):
                next_obs[action_2_idx, action_1_idx] = torch.from_numpy(results[action_2_idx][action_1_idx][0]).float()
                forecast[action_2_idx, action_1_idx] = torch.from_numpy(results[action_2_idx][action_1_idx][1]).float()
                rewards[action_2_idx, action_1_idx] = results[action_2_idx][action_1_idx][2]
        next_obs = next_obs.flatten(end_dim=1)
        forecast = forecast.flatten(end_dim=1)
        
        payoffs = None
        for net in self.nets:
            payoff = net.policy(next_obs, forecast)
            if payoffs is None:
                payoffs = payoff
            else:
                payoffs += payoff
        payoffs /= len(self.nets)
        action_idx = self.minimax(obs, payoffs)
        return actions[action_idx].cpu()
    
    @torch.no_grad()
    def batch_obs_search(self, obs: np.array, forecast: np.array):
        """
        Args:
            obs: [bs, obs_dim]
            forecast: [bs, forecast_step, obs_dim]
        Return:
            action: [bs, act_dim]
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device).float()
        if forecast is not None:
            if isinstance(forecast, np.ndarray):
                forecast = torch.from_numpy(forecast).to(self.device).float()
            forecast = forecast.to(self.device).float()
        payoffs = []
        for net in self.nets:
            payoffs.append(net.policy(obs, forecast))
        payoffs = torch.stack(payoffs, dim=0).mean(dim=0)
        action_idx = self.minimax(obs, payoffs)
        actions = self.actions[action_idx]
        return actions.cpu(), action_idx.cpu()
    
    @torch.no_grad()
    def latent_both_sides_search(self, latent_states):
        payoffs_player1  = []
        for net in self.nets:
            payoffs_player1.append(net.latent_policy(latent_states))
        payoffs_player1 = torch.stack(payoffs_player1, dim=0).mean(dim=0)
        payoffs_player2 = -payoffs_player1.permute(0, 2, 1) # Zero-sum assumption
        action_idx1 = self.minimax(None, payoffs_player1)
        action_idx2 = self.minimax(None, payoffs_player2)
        return self.actions[action_idx1], self.actions[action_idx2]
        
    @torch.no_grad()
    def both_sides_search(self, obs1: np.array, obs2: np.array, forecast1: np.array):
        """
        Args:
            obs1: [obs_dim]
            obs2: [obs_dim]
            forecast1: [forecast_step, obs_dim]
        Return:
            action1: [act_dim]
            action2: [act_dim]
        """
        if isinstance(obs1, np.ndarray):
            obs1 = torch.from_numpy(obs1).to(self.device).float()
            obs2 = torch.from_numpy(obs2).to(self.device).float()
        if forecast1 is not None:
            if isinstance(forecast1, np.ndarray):
                forecast1 = torch.from_numpy(forecast1).to(self.device).float()
            forecast1 = forecast1.to(self.device).float()
        payoffs_player1 = []
        for net in self.nets:
            payoffs_player1.append(net.policy(obs1, forecast1))
        payoffs_player1 = torch.stack(payoffs_player1, dim=0).mean(dim=0)
        payoffs_player2 = -payoffs_player1.permute(0, 2, 1) # Zero-sum assumption
        action_idx1 = self.minimax(obs1, payoffs_player1)
        action_idx2 = self.minimax(obs2, payoffs_player2)
        return action_idx1.cpu(), action_idx2.cpu()

    def get_step(self, obs: list[float]) -> list[float]:
        self.step += 1
        
        obs = np.array(obs)
        if self.config.env.obs_augmentation:
            obs = hockey_env.EnvWrapper.augment(obs, self.config)
        
        if self.config.env.use_forecast:
            forecast = hockey_env.EnvWrapper.forecast(obs, self.config.env.forecast_step, self.config.rl.frame_skip)
            forecast = torch.from_numpy(forecast).to(self.device).float()[None]
        else:
            forecast = None
        obs = torch.from_numpy(obs).to(self.device).float()[None]
        actions, _ = self.batch_obs_search(obs, forecast)
        
        return actions.cpu().numpy().tolist()

    def on_start_game(self, game_id) -> None:
        self._load_model()
        game_id = uuid.UUID(int=int.from_bytes(game_id, 'big'))
        self.game_id = game_id
        print(f"Game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        print(f"Game ended. My score: {stats[0]}. Opponent score: {stats[1]}")

def evaluate_against_basic_opponent():
    env = hockey_env.EnvWrapper()
    agent1 = HockeyAgent()
    agent2 = hockey_env.BasicOpponent(weak=False)
    step_count = 0
    games = 0
    while True:
        games += 1
        obs1, obs2 = env.reset(seed=games)
        trunc = False
        while not trunc:
            action1 = agent1.get_step(obs1)[0]
            action2 = agent2.act(obs2)
            obs1, obs2, reward, _, trunc = env.step(np.hstack([action1, action2]))
            env.env.render()
            step_count += 1
        print(f"Scored: {reward}")
    

if __name__ == "__main__":
    evaluate_against_basic_opponent()