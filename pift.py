import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import functools


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from diffusers import DDPMScheduler


class Config:
    PRETRAINED_MODEL_PATH = './model/ddpm_pretrained_model.pth'
    MODEL_INPUT_DIM = 3
    TARGET_CENTER_X = 0.0
    TARGET_CENTER_Y = 0.0
    TARGET_RADIUS = 0.75
    T = 4
    ALPHA_SCHEDULE = torch.linspace(0.9999, 0.9, T)
    SIGMA_SCHEDULE = torch.sqrt(1.0 - ALPHA_SCHEDULE**2)
    M_ITER = 100
    BATCH_SIZE = 512
    LR = 5e-4
    BETA = 30


class MLPModel(nn.Module):
    def __init__(self, input_dim=3, time_emb_dim=128, hidden_dim=256):
        super().__init__()
        self.time_embedding = nn.Sequential(nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        self.main_net = nn.Sequential(nn.Linear(input_dim + time_emb_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, input_dim))
        self.pos_freqs = nn.Parameter(torch.randn(1, time_emb_dim // 2) * 0.1, requires_grad=False)

    def forward(self, x, t):
        t = t.to(x.device)
        t_scaled = t.float() * 0.001
        if t_scaled.ndim == 0:
            t_scaled = t_scaled.unsqueeze(0)
        args = t_scaled.unsqueeze(-1) * self.pos_freqs
        time_features = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        time_emb = self.time_embedding(time_features)
        if time_emb.shape[0] != x.shape[0]:
            time_emb = time_emb.expand(x.shape[0], -1)
        x_with_time = torch.cat([x, time_emb], dim=1)
        return self.main_net(x_with_time)


def reward_with_constraint(samples_polar, center_x, center_y, target_radius, loc_sigma=0.1, constraint_sigma=0.1):
    r, c, s = samples_polar[:, 0], samples_polar[:, 1], samples_polar[:, 2]
    x, y = r * c, r * s
    dist_from_center = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    distance_error_sq = (dist_from_center - target_radius)**2
    location_reward = torch.exp(-distance_error_sq / (2 * loc_sigma**2))
    constraint_error_sq = (c**2 + s**2 - 1.0)**2
    constraint_reward = torch.exp(-constraint_error_sq / (2 * constraint_sigma**2))
    final_reward = location_reward * constraint_reward
    return final_reward


def run_theorem_verification_experiment(beta_val, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running Fine-tuning Experiment for beta = {beta_val} on {device} ---")
    
    s_pre_model = MLPModel(input_dim=config.MODEL_INPUT_DIM).to(device)
    s_pre_model.load_state_dict(torch.load(config.PRETRAINED_MODEL_PATH, map_location=device, weights_only=True))
    s_pre_model.eval()
    
    u_policies = [MLPModel(input_dim=config.MODEL_INPUT_DIM).to(device) for _ in range(config.T)]
    for u_net in u_policies:
        u_net.load_state_dict(s_pre_model.state_dict())
    
    u_optimizers = [optim.AdamW(u.parameters(), lr=config.LR) for u in u_policies]
    policy_change_histories = {t: [] for t in [20]}

    v_t_plus_1_fn = functools.partial(reward_with_constraint, 
                                      center_x=config.TARGET_CENTER_X, 
                                      center_y=config.TARGET_CENTER_Y, 
                                      target_radius=config.TARGET_RADIUS)
    
    for t in tqdm(reversed(range(config.T)), desc="Fine-tuning (Time Steps)"):
        u_t, u_optimizer = u_policies[t], u_optimizers[t]
        u_t.train()
        alpha_t, sigma_t = config.ALPHA_SCHEDULE[t].to(device), config.SIGMA_SCHEDULE[t].to(device)

        if t in policy_change_histories:
            prev_params = torch.cat([p.detach().flatten() for p in u_t.parameters()])
        
        inner_loop = tqdm(range(config.M_ITER), desc=f"Policy Iteration t={t}", leave=False)
        for m in inner_loop:
            y_t = torch.randn(config.BATCH_SIZE, config.MODEL_INPUT_DIM, device=device)
            y_t.requires_grad = True
            
            control_u_noise = u_t(y_t, torch.tensor(t, device=device))
            
            pred_original_sample = (y_t - sigma_t * control_u_noise) / torch.sqrt(alpha_t)
            y_t_plus_1 = torch.sqrt(alpha_t) * pred_original_sample + sigma_t * control_u_noise
            
            V_val = v_t_plus_1_fn(y_t_plus_1).sum()
            grad_V_y_t_plus_1, = torch.autograd.grad(V_val, y_t_plus_1, grad_outputs=torch.ones_like(V_val))

            with torch.no_grad():
                s_pre_noise = s_pre_model(y_t, torch.tensor(t, device=device))
                const = (torch.sqrt(alpha_t) * sigma_t**2) / ((1 - alpha_t) * beta_val)
                correction_term = const * grad_V_y_t_plus_1 
                target_noise = s_pre_noise + correction_term

            u_optimizer.zero_grad()
            u_pred_noise = u_t(y_t, torch.tensor(t, device=device))
            loss_u = nn.MSELoss()(u_pred_noise, target_noise)
            loss_u.backward()
            u_optimizer.step()

            if t in policy_change_histories:
                current_params = torch.cat([p.detach().flatten() for p in u_t.parameters()])
                policy_change = torch.linalg.norm(current_params - prev_params).item()
                policy_change_histories[t].append(policy_change)
                prev_params = current_params
      
        def get_v_t_fn(converged_u_t, next_v_fn, t_params):
            alpha_t_c, sigma_t_c, beta_c = t_params
          
            def v_t_fn(y_t_eval):
               
                control_u_final_noise = converged_u_t(y_t_eval, torch.tensor(t, device=device))
                
                pred_original_sample_final = (y_t_eval - sigma_t_c * control_u_final_noise) / torch.sqrt(alpha_t_c)
                y_t_plus_1_eval = torch.sqrt(alpha_t_c) * pred_original_sample_final + sigma_t_c * control_u_final_noise
                
                v_t_plus_1_val = next_v_fn(y_t_plus_1_eval)
                
                with torch.no_grad():
                    s_pre_noise_final = s_pre_model(y_t_eval, torch.tensor(t, device=device))
                
                kl_penalty = beta_c * ((1 - alpha_t_c)**2 / (2 * alpha_t_c * sigma_t_c**2)) * torch.sum((control_u_final_noise - s_pre_noise_final)**2, dim=1)
                
                return v_t_plus_1_val - kl_penalty
            return v_t_fn
        
        u_t.eval() 
        v_t_plus_1_fn = get_v_t_fn(u_t, v_t_plus_1_fn, (alpha_t, sigma_t, beta_val))
    
    return policy_change_histories, u_policies


def plot_results(policy_histories, final_policies, config):
    plt.figure(figsize=(10, 8))
    start_iter = 10
    end_iter = 70 
    for t, changes in policy_histories.items():
        if not changes: continue
        safe_end_iter = min(end_iter, len(changes))
        if start_iter >= safe_end_iter: continue
        sliced_changes = np.array(changes[start_iter:safe_end_iter])
        sliced_changes[sliced_changes < 1e-10] = 1e-10
        log_changes = np.log(sliced_changes)
        iterations = np.arange(start_iter, safe_end_iter)
        if len(iterations) > 0:
            plt.scatter(iterations, log_changes, alpha=0.6, s=15, label=f'Data points (t={t})')
            if len(iterations) > 1:
                coeffs = np.polyfit(iterations, log_changes, 1)
                fit_line = np.poly1d(coeffs)
                plt.plot(iterations, fit_line(iterations), linewidth=2, label=f'Fit line (t={t})')
    plt.title(f'Theorem 3.1 Verification (beta = {config.BETA})', fontsize=16)
    plt.xlabel('Iteration (m)', fontsize=12)
    plt.ylabel('Log of Policy Change ||θ_m+1 - θ_m||', fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.title('Final Generated Samples After Fine-tuning', fontsize=16)
    angles = np.linspace(0, 2*np.pi, 200)
    plt.plot(np.cos(angles), np.sin(angles), 'b--', alpha=0.4, label='Pre-trained Target (Unit Circle)')
    cx, cy, r_target = config.TARGET_CENTER_X, config.TARGET_CENTER_Y, config.TARGET_RADIUS
    target_circle = plt.Circle((cx, cy), r_target, fill=False, edgecolor='k', linestyle='--', label='Fine-tune Target')
    plt.gca().add_patch(target_circle)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    final_samples_polar = torch.randn(1024, config.MODEL_INPUT_DIM, device=device)
    scheduler_eval = DDPMScheduler(num_train_timesteps=1000)
    scheduler_eval.set_timesteps(config.T)
    for t_eval in tqdm(scheduler_eval.timesteps, desc="Final Sampling"):
        with torch.no_grad():
            time_idx = min(int(t_eval * config.T / 1000), config.T - 1)
            u_policy = final_policies[time_idx].eval()
            noise_pred = u_policy(final_samples_polar, t_eval)
        final_samples_polar = scheduler_eval.step(noise_pred, t_eval, final_samples_polar).prev_sample
    samples_polar = final_samples_polar.cpu()
    r, c, s = samples_polar[:, 0], samples_polar[:, 1], samples_polar[:, 2]
    x, y = r * c, r * s
    plt.scatter(x, y, alpha=0.3, s=10, c='red', label='Fine-tuned Samples')
    plt.xlabel('x'); plt.ylabel('y')
    plt.legend(); plt.grid(True); plt.axis('equal')
    plt.xlim(-1.5, 1.5); plt.ylim(-1, 2)
    plt.show()


if __name__ == '__main__':
    config = Config()

    policy_histories, final_policies = run_theorem_verification_experiment(config.BETA, config)
    plot_results(policy_histories, final_policies, config)