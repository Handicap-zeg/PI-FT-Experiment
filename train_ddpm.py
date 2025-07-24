
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from diffusers import DDPMScheduler


class Config:

    NUM_SAMPLES = 20000
    RADIUS = 1.0         
    NOISE_STD = 0.05

    
    TIMESTEPS = 1000

   
    BATCH_SIZE = 512
    EPOCHS = 15000
    LEARNING_RATE = 2e-4
    
  
    MODEL_INPUT_DIM = 3 
    

    MODEL_PATH = 'ddpm_pretrained_radius1_polar_model.pth'


def generate_circle_data_final(num_samples=Config.NUM_SAMPLES, radius=Config.RADIUS, noise=Config.NOISE_STD):
    """生成最终的数据表示 (r, cos(theta), sin(theta))"""
    angles = torch.rand(num_samples) * 2 * np.pi
    radii = torch.randn(num_samples) * noise + radius
    
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    dataset = torch.stack([radii, cos_angles, sin_angles], dim=1)
    print(f"Generated 3D dataset with shape: {dataset.shape}")
    return TensorDataset(dataset)

class MLPModel(nn.Module):
    def __init__(self, input_dim=3, time_emb_dim=128, hidden_dim=256):
        super().__init__()
        
        self.time_embedding = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        self.main_net = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
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

def test_and_visualize_model(model, scheduler, original_data_3d):
    """使用训练好的模型生成样本并进行可视化"""
    print("\n--- Generating samples from the trained model ---")
    
    model.eval()
    device = next(model.parameters()).device
   
    samples_3d = torch.randn(500, Config.MODEL_INPUT_DIM, device=device)
    
    for t in tqdm(scheduler.timesteps, desc="Sampling"):
        with torch.no_grad():
            noise_pred = model(samples_3d, t)
        samples_3d = scheduler.step(noise_pred, t, samples_3d).prev_sample

 
    r_orig, cos_orig, sin_orig = original_data_3d[:, 0], original_data_3d[:, 1], original_data_3d[:, 2]
    x_orig = r_orig * cos_orig
    y_orig = r_orig * sin_orig
    
 
    samples_3d_cpu = samples_3d.cpu()
    r_gen, cos_gen, sin_gen = samples_3d_cpu[:, 0], samples_3d_cpu[:, 1], samples_3d_cpu[:, 2]
    x_gen = r_gen * cos_gen
    y_gen = r_gen * sin_gen

    plt.figure(figsize=(8, 8))
    plt.scatter(x_orig, y_orig, alpha=0.1, label='Original Training Data (r=1)', s=10)
    plt.scatter(x_gen, y_gen, alpha=0.5, label='Generated Samples', s=10, c='red')
    plt.title('Pre-trained DDPM Performance (3D Polar Representation)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def main():
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset = generate_circle_data_final()
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    
    model = MLPModel(input_dim=config.MODEL_INPUT_DIM).to(device)
    print(f"Model created. Parameter count: {sum(p.numel() for p in model.parameters())}")

    scheduler = DDPMScheduler(num_train_timesteps=config.TIMESTEPS)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.MSELoss()

    print("\n--- Starting Pre-training (3D Polar Representation, Radius=1) ---")
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for batch in progress_bar:
  
            clean_data_3d = batch[0].to(device)
            
            noise = torch.randn_like(clean_data_3d)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (clean_data_3d.shape[0],), device=device)
            noisy_data_3d = scheduler.add_noise(clean_data_3d, noise, timesteps)
            
            noise_pred = model(noisy_data_3d, timesteps)
            
            loss = loss_fn(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    print("\n--- Pre-training finished ---")
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"Pre-trained model saved to: {config.MODEL_PATH}")
    
    test_and_visualize_model(model, scheduler, train_dataset.tensors[0])

if __name__ == '__main__':
    main()