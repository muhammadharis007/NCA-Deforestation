#Redundant file - for deployment purposes
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# ----------------- Model Architecture -----------------
class CAModel(torch.nn.Module):
    def __init__(self, channel_n=16, fire_rate=0.5, device='cpu'):
        super().__init__()
        self.device = device
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        
        self.fc1 = torch.nn.Conv2d(channel_n * 3, 128, 1)
        self.fc2 = torch.nn.Conv2d(128, channel_n, 1)
        
        # Initialize weights to zero to ensure stability at the beginning
        with torch.no_grad():
            self.fc2.weight.fill_(0.0)
            self.fc2.bias.fill_(0.0)
            
        self.to(device)

    def perceive(self, x):
        # Sobel filters for spatial gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.t()
        
        w1 = torch.stack([sobel_x, sobel_y], dim=0)
        w = w1.repeat(self.channel_n, 1, 1, 1).to(self.device)
        
        g = torch.nn.functional.conv2d(x, w, groups=self.channel_n, padding=1)
        return torch.cat([x, g], dim=1)

    def forward(self, x, steps=1, distort_mask=None):
        for _ in range(steps):
            y = self.perceive(x)
            y = torch.relu(self.fc1(y))
            y = self.fc2(y)
            
            # Stochastic update (cellular automata behavior)
            update_mask = torch.rand(x.size(0), 1, x.size(2), x.size(3), device=self.device) < self.fire_rate
            x = x + y * update_mask.float()
            
            # Re-apply distortion mask if provided (for testing robustness)
            if distort_mask is not None:
                x = x * distort_mask
                
        return x

# ----------------- Helper Functions -----------------
@st.cache_resource
def load_nca_model(model_path):
    """Loads the pre-trained NCA model onto the CPU."""
    model = CAModel(channel_n=16, fire_rate=0.5, device='cpu')
    # Load state dict safely on CPU
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def to_rgb(x):
    """Converts 16-channel CA state grid to an RGB image."""
    rgb = x[0, :3, :, :].detach().cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)

def apply_damage(grid, damage_type="Circle", radius=10):
    """Applies a specific visual damage to the hidden states grid."""
    mask = torch.ones_like(grid)
    h, w = grid.shape[2], grid.shape[3]
    cy, cx = h // 2, w // 2
    
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    dist = torch.sqrt((x - cx)**2 + (y - cy)**2)
    
    if damage_type == "Circle":
        mask[:, :, dist < radius] = 0.0
    elif damage_type == "Half Cut":
        mask[:, :, y > cy] = 0.0
        
    return grid * mask, mask

# ----------------- Streamlit UI Configuration -----------------
st.set_page_config(page_title="NCA Regenerative Dashboard", layout="wide")

st.title("🌱 Neural Cellular Automata (NCA) Interactive Dashboard")
st.markdown("### Developed by: **Muhammad Haris, Muhammad Ahsan Shaikh, Muhammad Abdullah**")
st.write("Explore the emergence, growth, and autonomous self-repair capabilities of Neural Cellular Automata.")

# Model selection and path setup
MODEL_PATH = "nca_best_of_both.pth"

try:
    nca_model = load_nca_model(MODEL_PATH)
    st.sidebar.success("✅ Pre-trained model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model file '{MODEL_PATH}'. Details: {e}")
    st.stop()

# Sidebar Controls
st.sidebar.header("🎛️ Simulation Configurations")
grid_size = st.sidebar.slider("Grid Resolution (Square Size)", min_value=32, max_value=64, value=40, step=8)
num_steps = st.sidebar.slider("NCA Iteration Steps", min_value=1, max_value=500, value=100, step=10)

st.sidebar.header("💥 Damage Simulation Settings")
enable_damage = st.sidebar.checkbox("Inflict Grid Damage", value=False)
damage_type = st.sidebar.selectbox("Damage Shape Pattern", ["Circle", "Half Cut"], disabled=not enable_damage)
damage_radius = st.sidebar.slider("Damage Impact Radius", min_value=5, max_value=20, value=10, disabled=(not enable_damage or damage_type=="Half Cut"))

# Core Session state to maintain the cell grid across dashboard interactions
if 'ca_grid' not in st.session_state or st.sidebar.button("♻️ Reset Grid Seed"):
    # Initialize grid with a seed: 16 channels, all 0 except the center pixel which has hidden states set to 1.0
    init_grid = torch.zeros((1, 16, grid_size, grid_size))
    init_grid[:, 3:, grid_size//2, grid_size//2] = 1.0  # Living channel + hidden features
    st.session_state.ca_grid = init_grid
    st.session_state.history = [to_rgb(init_grid)]

# Layout Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📺 Interactive Execution Workspace")
    
    if st.button("🚀 Run Growth Simulation", use_container_width=True):
        current_grid = st.session_state.ca_grid
        
        # Apply damage if active before growing
        distort_mask = None
        if enable_damage:
            current_grid, distort_mask = apply_damage(current_grid, damage_type, damage_radius)
            st.warning(f"⚠️ Applied '{damage_type}' damage to the grid before stepping forward.")
        
        # Run inference through the model
        with st.spinner(f"Simulating {num_steps} cellular timesteps..."):
            with torch.no_grad():
                updated_grid = nca_model(current_grid, steps=num_steps)
            
            # Save updated grid state back into the browser session
            st.session_state.ca_grid = updated_grid
            st.session_state.history.append(to_rgb(updated_grid))
            
    # Always render current status frame
    current_img = to_rgb(st.session_state.ca_grid)
    # Upscale image for better visibility using nearest neighbor to avoid blur
    upscaled_img = cv2.resize(current_img, (300, 300), interpolation=cv2.INTER_NEAREST)
    st.image(upscaled_img, caption="Current State Grid (RGB Channels)", use_column_width=False)

with col2:
    st.subheader("⏳ Evolutionary History")
    st.write(f"Total recorded key frames: {len(st.session_state.history)}")
    
    # Display historical steps if there are multiple frames
    if len(st.session_state.history) > 1:
        frame_idx = st.slider("Browse Timeline History", 0, len(st.session_state.history)-1, len(st.session_state.history)-1)
        selected_img = st.session_state.history[frame_idx]
        upscaled_hist = cv2.resize(selected_img, (300, 300), interpolation=cv2.INTER_NEAREST)
        st.image(upscaled_hist, caption=f"Snapshot Frame #{frame_idx}", use_column_width=False)
    else:
        st.info("Run the simulation to generate timeline frames and monitor morphogenetic growth.")