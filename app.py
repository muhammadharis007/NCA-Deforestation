import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2

# ----------------- Model Architecture (ResearchNCA) -----------------
class CAModel(torch.nn.Module):
    def __init__(self, channels=32, device='cpu'):
        super().__init__()
        self.device = device
        self.channels = channels
        
        # Sobel filters initialized to a 4D shape [1, 1, 3, 3]
        self.sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3) / 8.0
        self.sobel_y = self.sobel_x.transpose(2,3)
        self.register_buffer('Kx', self.sobel_x)
        self.register_buffer('Ky', self.sobel_y)

        # Perceive convolution WITH bias (matches your checkpoint)
        self.perceive_conv = torch.nn.Conv2d(channels, channels, 3, padding=1)
        
        # 1x1 Convolutions mapping 130 channels -> 128 channels -> 32 channels
        self.w1 = torch.nn.Conv2d((channels * 4) + 2, 128, 1)
        self.w2 = torch.nn.Conv2d(128, channels, 1)
        
        self.to(device)

    def perceive(self, state):
        k = state.shape[1]
        x_grad = torch.nn.functional.conv2d(state, self.Kx.repeat(k,1,1,1), padding=1, groups=k)
        y_grad = torch.nn.functional.conv2d(state, self.Ky.repeat(k,1,1,1), padding=1, groups=k)
        learned = self.perceive_conv(state)
        return torch.cat([state, x_grad, y_grad, learned], dim=1)

    def get_slope(self, state):
        elev = state[:, 2:3]
        dx = torch.nn.functional.conv2d(elev, self.Kx, padding=1)
        dy = torch.nn.functional.conv2d(elev, self.Ky, padding=1)
        return torch.sqrt(dx**2 + dy**2)

    def forward(self, x, steps=32, use_physics=False):
        # x is expected to have 3 channels: [forest_cover, roads, elevation]
        forest_init = x[:, 0:1]
        static = x[:, 1:3]
        b, _, h, w = forest_init.shape
        hidden = torch.zeros(b, self.channels - 1, h, w, device=x.device)
        state = torch.cat([forest_init, hidden], dim=1)

        slope_map = self.get_slope(x)
        if slope_map.max() > 0: 
            slope_map /= slope_map.max()

        for step in range(steps):
            perception = self.perceive(state)
            model_input = torch.cat([perception, static], dim=1)
            update = self.w2(torch.nn.functional.relu(self.w1(model_input)))

            if use_physics:
                resistance = 1.0 - (slope_map * 3.0)
                update = update * torch.clamp(resistance, 0.0, 1.0)

            state = state + update
            forest = torch.min(state[:, 0:1], forest_init).clamp(0, 1)
            state = torch.cat([forest, state[:, 1:]], dim=1)

        return state, state[:, 0:1]

# ----------------- Helper Functions -----------------
@st.cache_resource
def load_nca_model(model_path):
    """Loads the pre-trained NCA model onto the CPU."""
    # Here is the fix: Using channels=32 to match your initialized class!
    model = CAModel(channels=32, device='cpu')
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def to_rgb(x_tensor):
    """Converts 1-channel forest state to RGB for visualization"""
    forest_map = x_tensor[0, 0].detach().cpu().numpy()
    
    # Create a simple colormap (Forest goes into the Green channel)
    rgb = np.zeros((forest_map.shape[0], forest_map.shape[1], 3))
    rgb[:, :, 1] = forest_map  
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)

def apply_damage(grid, damage_type="Circle", radius=10):
    """Applies a specific visual deforestation damage."""
    mask = torch.ones_like(grid[:, 0:1])
    h, w = grid.shape[2], grid.shape[3]
    cy, cx = h // 2, w // 2
    
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    dist = torch.sqrt((x - cx)**2 + (y - cy)**2)
    
    if damage_type == "Circle":
        mask[:, :, dist < radius] = 0.0
    elif damage_type == "Half Cut":
        mask[:, :, y > cy] = 0.0
        
    grid[:, 0:1] = grid[:, 0:1] * mask
    return grid

# ----------------- Streamlit UI Configuration -----------------
st.set_page_config(page_title="NCA Deforestation Dashboard", layout="wide")

st.title("🌲 Neural Cellular Automata (NCA) Deforestation")
st.markdown("### Developed by: **Muhammad Haris, Muhammad Ahsan Shaikh, Muhammad Abdullah**")

MODEL_PATH = "nca_best_of_both.pth"

try:
    nca_model = load_nca_model(MODEL_PATH)
    st.sidebar.success("✅ Pre-trained model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model file '{MODEL_PATH}'. Details: {e}")
    st.stop()

# Sidebar Controls
st.sidebar.header("🎛️ Simulation Configurations")
grid_size = st.sidebar.slider("Grid Resolution (Square Size)", min_value=32, max_value=64, value=64, step=8)
num_steps = st.sidebar.slider("NCA Iteration Steps", min_value=1, max_value=100, value=32, step=1)

st.sidebar.header("💥 Deforestation Settings")
enable_damage = st.sidebar.checkbox("Inflict Deforestation Damage", value=False)
damage_type = st.sidebar.selectbox("Damage Shape Pattern", ["Circle", "Half Cut"], disabled=not enable_damage)
damage_radius = st.sidebar.slider("Damage Impact Radius", min_value=5, max_value=20, value=10, disabled=(not enable_damage or damage_type=="Half Cut"))

# Core Session state
if 'ca_input' not in st.session_state or st.sidebar.button("♻️ Reset Landscape"):
    # Input has 3 channels: [Forest, Roads, Elevation]
    init_grid = torch.zeros((1, 3, grid_size, grid_size))
    init_grid[:, 0, :, :] = 1.0  # Fully forested initially
    
    st.session_state.ca_input = init_grid
    st.session_state.current_forest = init_grid[:, 0:1].clone()
    st.session_state.history = [to_rgb(st.session_state.current_forest)]

col1, col2 = st.columns(2)

with col1:
    st.subheader("📺 Interactive Execution Workspace")
    
    if st.button("🚀 Run Prediction", use_container_width=True):
        current_input = st.session_state.ca_input.clone()
        
        # Apply damage to the current forest state before running
        if enable_damage:
            current_input = apply_damage(current_input, damage_type, damage_radius)
            st.warning(f"⚠️ Applied '{damage_type}' deforestation to the landscape.")
            st.session_state.ca_input = current_input
        
        with st.spinner(f"Simulating {num_steps} cellular timesteps..."):
            with torch.no_grad():
                # Forward returns (full_state, forest_prediction)
                _, predicted_forest = nca_model(current_input, steps=num_steps, use_physics=True)
            
            # Update the history and current forest state
            st.session_state.ca_input[:, 0:1] = predicted_forest
            st.session_state.current_forest = predicted_forest
            st.session_state.history.append(to_rgb(predicted_forest))
            
    # Always render current status frame
    current_img = to_rgb(st.session_state.current_forest)
    upscaled_img = cv2.resize(current_img, (300, 300), interpolation=cv2.INTER_NEAREST)
    st.image(upscaled_img, caption="Current Forest State", use_column_width=False)

with col2:
    st.subheader("⏳ Environmental History")
    st.write(f"Total recorded key frames: {len(st.session_state.history)}")
    
    if len(st.session_state.history) > 1:
        frame_idx = st.slider("Browse Timeline History", 0, len(st.session_state.history)-1, len(st.session_state.history)-1)
        selected_img = st.session_state.history[frame_idx]
        upscaled_hist = cv2.resize(selected_img, (300, 300), interpolation=cv2.INTER_NEAREST)
        st.image(upscaled_hist, caption=f"Snapshot Frame #{frame_idx}", use_column_width=False)
    else:
        st.info("Run the simulation to generate timeline frames.")