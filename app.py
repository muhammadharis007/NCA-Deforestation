import streamlit as st
import torch
import numpy as np
import cv2
import time
from streamlit_drawable_canvas import st_canvas
from perlin_noise import PerlinNoise

# ----------------- Model Architecture (ResearchNCA) -----------------
class CAModel(torch.nn.Module):
    def __init__(self, channels=32, device='cpu'):
        super().__init__()
        self.device = device
        self.channels = channels
        self.sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3) / 8.0
        self.sobel_y = self.sobel_x.transpose(2,3)
        self.register_buffer('Kx', self.sobel_x)
        self.register_buffer('Ky', self.sobel_y)
        self.perceive_conv = torch.nn.Conv2d(channels, channels, 3, padding=1)
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

    def forward(self, state, steps=1, use_physics=True):
        forest_init = state[:, 0:1].clone()
        static = state[:, 1:3]
        
        slope_map = self.get_slope(state)
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
            # Enforce that forest cannot grow beyond its initial bound (deforestation only)
            forest = torch.min(state[:, 0:1], forest_init).clamp(0, 1)
            state = torch.cat([forest, state[:, 1:]], dim=1)

        return state, state[:, 0:1]

# ----------------- Helper Functions -----------------
@st.cache_resource
def load_nca_model(model_path):
    model = CAModel(channels=32, device='cpu')
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def generate_procedural_forest(grid_size):
    """Generates a natural-looking procedural forest layout using Perlin Noise."""
    noise = PerlinNoise(octaves=4, seed=np.random.randint(0, 1000))
    forest_map = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            val = noise([i/grid_size, j/grid_size])
            # Threshold noise to create dense forest patches
            forest_map[i, j] = 1.0 if val > -0.05 else 0.0 
    return torch.tensor(forest_map, dtype=torch.float32).view(1, 1, grid_size, grid_size)

def state_to_rgb(forest_tensor, roads_tensor):
    """Converts the NCA state to an RGB image (Green = Forest, Gray = Roads)"""
    forest = forest_tensor[0, 0].detach().cpu().numpy()
    roads = roads_tensor[0, 0].detach().cpu().numpy()
    
    img = np.zeros((forest.shape[0], forest.shape[1], 3))
    img[:, :, 1] = forest  # Green channel for forest
    
    # Overlay roads as gray
    road_mask = roads > 0.5
    img[road_mask] = [0.5, 0.5, 0.5] 
    
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)

# ----------------- Streamlit UI Configuration -----------------
st.set_page_config(page_title="NCA Deforestation Simulator", layout="wide")
st.title("🌲 Interactive NCA Deforestation Simulator")

MODEL_PATH = "nca_best_of_both.pth"
nca_model = load_nca_model(MODEL_PATH)
GRID_SIZE = 64

# Initialize session state
if 'base_state' not in st.session_state:
    st.session_state.base_state = torch.zeros((1, 32, GRID_SIZE, GRID_SIZE))
    st.session_state.base_state[:, 0:1] = generate_procedural_forest(GRID_SIZE) # Procedural Forest
    st.session_state.base_state[:, 2:3] = 0.5 # Flat elevation default

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Draw Infrastructure (Roads/Logging Pads)")
    st.write("Draw gray lines to simulate human roads cutting into the forest.")
    
    # Render the current forest as the background of the canvas
    bg_image = state_to_rgb(st.session_state.base_state[:, 0:1], torch.zeros((1,1,GRID_SIZE,GRID_SIZE)))
    bg_image_resized = cv2.resize(bg_image, (400, 400), interpolation=cv2.INTER_NEAREST)
    
    # Interactive Drawing Pad
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color
        stroke_width=st.slider("Road Thickness", 1, 10, 3),
        stroke_color="#808080", # Gray for roads
        background_image=fromarray(bg_image_resized) if 'fromarray' in globals() else None,
        update_streamlit=True,
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("2. Live Simulation Animation")
    simulation_steps = st.slider("Animation Frames (Steps)", 10, 200, 60)
    
    if st.button("▶️ Play Deforestation Simulation", use_container_width=True):
        # 1. Extract drawn roads from canvas
        drawn_roads = np.zeros((GRID_SIZE, GRID_SIZE))
        if canvas_result.image_data is not None:
            # Resize canvas output back to grid size
            canvas_img = cv2.resize(canvas_result.image_data, (GRID_SIZE, GRID_SIZE))
            # Extract gray pixels as road paths
            drawn_roads = (canvas_img[:, :, 0] > 0).astype(np.float32)
        
        # 2. Setup initial state for this run
        run_state = st.session_state.base_state.clone()
        run_state[:, 1:2] = torch.tensor(drawn_roads).view(1, 1, GRID_SIZE, GRID_SIZE) # Inject roads
        
        # 3. Create a placeholder for GIF-like animation
        animation_placeholder = st.empty()
        
        # 4. Live animation loop
        for step in range(simulation_steps):
            with torch.no_grad():
                # Step the model forward by exactly 1 tick
                run_state, forest_pred = nca_model(run_state, steps=1, use_physics=True)
            
            # Render current frame
            frame = state_to_rgb(forest_pred, run_state[:, 1:2])
            frame_resized = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_NEAREST)
            
            # Update the image in place to create GIF effect
            animation_placeholder.image(frame_resized, caption=f"Tick: {step+1} / {simulation_steps}")
            
            # Tiny sleep to make the animation viewable
            time.sleep(0.05)
            
        st.success("Simulation Complete!")

# Add Pillow image handler globally for canvas background
from PIL.Image import fromarray