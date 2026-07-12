import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ==========================================
# 1. ARCHITECTURE (Must match trained model)
# ==========================================
class ResearchNCA(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.channels = channels
        self.sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)/8.0
        self.sobel_y = self.sobel_x.transpose(2,3)
        self.register_buffer('Kx', self.sobel_x)
        self.register_buffer('Ky', self.sobel_y)
        self.perceive_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.w1 = nn.Conv2d((channels*4)+2, 128, 1)
        self.w2 = nn.Conv2d(128, channels, 1)

    def perceive(self, state):
        k = state.shape[1]
        x_grad = F.conv2d(state, self.Kx.repeat(k,1,1,1), padding=1, groups=k)
        y_grad = F.conv2d(state, self.Ky.repeat(k,1,1,1), padding=1, groups=k)
        learned = self.perceive_conv(state)
        return torch.cat([state, x_grad, y_grad, learned], dim=1)

    def get_slope(self, state):
        elev = state[:, 2:3]
        dx = F.conv2d(elev, self.Kx, padding=1)
        dy = F.conv2d(elev, self.Ky, padding=1)
        return torch.sqrt(dx**2 + dy**2)

    def forward(self, x, steps=32, impact_factor=1.0, use_physics=True):
        forest_init = x[:, 0:1]
        static = x[:, 1:3]
        b, _, h, w = forest_init.shape
        hidden = torch.zeros(b, self.channels - 1, h, w, device=x.device)
        state = torch.cat([forest_init, hidden], dim=1)
        
        slope_map = self.get_slope(x)
        if slope_map.max() > 0: slope_map /= slope_map.max()

        for step in range(steps):
            perception = self.perceive(state)
            model_input = torch.cat([perception, static], dim=1)
            update = self.w2(F.relu(self.w1(model_input)))
            
            if use_physics:
                resistance = 1.0 - (slope_map * 3.0) 
                update = update * torch.clamp(resistance, 0.0, 1.0)

            state = state + (update * 0.5 * impact_factor)
            state = torch.clamp(state, -1.0, 1.0)
            
            forest = torch.min(state[:, 0:1], forest_init).clamp(0, 1)
            state = torch.cat([forest, state[:, 1:]], dim=1)

        return state[:, 0:1], slope_map

# ==========================================
# 2. GENERATOR
# ==========================================
def generate_complex_terrain(size=64, seed=None):
    if seed: np.random.seed(seed)
    # Forest
    low_res_f = np.random.rand(8, 8)
    forest_noise = np.array(Image.fromarray(low_res_f).resize((size, size), Image.BICUBIC))
    forest_channel = (forest_noise > 0.4).astype(np.float32)
    # Elevation
    low_res_e = np.random.rand(4, 4)
    elev_noise = np.array(Image.fromarray(low_res_e).resize((size, size), Image.BICUBIC))
    elev_channel = elev_noise ** 2
    elev_channel[elev_channel < 0.2] = 0.0
    if elev_channel.max() > 0:
        elev_channel = (elev_channel - elev_channel.min()) / (elev_channel.max() - elev_channel.min())
    # Roads
    roads_channel = np.zeros((size, size), dtype=np.float32)
    if np.random.rand() > 0.4:
        x = np.random.randint(10, 54)
        roads_channel[:, x-1:x+1] = 1.0
        forest_channel[roads_channel > 0.5] = 0.0

    patch = np.zeros((3, size, size), dtype=np.float32)
    patch[0] = forest_channel
    patch[1] = roads_channel
    patch[2] = elev_channel
    return patch

# ==========================================
# 3. APP LOGIC
# ==========================================
st.set_page_config(layout="wide", page_title="NCA Lab")

@st.cache_resource
def load_resources():
    model = ResearchNCA(channels=32)
    status = "Model Loaded"
    
    # TRY TO LOAD WEIGHTS
    model_path = "nca_best_of_both.pth"
    
    if os.path.exists(model_path):
        try:
            # map_location='cpu' is critical for local inference!
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        except Exception as e:
            status = f"Weight Error: {e}"
    else:
        status = "⚠️ No weights found. Using random init (Untrained)."
        
    model.eval()
    
    patch = generate_complex_terrain() 
    tensor_patch = torch.nn.functional.interpolate(
        torch.tensor(patch).unsqueeze(0).float(), size=(64,64)
    )
    
    f_vis = np.clip(patch[0], 0, 1) * 255
    r_vis = np.clip(patch[1], 0, 1) * 255
    e_vis = np.clip(patch[2], 0, 1) * 255
    
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    rgb[:, :, 1] = torch.nn.functional.interpolate(torch.tensor(f_vis).unsqueeze(0).unsqueeze(0), size=(200,200)).squeeze().numpy().astype(np.uint8)
    rgb[:, :, 0] = torch.nn.functional.interpolate(torch.tensor(r_vis).unsqueeze(0).unsqueeze(0), size=(200,200)).squeeze().numpy().astype(np.uint8)
    rgb[:, :, 2] = torch.nn.functional.interpolate(torch.tensor(e_vis).unsqueeze(0).unsqueeze(0), size=(200,200)).squeeze().numpy().astype(np.uint8) // 3
    
    bg_image = Image.fromarray(rgb)
    return model, bg_image, tensor_patch, status

model, bg_image, real_tensor, status_msg = load_resources()

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption(f"Status: {status_msg}")
    frames = st.slider("Time Horizon", 1, 10, 5)
    impact = st.slider("Road Impact", 0.4, 1.0, 1.0)
    steps = st.slider("Speed", 1, 5, 2)
    use_physics = st.toggle("Slope Physics", value=True)
    if st.button("New Map"): st.cache_resource.clear(); st.rerun()

st.title("🛰️ NCA Deforestation Lab")
col1, col2 = st.columns([1, 1])

# CANVAS
with col1:
    st.subheader("1. Input")
    st.image(bg_image, caption="Reference (Blue=Elevation)", width=300)
    try:
        from streamlit_drawable_canvas import st_canvas
        # Canvas Logic
        canvas = st_canvas(
            fill_color="rgba(255, 0, 0, 0.5)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_color="#000000", # Black background for easy parsing
            height=300, width=300,
            drawing_mode="freedraw", key="canvas"
        )
    except: st.error("Install streamlit-drawable-canvas")

# PREDICTION
with col2:
    st.subheader("2. Forecast")
    output_container = st.empty()
    stats = st.container()
    
    if st.button("Run Simulation", type="primary"):
        import tempfile, imageio
        
        # 1. User Input parsing
        new_roads = torch.zeros(1, 1, 64, 64)
        if canvas.image_data is not None:
             # Resize canvas to 64x64
            user_draw = Image.fromarray(canvas.image_data.astype('uint8')).resize((64, 64), Image.NEAREST)
            user_arr = np.array(user_draw)
            # Detect red lines (Channel 0 > 50)
            if user_arr.shape[2] >= 3:
                new_roads = torch.tensor((user_arr[:,:,0]>50)).float().unsqueeze(0).unsqueeze(0)
        
        # 2. Setup
        curr_x = real_tensor.clone()
        # Physics: Road destroys forest instantly
        road_mask = (torch.max(curr_x[:,1], new_roads) > 0.5).float()
        curr_x[:,1] = road_mask
        curr_x[:,0] = curr_x[:,0] * (1.0 - road_mask)
        
        initial = curr_x[0,0].numpy().copy()
        frames_list = []
        
        # 3. Loop
        for i in range(frames):
            with torch.no_grad():
                curr_x[:,0:1], _ = model(curr_x, steps=steps, impact_factor=impact, use_physics=use_physics)
            
            # Render
            f = curr_x[0,0].numpy()
            r = curr_x[0,1].numpy()
            
            img = np.zeros((64,64,3), dtype=np.uint8) + [20,30,50]
            mask_f = np.clip(f, 0, 1)[:, :, None]
            img = img * (1-mask_f) + np.array([34, 139, 34]) * mask_f # Green
            
            loss = np.clip(initial - f, 0, 1)
            loss[loss<0.1] = 0
            img[:, :, 0] = np.maximum(img[:, :, 0], (loss*255).astype(np.uint8)) # Red Fire
            
            img[r>0.5] = [255,255,255] # White Roads
            
            pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((300,300), Image.NEAREST)
            frames_list.append(pil)
            
        # Display
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
            np_frames = [np.array(f) for f in frames_list]
            imageio.mimsave(tmpfile.name, np_frames, fps=15)
            output_container.image(tmpfile.name, caption="Forecast")
            
        final_loss = ((np.sum(initial) - np.sum(f)) / np.sum(initial) * 100) if np.sum(initial) > 0 else 0
        stats.metric("Projected Loss", f"{final_loss:.1f}%", delta_color="inverse")