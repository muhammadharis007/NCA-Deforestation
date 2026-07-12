import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
from perlin_noise import PerlinNoise
from PIL import Image

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==============================================================================
st.set_page_config(
    page_title="NCA Deforestation Simulator | Advanced Dashboard",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for a more professional dashboard feel
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #2E8B57; margin-bottom: 0px; }
    .sub-header { font-size: 1.2rem; font-weight: 400; color: #555555; margin-bottom: 30px; }
    .metric-card { background-color: #1E1E1E; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stProgress .st-bo { background-color: #2E8B57; }
    div[data-testid="stImage"] img { border: 2px solid #333; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. NEURAL CELLULAR AUTOMATA ARCHITECTURE (RESEARCH NCA)
# ==============================================================================
class CAModel(torch.nn.Module):
    """
    Neural Cellular Automata designed to simulate deforestation dynamics.
    Utilizes localized perception via Sobel filters and 1x1 convolutions 
    to map complex non-linear spatial interactions.
    """
    def __init__(self, channels=32, device='cpu'):
        super().__init__()
        self.device = device
        self.channels = channels
        
        # ---------------------------------------------------------
        # Spatial Perception Filters (Sobel X and Y)
        # ---------------------------------------------------------
        self.sobel_x = torch.tensor([[-1, 0, 1], 
                                     [-2, 0, 2], 
                                     [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.sobel_y = self.sobel_x.transpose(2, 3)
        self.register_buffer('Kx', self.sobel_x)
        self.register_buffer('Ky', self.sobel_y)

        # ---------------------------------------------------------
        # Learnable Layers
        # ---------------------------------------------------------
        # Perceive convolution WITH bias (matches checkpoint exactly)
        self.perceive_conv = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # 1x1 Convolutions mapping 130 channels -> 128 channels -> 32 channels
        # 130 comes from: (32 original * 4 perception vectors) + 2 static features
        self.w1 = torch.nn.Conv2d((channels * 4) + 2, 128, kernel_size=1)
        self.w2 = torch.nn.Conv2d(128, channels, kernel_size=1)
        
        # Move model to designated device (CPU for Streamlit cloud)
        self.to(device)

    def perceive(self, state):
        """
        Calculates the perception vector for each cell.
        Returns concatenation of: [identity, grad_x, grad_y, learned_perception]
        """
        k = state.shape[1]
        # Calculate spatial gradients
        x_grad = F.conv2d(state, self.Kx.repeat(k, 1, 1, 1), padding=1, groups=k)
        y_grad = F.conv2d(state, self.Ky.repeat(k, 1, 1, 1), padding=1, groups=k)
        # Apply learned spatial filter
        learned = self.perceive_conv(state)
        
        return torch.cat([state, x_grad, y_grad, learned], dim=1)

    def get_slope(self, state):
        """
        Extracts elevation map (channel 2) and calculates geographical slope magnitude.
        Used to calculate resistance to human infrastructure expansion.
        """
        elev = state[:, 2:3]
        dx = F.conv2d(elev, self.Kx, padding=1)
        dy = F.conv2d(elev, self.Ky, padding=1)
        return torch.sqrt(dx**2 + dy**2)

    def forward(self, state, steps=1, use_physics=True):
        """
        Executes the cellular automata simulation forward in time.
        """
        # Channel 0: Forest Cover (Dynamic)
        # Channel 1: Roads/Infrastructure (Static)
        # Channel 2: Elevation (Static)
        forest_init = state[:, 0:1].clone()
        static_features = state[:, 1:3]
        
        # Calculate physics-based resistance once per run
        slope_map = self.get_slope(state)
        if slope_map.max() > 0: 
            slope_map /= slope_map.max()

        for step in range(steps):
            # 1. Perception Phase
            perception = self.perceive(state)
            model_input = torch.cat([perception, static_features], dim=1)
            
            # 2. Update Phase
            hidden = F.relu(self.w1(model_input))
            update = self.w2(hidden)

            # 3. Physics & Constraints Application
            if use_physics:
                # Steep slopes resist deforestation expansion
                resistance = 1.0 - (slope_map * 3.0)
                update = update * torch.clamp(resistance, 0.0, 1.0)

            # 4. State Modification
            state = state + update
            
            # 5. Natural Constraints (Forests can't regrow in this strict model, only degrade)
            forest = torch.min(state[:, 0:1], forest_init).clamp(0, 1)
            
            # Reconstruct full state tensor
            state = torch.cat([forest, state[:, 1:]], dim=1)

        return state, state[:, 0:1]

# ==============================================================================
# 3. HELPER FUNCTIONS & PROCEDURAL GENERATION
# ==============================================================================
@st.cache_resource
def load_nca_model(model_path):
    """Safely loads the PyTorch weights into the CAModel architecture."""
    try:
        model = CAModel(channels=32, device='cpu')
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model weights: {e}")
        st.stop()

def generate_procedural_environment(grid_size, seed_offset=0):
    """
    Generates complex procedural terrain using multi-octave Perlin noise.
    Creates both a Forest Map and an Elevation Map.
    """
    noise1 = PerlinNoise(octaves=4, seed=np.random.randint(0, 1000) + seed_offset)
    noise2 = PerlinNoise(octaves=8, seed=np.random.randint(1000, 2000) + seed_offset)
    
    forest_map = np.zeros((grid_size, grid_size))
    elevation_map = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Generate Forest (Dense patches with some clearings)
            f_val = noise1([i/grid_size, j/grid_size]) + 0.5 * noise2([i/grid_size, j/grid_size])
            forest_map[i, j] = 1.0 if f_val > -0.1 else 0.0
            
            # Generate Elevation (Smoother gradients)
            e_val = noise1([j/grid_size, i/grid_size]) # Swapped axes for variation
            elevation_map[i, j] = np.clip((e_val + 0.5), 0.0, 1.0)
            
    # Convert to PyTorch tensors
    f_tensor = torch.tensor(forest_map, dtype=torch.float32).view(1, 1, grid_size, grid_size)
    e_tensor = torch.tensor(elevation_map, dtype=torch.float32).view(1, 1, grid_size, grid_size)
    
    return f_tensor, e_tensor

def state_to_rgb(forest_tensor, roads_tensor, elevation_tensor=None, show_elevation=False):
    """
    Transforms the abstract mathematical tensors into a visual RGB image.
    Handles blending of Forest, Roads, and optionally Elevation overlays.
    """
    forest = forest_tensor[0, 0].detach().cpu().numpy()
    roads = roads_tensor[0, 0].detach().cpu().numpy()
    
    img = np.zeros((forest.shape[0], forest.shape[1], 3))
    
    if show_elevation and elevation_tensor is not None:
        # Render elevation as a topological background (brown/yellow hints)
        elev = elevation_tensor[0, 0].detach().cpu().numpy()
        img[:, :, 0] = elev * 0.4 # Red channel
        img[:, :, 1] = elev * 0.3 # Green channel
        img[:, :, 2] = elev * 0.1 # Blue channel
        
        # Overlay forest heavily where it exists
        img[:, :, 1] += forest * 0.7 
    else:
        # Standard rendering: Black background, Green forest
        img[:, :, 1] = forest  
    
    # Overlay infrastructure/roads (Bright Gray/White)
    road_mask = roads > 0.5
    img[road_mask] = [0.8, 0.8, 0.8] 
    
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)

def calculate_forest_area(forest_tensor):
    """Calculates the exact percentage of the grid covered by forest."""
    forest_array = forest_tensor[0, 0].detach().cpu().numpy()
    total_pixels = forest_array.size
    active_pixels = np.sum(forest_array > 0.5)
    return (active_pixels / total_pixels) * 100.0

# ==============================================================================
# 4. UI SETUP & STATE MANAGEMENT
# ==============================================================================
st.markdown('<p class="main-header">🌲 Advanced NCA Deforestation Simulator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Research Dashboard | Developed by: Haris, Ahsan, Abdullah</p>', unsafe_allow_html=True)

MODEL_PATH = "nca_best_of_both.pth"
nca_model = load_nca_model(MODEL_PATH)

# Global configuration constants
GRID_SIZE = 64
CANVAS_DISPLAY_SIZE = 400

# Initialize core session state if it doesn't exist
if 'environment_seed' not in st.session_state:
    st.session_state.environment_seed = 0
    st.session_state.base_state = torch.zeros((1, 32, GRID_SIZE, GRID_SIZE))
    f_map, e_map = generate_procedural_environment(GRID_SIZE, seed_offset=0)
    st.session_state.base_state[:, 0:1] = f_map
    st.session_state.base_state[:, 2:3] = e_map
    st.session_state.metrics_log = []

# Sidebar configurations
with st.sidebar:
    st.title("🎛️ Simulation Controls")
    
    st.header("1. Environment Initialization")
    if st.button("🎲 Generate New Terrain", use_container_width=True):
        st.session_state.environment_seed += 1
        f_map, e_map = generate_procedural_environment(GRID_SIZE, seed_offset=st.session_state.environment_seed)
        st.session_state.base_state[:, 0:1] = f_map
        st.session_state.base_state[:, 2:3] = e_map
        st.session_state.metrics_log = [] # Reset metrics
        st.rerun()
        
    render_elevation = st.checkbox("Show Elevation Map Overlay", value=False)
    
    st.header("2. Dynamics Settings")
    simulation_steps = st.slider("Total Ticks (Frames)", min_value=10, max_value=300, value=100, step=10)
    animation_speed = st.slider("Animation Speed", min_value=0.01, max_value=0.2, value=0.05, format="%.2fs")
    use_physics = st.checkbox("Enable Topographical Resistance", value=True, help="If enabled, steep elevations will slow down deforestation expansion.")
    
    st.header("3. Drawing Tools")
    brush_size = st.slider("Brush Thickness", min_value=1, max_value=10, value=3)
    
    st.markdown("---")
    st.markdown("**About this project:**\nThis simulator uses Neural Cellular Automata (a decentralized AI architecture) to predict the emergent, decentralized spread of deforestation based on initial infrastructure placement and geographic topology.")

# ==============================================================================
# 5. MAIN DASHBOARD WORKSPACE
# ==============================================================================
col1, col2 = st.columns([1.2, 1])

# Left Column: Interactive Drawing Pad
with col1:
    st.subheader("Step 1: Plan Infrastructure")
    st.write("Draw roads or logging camps on the canvas. The NCA will use this as the seed for deforestation.")
    
    # Render background for the canvas
    bg_image = state_to_rgb(
        st.session_state.base_state[:, 0:1], 
        torch.zeros((1, 1, GRID_SIZE, GRID_SIZE)),
        st.session_state.base_state[:, 2:3],
        render_elevation
    )
    bg_image_resized = cv2.resize(bg_image, (CANVAS_DISPLAY_SIZE, CANVAS_DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
    
    # Display the interactive canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", 
        stroke_width=brush_size,
        stroke_color="#FFFFFF", # Drawing in white maps to roads in our array processing
        background_image=Image.fromarray(bg_image_resized),
        update_streamlit=True,
        height=CANVAS_DISPLAY_SIZE,
        width=CANVAS_DISPLAY_SIZE,
        drawing_mode="freedraw",
        key="infrastructure_canvas",
    )
    
    # Calculate initial stats
    initial_forest = calculate_forest_area(st.session_state.base_state[:, 0:1])
    st.metric(label="Initial Forest Cover", value=f"{initial_forest:.1f}%")

# Right Column: Live Simulation Execution
with col2:
    st.subheader("Step 2: Execute NCA Simulation")
    
    # UI Elements for the live animation
    animation_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Display the static initial state before running
    if not canvas_result.image_data is None:
        animation_placeholder.image(bg_image_resized, caption="Awaiting Simulation...", use_column_width=True)

    if st.button("▶️ Launch Predictive Simulation", use_container_width=True, type="primary"):
        # 1. Process the user's drawing
        drawn_roads = np.zeros((GRID_SIZE, GRID_SIZE))
        if canvas_result.image_data is not None:
            # Resize the high-res canvas drawing down to our 64x64 tensor grid
            canvas_img = cv2.resize(canvas_result.image_data, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_NEAREST)
            # Alpha channel > 0 means the user drew there
            drawn_roads = (canvas_img[:, :, 3] > 0).astype(np.float32)
            
        # 2. Setup the state tensor for the run
        run_state = st.session_state.base_state.clone()
        # Inject the drawn roads into Channel 1
        run_state[:, 1:2] = torch.tensor(drawn_roads).view(1, 1, GRID_SIZE, GRID_SIZE) 
        
        # Reset logs for new run
        st.session_state.metrics_log = []
        
        # 3. Execution Loop
        start_time = time.time()
        for step in range(simulation_steps):
            with torch.no_grad():
                # Step the model forward exactly 1 tick
                run_state, forest_pred = nca_model(run_state, steps=1, use_physics=use_physics)
            
            # Record analytics
            current_forest_area = calculate_forest_area(forest_pred)
            st.session_state.metrics_log.append({
                "Tick": step + 1,
                "Forest Cover (%)": current_forest_area
            })
            
            # Visual Rendering
            frame = state_to_rgb(forest_pred, run_state[:, 1:2], run_state[:, 2:3], render_elevation)
            frame_resized = cv2.resize(frame, (CANVAS_DISPLAY_SIZE, CANVAS_DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
            
            # Update UI dynamically (Creates the GIF effect)
            animation_placeholder.image(frame_resized, caption=f"Processing Tick: {step+1} / {simulation_steps}", use_column_width=True)
            progress_bar.progress((step + 1) / simulation_steps)
            status_text.text(f"Live Forest Cover: {current_forest_area:.1f}%")
            
            # Sleep dictates the visual framerate
            time.sleep(animation_speed)
            
        end_time = time.time()
        st.success(f"Simulation Complete in {end_time - start_time:.2f} seconds!")

# ==============================================================================
# 6. POST-SIMULATION ANALYTICS DASHBOARD
# ==============================================================================
st.markdown("---")
st.subheader("📊 Post-Simulation Analytics")

if len(st.session_state.metrics_log) > 0:
    # Convert recorded metrics to a Pandas DataFrame
    df_metrics = pd.DataFrame(st.session_state.metrics_log)
    
    # Calculate overall loss
    initial_cov = df_metrics.iloc[0]["Forest Cover (%)"]
    final_cov = df_metrics.iloc[-1]["Forest Cover (%)"]
    total_loss = initial_cov - final_cov
    
    # Display KPI Cards
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="Starting Cover", value=f"{initial_cov:.2f}%")
    kpi2.metric(label="Ending Cover", value=f"{final_cov:.2f}%", delta=f"-{total_loss:.2f}%", delta_color="inverse")
    kpi3.metric(label="Average Rate of Loss", value=f"{(total_loss / simulation_steps):.3f}% per tick")
    
    # Generate Interactive Plotly Chart
    fig = px.line(
        df_metrics, 
        x="Tick", 
        y="Forest Cover (%)", 
        title="Deforestation Progression Over Time",
        markers=True,
        color_discrete_sequence=["#FF4B4B"]
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Simulation Time (Ticks)",
        yaxis_title="Forest Remaining (%)",
        yaxis=dict(range=[0, 100])
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Allow users to download the data
    csv = df_metrics.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Analytics CSV",
        data=csv,
        file_name='deforestation_metrics.csv',
        mime='text/csv',
    )
else:
    st.info("Run the simulation above to generate environmental impact analytics.")