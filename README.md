# Physics-Informed & Latent Space Neural Cellular Automata (NCA) for Deforestation Modeling

## 📌 Project Overview
This project applies **Neural Cellular Automata (NCA)** to simulate and predict deforestation dynamics in the Amazon rainforest (Rondônia, Brazil). Moving beyond traditional pixel-based deep learning, this research introduces a "Physics-Informed" approach and subsequently evolves into a **Latent Space Denoising (LSD)** architecture to handle noise and capture high-level abstract dynamics.

## 👥 Authors
* **Muhammad Haris**
* **Muhammad Abdullah**
* **Muhammad Ahsan**

---

## 🏗️ System Architecture: The "Hybrid Pipeline"
To overcome the computational rate limits of **Google Earth Engine (GEE)** while maintaining high-throughput training for PyTorch, we implemented a decoupled architecture:

1.  **Distributed Backend (GEE):** Acts as the **OLAP (Online Analytical Processing)** engine. It handles petabyte-scale queries for Sentinel-2 imagery, forest cover change (Hansen/UMD), and terrain data (SRTM).
2.  **Intermediate Feature Store (MongoDB):** We integrated a NoSQL database to act as a caching layer. Processed tensors and static feature maps are extracted from GEE and cached locally.
3.  **Training Pipeline (PyTorch):** The model consumes data directly from the MongoDB cache, ensuring GPU utilization is not bottlenecked by API latency.

---

## 📂 Project Phases & Models

### Phase 1: Standard Physics-Informed NCA (`nca_lab_final.ipynb`)
The initial iteration of the project.
* **Mechanism:** Operates directly on raw satellite pixels.
* **Physics Constraints:** Incorporates environmental variables like **Terrain Slope** (derived via Sobel filters) to create "resistance" maps that prevent unrealistic deforestation uphill.
* **Limitation:** Performance was hindered by sensor noise in raw pixel inputs.

### Phase 2: LSD-NCA (Latent Space Denoising) (`LSD_NCA.ipynb`)
The advanced extension of the project, significantly outperforming previous baselines.

#### 🧠 Concept: Learning in Latent Space
Instead of simulating cellular automata on noisy RGB pixels, the LSD-NCA utilizes an **Encoder-Decoder** architecture to move the simulation into a compressed "Latent Space." This is conceptually similar to **"World Models" (Ha et al., 2018)**, where agents learn dynamics in a simplified abstract representation.

1.  **Encoder:** Compresses the noisy input (64x64 satellite patches) into abstract, high-level features using convolutional blocks.
2.  **Latent NCA Dynamics:** The Cellular Automata update rules run inside this compressed representation. The model learns the "physics" of deforestation concepts (e.g., "expansion pressure," "road proximity effects") rather than just color changes.
3.  **Decoder:** Reconstructs the evolved latent state back into a viable deforestation probability map.

---

## 📊 Benchmarks & Results
We benchmarked the LSD-NCA against our initial Physics-Informed NCA and a standard Random Forest Regressor.

| Model Architecture | IoU (Intersection over Union) | Status |
| :--- | :--- | :--- |
| **Standard NCA (Phase 1)** | 0.81 | Baseline |
| **Random Forest** | 0.89 | Strong Baseline |
| **LSD-NCA (Phase 2)** | **0.916** | **New SOTA 🏆** |

The LSD-NCA achieves state-of-the-art performance for this specific task by effectively denoising the input and learning robust transition rules in the latent space.

---

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch (Custom NCA Layers, Autoencoders)
* **Spatial Database:** Google Earth Engine (Compute), MongoDB (Storage/Caching)
* **Visualization:** Streamlit (Interactive Dashboard), Matplotlib
* **Geospatial Processing:** Rasterio, earthengine-api

## ⚙️ Setup & Installation
1.  **Install Dependencies:**
    ```bash
    pip install torch torchvision earthengine-api rasterio pymongo
    pip install streamlit streamlit-drawable-canvas
    pip install scikit-learn matplotlib pandas
    ```
2.  **Database Setup:**
    Ensure a local or cloud instance of MongoDB is running if executing the full training pipeline in `LSD_NCA.ipynb`.
3.  **GEE Auth:**
    Authenticate with Google Earth Engine using `earthengine authenticate` in your terminal or via the pop-up in the notebook.

## 🚀 Usage
* **Research Replication:** Run `nca_lab_final.ipynb` to observe the baseline physics-informed approach.
* **LSD Architecture:** Run `LSD_NCA.ipynb` to train the Encoder-Decoder NCA model or use the interactive interface (if enabled) to visualize latent space transitions.
