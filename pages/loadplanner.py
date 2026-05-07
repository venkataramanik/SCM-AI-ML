
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
from itertools import combinations

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Load Planner - Trailer Optimization",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .pallet-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.3rem 0;
        border-left: 4px solid #3498db;
    }
    .trailer-info {
        background: #e8f4f8;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3498db, #2980b9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2980b9, #1f5f8b);
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        color: #155724;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class Pallet:
    id: str
    length: float  # inches
    width: float   # inches
    height: float  # inches
    weight: float  # lbs
    stackable: bool
    fragile: bool
    max_stack_height: int  # how many can stack on top
    delivery_stop: int     # stop sequence (1 = first unload, higher = later)
    sku: str
    customer: str
    color: str

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def floor_area(self) -> float:
        return self.length * self.width

@dataclass
class Trailer:
    length: float      # 636 inches (53ft)
    width: float       # 102 inches (8.5ft)
    height: float      # 108 inches (9ft)
    max_weight: float  # 45000 lbs
    tare_weight: float # 15000 lbs (trailer empty)

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

@dataclass
class Position:
    x: float  # from nose (0) to tail
    y: float  # from left wall (0) to right
    z: float  # from floor (0) up
    pallet: Pallet

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================
def generate_synthetic_orders(num_orders: int = 25, seed: int = 42) -> List[Pallet]:
    random.seed(seed)
    np.random.seed(seed)

    # Standard pallet sizes (inches)
    pallet_types = [
        {"name": "Standard 48x40", "length": 48, "width": 40, "weight_range": (500, 1500), "color": "#3498db"},
        {"name": "Standard 48x48", "length": 48, "width": 48, "weight_range": (800, 2000), "color": "#e74c3c"},
        {"name": "Euro 47.2x39.4", "length": 47.2, "width": 39.4, "weight_range": (400, 1200), "color": "#2ecc71"},
        {"name": "Heavy 48x42", "length": 48, "width": 42, "weight_range": (1500, 2500), "color": "#f39c12"},
        {"name": "Half 24x40", "length": 24, "width": 40, "weight_range": (200, 800), "color": "#9b59b6"},
    ]

    skus = ["SKU-1001", "SKU-1002", "SKU-1003", "SKU-1004", "SKU-1005", 
            "SKU-2001", "SKU-2002", "SKU-3001", "SKU-3002", "SKU-4001"]
    customers = ["Walmart-DC1", "Target-DC2", "HomeDepot-DC3", "Lowes-DC4", "Costco-DC5",
                 "BestBuy-DC6", "Amazon-FC1", "Amazon-FC2", "DollarGen-DC1", "Kroger-DC1"]

    pallets = []
    for i in range(num_orders):
        ptype = random.choice(pallet_types)

        # Height varies by content (30-72 inches)
        height = random.choice([36, 42, 48, 54, 60, 66, 72])

        # Weight based on type
        weight = round(random.uniform(*ptype["weight_range"]), 1)

        # Stackability logic
        stackable = random.random() > 0.3  # 70% stackable
        fragile = random.random() > 0.85  # 15% fragile
        max_stack = random.randint(1, 3) if stackable else 1

        # Delivery stops (1-5, where 1 is first drop, 5 is last)
        delivery_stop = random.randint(1, 5)

        pallet = Pallet(
            id=f"PLT-{i+1:03d}",
            length=ptype["length"],
            width=ptype["width"],
            height=height,
            weight=weight,
            stackable=stackable,
            fragile=fragile,
            max_stack_height=max_stack,
            delivery_stop=delivery_stop,
            sku=random.choice(skus),
            customer=random.choice(customers),
            color=ptype["color"]
        )
        pallets.append(pallet)

    return pallets

# ============================================================
# LOAD OPTIMIZATION ENGINE
# ============================================================
class LoadOptimizer:
    def __init__(self, trailer: Trailer):
        self.trailer = trailer
        self.positions: List[Position] = []
        self.grid_resolution = 2.0  # 2-inch grid

    def optimize(self, pallets: List[Pallet]) -> Tuple[List[Position], dict]:
        """
        Greedy heuristic with layer-based loading (LIFO-aware)
        Sorts by delivery stop (last delivery first = loaded first at back)
        """
        # Sort: later delivery stops first (they go in first, at the back)
        # Within same stop: larger base area first, heavier first
        sorted_pallets = sorted(
            pallets,
            key=lambda p: (-p.delivery_stop, -p.floor_area, -p.weight)
        )

        positions = []
        occupied = []  # list of (x1, x2, y1, y2, z1, z2) tuples

        total_weight = 0
        total_volume = 0

        for pallet in sorted_pallets:
            # Try to place pallet
            placed = self._place_pallet(pallet, positions, occupied)
            if placed:
                positions.append(placed)
                total_weight += pallet.weight
                total_volume += pallet.volume
                occupied.append(self._get_bounds(placed))

        # Calculate metrics
        unplaced = [p for p in pallets if not any(pos.pallet.id == p.id for pos in positions)]

        metrics = {
            "total_pallets": len(pallets),
            "placed": len(positions),
            "unplaced": len(unplaced),
            "weight_util": (total_weight / self.trailer.max_weight) * 100,
            "volume_util": (total_volume / self.trailer.volume) * 100,
            "floor_util": self._calc_floor_utilization(positions),
            "total_weight": total_weight,
            "total_volume": total_volume,
            "unplaced_pallets": unplaced
        }

        return positions, metrics

    def _place_pallet(self, pallet: Pallet, existing: List[Position], occupied: List) -> Optional[Position]:
        """Find best position using greedy placement with stability checks"""

        # Try orientations (0 and 90 degrees)
        orientations = [
            (pallet.length, pallet.width),
            (pallet.width, pallet.length)
        ]

        best_pos = None
        best_score = float('inf')

        for orient_idx, (p_len, p_wid) in enumerate(orientations):
            # Try positions from back of trailer (x=max) to front, bottom-up
            # Grid search with resolution
            x_positions = np.arange(0, self.trailer.length - p_len + 0.1, self.grid_resolution)
            y_positions = np.arange(0, self.trailer.width - p_wid + 0.1, self.grid_resolution)

            for x in reversed(x_positions):  # Start from back
                for y in y_positions:
                    for z in np.arange(0, self.trailer.height - pallet.height + 0.1, self.grid_resolution):
                        # Check bounds
                        if (x + p_len > self.trailer.length or 
                            y + p_wid > self.trailer.width or 
                            z + pallet.height > self.trailer.height):
                            continue

                        # Check collision
                        if self._collides(x, y, z, p_len, p_wid, pallet.height, occupied):
                            continue

                        # Check weight capacity (rough axle distribution check)
                        if not self._check_weight_distribution(x, p_len, pallet.weight, existing):
                            continue

                        # Check stackability if z > 0
                        if z > 0 and not self._check_stack_support(x, y, z, p_len, p_wid, existing, pallet):
                            continue

                        # Score: prefer back, tight packing, low center of gravity
                        score = (self.trailer.length - x) * 0.5 + z * 2 + y * 0.1

                        if score < best_score:
                            best_score = score
                            # Create new pallet with orientation if rotated
                            if orient_idx == 1:
                                rotated_pallet = Pallet(
                                    id=pallet.id, length=p_wid, width=p_len,
                                    height=pallet.height, weight=pallet.weight,
                                    stackable=pallet.stackable, fragile=pallet.fragile,
                                    max_stack_height=pallet.max_stack_height,
                                    delivery_stop=pallet.delivery_stop,
                                    sku=pallet.sku, customer=pallet.customer,
                                    color=pallet.color
                                )
                                best_pos = Position(x, y, z, rotated_pallet)
                            else:
                                best_pos = Position(x, y, z, pallet)

        return best_pos

    def _collides(self, x, y, z, p_len, p_wid, p_height, occupied):
        """Check if proposed position collides with occupied space"""
        for (ox1, ox2, oy1, oy2, oz1, oz2) in occupied:
            if not (x + p_len <= ox1 or x >= ox2 or 
                    y + p_wid <= oy1 or y >= oy2 or 
                    z + p_height <= oz1 or z >= oz2):
                return True
        return False

    def _get_bounds(self, pos: Position):
        p = pos.pallet
        return (pos.x, pos.x + p.length, 
                pos.y, pos.y + p.width, 
                pos.z, pos.z + p.height)

    def _check_weight_distribution(self, x, p_len, weight, existing):
        """Rough check: keep heavier items over axles (roughly middle-back)"""
        # Simplified: allow all for demo, but could enforce rules
        return True

    def _check_stack_support(self, x, y, z, p_len, p_wid, existing, pallet):
        """Check if there's something to stack on and if it's allowed"""
        # Find what we're stacking on
        support_found = False
        for pos in existing:
            p = pos.pallet
            # Check if this pallet is directly below
            if (abs(pos.z + p.height - z) < 0.1 and  # top of support = bottom of new
                not (pos.x + p.length <= x or pos.x >= x + p_len or
                     pos.y + p.width <= y or pos.y >= y + p_wid)):
                # Check if support allows stacking
                if not p.stackable or p.fragile:
                    return False
                support_found = True

        return support_found

    def _calc_floor_utilization(self, positions: List[Position]) -> float:
        """Calculate floor space used"""
        if not positions:
            return 0
        floor_area = sum(p.pallet.floor_area for p in positions if p.z < 0.1)
        return (floor_area / (self.trailer.length * self.trailer.width)) * 100

# ============================================================
# 3D VISUALIZATION
# ============================================================
def create_3d_visualization(positions: List[Position], trailer: Trailer):
    """Create interactive 3D trailer visualization using Plotly"""

    fig = go.Figure()

    # Trailer wireframe
    trailer_corners = [
        [0, 0, 0], [trailer.length, 0, 0], [trailer.length, trailer.width, 0], [0, trailer.width, 0],
        [0, 0, trailer.height], [trailer.length, 0, trailer.height], 
        [trailer.length, trailer.width, trailer.height], [0, trailer.width, trailer.height]
    ]

    edges = [
        [0,1], [1,2], [2,3], [3,0],  # bottom
        [4,5], [5,6], [6,7], [7,4],  # top
        [0,4], [1,5], [2,6], [3,7]   # vertical
    ]

    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=[trailer_corners[edge[0]][0], trailer_corners[edge[1]][0]],
            y=[trailer_corners[edge[0]][1], trailer_corners[edge[1]][1]],
            z=[trailer_corners[edge[0]][2], trailer_corners[edge[1]][2]],
            mode='lines',
            line=dict(color='gray', width=2),
            hoverinfo='skip',
            showlegend=False
        ))

    # Add floor grid
    for i in range(0, int(trailer.length) + 1, 48):
        fig.add_trace(go.Scatter3d(
            x=[i, i], y=[0, trailer.width], z=[0, 0],
            mode='lines', line=dict(color='lightgray', width=1),
            hoverinfo='skip', showlegend=False
        ))
    for i in range(0, int(trailer.width) + 1, 40):
        fig.add_trace(go.Scatter3d(
            x=[0, trailer.length], y=[i, i], z=[0, 0],
            mode='lines', line=dict(color='lightgray', width=1),
            hoverinfo='skip', showlegend=False
        ))

    # Color map for delivery stops
    stop_colors = {1: '#e74c3c', 2: '#f39c12', 3: '#f1c40f', 4: '#2ecc71', 5: '#3498db'}

    # Add pallets as 3D boxes
    for pos in positions:
        p = pos.pallet

        # Box vertices
        vertices = [
            [pos.x, pos.y, pos.z],
            [pos.x + p.length, pos.y, pos.z],
            [pos.x + p.length, pos.y + p.width, pos.z],
            [pos.x, pos.y + p.width, pos.z],
            [pos.x, pos.y, pos.z + p.height],
            [pos.x + p.length, pos.y, pos.z + p.height],
            [pos.x + p.length, pos.y + p.width, pos.z + p.height],
            [pos.x, pos.y + p.width, pos.z + p.height]
        ]

        # Box faces
        faces = [
            [0,1,2,3], [4,5,6,7], [0,1,5,4],
            [2,3,7,6], [0,3,7,4], [1,2,6,5]
        ]

        for face in faces:
            fig.add_trace(go.Mesh3d(
                x=[vertices[i][0] for i in face],
                y=[vertices[i][1] for i in face],
                z=[vertices[i][2] for i in face],
                color=stop_colors.get(p.delivery_stop, p.color),
                opacity=0.85,
                hovertemplate=f"""
                <b>Pallet: {p.id}</b><br>
                SKU: {p.sku}<br>
                Customer: {p.customer}<br>
                Stop: {p.delivery_stop}<br>
                Dimensions: {p.length:.1f}×{p.width:.1f}×{p.height:.1f}<br>
                Weight: {p.weight:.1f} lbs<br>
                Stackable: {'Yes' if p.stackable else 'No'}<br>
                Position: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})
                """,
                showlegend=False,
                flatshading=True
            ))

        # Add edges for definition
        box_edges = [
            [0,1], [1,2], [2,3], [3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7]
        ]
        for edge in box_edges:
            fig.add_trace(go.Scatter3d(
                x=[vertices[edge[0]][0], vertices[edge[1]][0]],
                y=[vertices[edge[0]][1], vertices[edge[1]][1]],
                z=[vertices[edge[0]][2], vertices[edge[1]][2]],
                mode='lines',
                line=dict(color='black', width=1),
                hoverinfo='skip',
                showlegend=False
            ))

    # Layout
    fig.update_layout(
        title=dict(
            text="3D Trailer Load Plan - Interactive View",
            font=dict(size=18, color='#1E3A5F')
        ),
        scene=dict(
            xaxis=dict(title='Length (in) - Nose → Tail', range=[0, trailer.length], dtick=48),
            yaxis=dict(title='Width (in) - Left → Right', range=[0, trailer.width], dtick=40),
            zaxis=dict(title='Height (in) - Floor → Ceiling', range=[0, trailer.height], dtick=24),
            aspectmode='data',
            camera=dict(eye=dict(x=1.3, y=1.3, z=0.8))
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False
    )

    return fig

def create_side_view(positions: List[Position], trailer: Trailer):
    """Create 2D side view (length vs height)"""
    fig = go.Figure()

    # Trailer outline
    fig.add_trace(go.Scatter(
        x=[0, trailer.length, trailer.length, 0, 0],
        y=[0, 0, trailer.height, trailer.height, 0],
        mode='lines',
        line=dict(color='black', width=2),
        fill='toself',
        fillcolor='rgba(200,200,200,0.1)',
        name='Trailer',
        hoverinfo='skip'
    ))

    stop_colors = {1: '#e74c3c', 2: '#f39c12', 3: '#f1c40f', 4: '#2ecc71', 5: '#3498db'}

    for pos in positions:
        p = pos.pallet
        color = stop_colors.get(p.delivery_stop, p.color)

        fig.add_trace(go.Scatter(
            x=[pos.x, pos.x + p.length, pos.x + p.length, pos.x, pos.x],
            y=[pos.z, pos.z, pos.z + p.height, pos.z + p.height, pos.z],
            mode='lines',
            line=dict(color='black', width=1),
            fill='toself',
            fillcolor=color,
            opacity=0.8,
            name=f"{p.id} (Stop {p.delivery_stop})",
            hovertemplate=f"Pallet {p.id}<br>Stop {p.delivery_stop}<br>{p.weight:.0f} lbs<br>{p.sku}"
        ))

        # Add ID label
        fig.add_trace(go.Scatter(
            x=[pos.x + p.length/2],
            y=[pos.z + p.height/2],
            mode='text',
            text=[p.id.split('-')[1]],
            textfont=dict(size=8, color='white'),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title="Side View: Length × Height (Nose → Tail)",
        xaxis_title="Length from Nose (inches)",
        yaxis_title="Height from Floor (inches)",
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, trailer.length]),
        yaxis=dict(range=[0, trailer.height])
    )

    return fig

def create_top_view(positions: List[Position], trailer: Trailer):
    """Create 2D top view (length vs width)"""
    fig = go.Figure()

    # Trailer outline
    fig.add_trace(go.Scatter(
        x=[0, trailer.length, trailer.length, 0, 0],
        y=[0, 0, trailer.width, trailer.width, 0],
        mode='lines',
        line=dict(color='black', width=2),
        fill='toself',
        fillcolor='rgba(200,200,200,0.1)',
        hoverinfo='skip'
    ))

    stop_colors = {1: '#e74c3c', 2: '#f39c12', 3: '#f1c40f', 4: '#2ecc71', 5: '#3498db'}

    for pos in positions:
        p = pos.pallet
        color = stop_colors.get(p.delivery_stop, p.color)

        fig.add_trace(go.Scatter(
            x=[pos.x, pos.x + p.length, pos.x + p.length, pos.x, pos.x],
            y=[pos.y, pos.y, pos.y + p.width, pos.y + p.width, pos.y],
            mode='lines',
            line=dict(color='black', width=1),
            fill='toself',
            fillcolor=color,
            opacity=0.8,
            hovertemplate=f"Pallet {p.id}<br>Stop {p.delivery_stop}<br>{p.weight:.0f} lbs"
        ))

        fig.add_trace(go.Scatter(
            x=[pos.x + p.length/2],
            y=[pos.y + p.width/2],
            mode='text',
            text=[p.id.split('-')[1]],
            textfont=dict(size=8, color='white'),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title="Top View: Floor Plan (Nose → Tail, Left → Right)",
        xaxis_title="Length from Nose (inches)",
        yaxis_title="Width from Left Wall (inches)",
        height=350,
        showlegend=False,
        xaxis=dict(range=[0, trailer.length]),
        yaxis=dict(range=[0, trailer.width])
    )

    return fig

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown('<div class="main-header">🚛 AI Load Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Trailer Optimization with 3D Visualization</div>', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Trailer Specs")
        trailer_length = st.selectbox("Trailer Length", ["53ft (636in)", "48ft (576in)", "45ft (540in)"], index=0)
        trailer_type = st.selectbox("Trailer Type", ["Dry Van", "Reefer", "Flatbed"], index=0)

        length_map = {"53ft (636in)": 636, "48ft (576in)": 576, "45ft (540in)": 540}

        st.subheader("Order Settings")
        num_orders = st.slider("Number of Pallets", 10, 50, 25)
        seed = st.number_input("Random Seed", 1, 1000, 42)

        st.subheader("Optimization")
        optimization_level = st.select_slider(
            "Optimization Level",
            options=["Fast", "Balanced", "Maximum"],
            value="Balanced"
        )

        st.markdown("---")
        st.markdown("**Delivery Stop Colors:**")
        st.markdown("🔴 Stop 1 (First Drop) | 🟠 Stop 2 | 🟡 Stop 3 | 🟢 Stop 4 | 🔵 Stop 5 (Last Drop)")

    # Initialize trailer
    trailer = Trailer(
        length=length_map[trailer_length],
        width=102,
        height=108,
        max_weight=45000,
        tare_weight=15000
    )

    # Generate data
    if 'pallets' not in st.session_state or st.session_state.get('seed') != seed or st.session_state.get('num_orders') != num_orders:
        st.session_state.pallets = generate_synthetic_orders(num_orders, seed)
        st.session_state.seed = seed
        st.session_state.num_orders = num_orders
        st.session_state.optimized = False

    pallets = st.session_state.pallets

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📋 Order Manifest")

        # Create DataFrame for display
        df_data = []
        for p in pallets:
            df_data.append({
                "Pallet ID": p.id,
                "SKU": p.sku,
                "Customer": p.customer,
                "Dimensions": f"{p.length:.0f}×{p.width:.0f}×{p.height:.0f}",
                "Weight": f"{p.weight:.0f} lbs",
                "Delivery Stop": p.delivery_stop,
                "Stackable": "✅" if p.stackable else "❌",
                "Fragile": "⚠️" if p.fragile else "✓",
                "Volume": f"{p.volume/1728:.1f} cu ft"
            })

        df = pd.DataFrame(df_data)

        # Color code by delivery stop
        def color_stop(val):
            colors = {1: '#ffcccc', 2: '#ffe6cc', 3: '#ffffcc', 4: '#ccffcc', 5: '#ccccff'}
            return f'background-color: {colors.get(val, "white")}'

        st.dataframe(
            df.style.applymap(color_stop, subset=['Delivery Stop']),
            use_container_width=True,
            height=400
        )

        # Summary stats
        total_weight = sum(p.weight for p in pallets)
        total_volume = sum(p.volume for p in pallets) / 1728  # cu ft

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Pallets", len(pallets))
        c2.metric("Total Weight", f"{total_weight:,.0f} lbs")
        c3.metric("Total Volume", f"{total_volume:.1f} cu ft")
        c4.metric("Unique Customers", len(set(p.customer for p in pallets)))

    with col2:
        st.markdown("### 🎯 Actions")

        if st.button("🚀 Optimize Load", use_container_width=True):
            with st.spinner("Running optimization algorithm..."):
                optimizer = LoadOptimizer(trailer)
                positions, metrics = optimizer.optimize(pallets)
                st.session_state.positions = positions
                st.session_state.metrics = metrics
                st.session_state.optimized = True
            st.success("✅ Optimization Complete!")

        if st.button("🔄 Regenerate Orders", use_container_width=True):
            st.session_state.pallets = generate_synthetic_orders(num_orders, random.randint(1, 1000))
            st.session_state.optimized = False
            st.rerun()

        st.markdown("---")
        st.markdown("#### 📊 Trailer Capacity")

        # Progress bars for capacity
        weight_pct = (total_weight / trailer.max_weight) * 100
        volume_pct = (total_volume / (trailer.volume / 1728)) * 100

        st.progress(min(weight_pct / 100, 1.0), text=f"Weight: {weight_pct:.1f}% of {trailer.max_weight:,.0f} lbs")
        st.progress(min(volume_pct / 100, 1.0), text=f"Volume: {volume_pct:.1f}% of {trailer.volume/1728:.0f} cu ft")

        st.markdown("---")
        st.markdown("#### 🎨 Legend")
        st.markdown("""
        <div style="font-size: 0.85rem;">
        <span style="color: #e74c3c;">■</span> Stop 1 (First Unload)<br>
        <span style="color: #f39c12;">■</span> Stop 2<br>
        <span style="color: #f1c40f;">■</span> Stop 3<br>
        <span style="color: #2ecc71;">■</span> Stop 4<br>
        <span style="color: #3498db;">■</span> Stop 5 (Last Unload)<br>
        <br>
        <b>Loading Rule:</b> Last delivery → Back of trailer<br>
        First delivery → Near nose
        </div>
        """, unsafe_allow_html=True)

    # Optimization Results
    if st.session_state.get('optimized', False):
        positions = st.session_state.positions
        metrics = st.session_state.metrics

        st.markdown("---")
        st.markdown("## 📈 Optimization Results")

        # Metrics cards
        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics['placed']}/{metrics['total_pallets']}</div>
                <div class="metric-label">Pallets Loaded</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                <div class="metric-value">{metrics['weight_util']:.1f}%</div>
                <div class="metric-label">Weight Utilization</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="metric-value">{metrics['volume_util']:.1f}%</div>
                <div class="metric-label">Volume Utilization</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                <div class="metric-value">{metrics['floor_util']:.1f}%</div>
                <div class="metric-label">Floor Utilization</div>
            </div>
            """, unsafe_allow_html=True)

        with m5:
            unplaced_count = metrics['unplaced']
            color = "#e74c3c" if unplaced_count > 0 else "#2ecc71"
            st.markdown(f"""
            <div class="metric-card" style="background: {color};">
                <div class="metric-value">{unplaced_count}</div>
                <div class="metric-label">Unplaced Pallets</div>
            </div>
            """, unsafe_allow_html=True)

        # 3D Visualization
        st.markdown("### 🎯 3D Interactive Load Plan")
        st.markdown("*Click and drag to rotate. Scroll to zoom. Hover over pallets for details.*")

        fig_3d = create_3d_visualization(positions, trailer)
        st.plotly_chart(fig_3d, use_container_width=True, height=700)

        # 2D Views
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig_side = create_side_view(positions, trailer)
            st.plotly_chart(fig_side, use_container_width=True)
        with col_v2:
            fig_top = create_top_view(positions, trailer)
            st.plotly_chart(fig_top, use_container_width=True)

        # Load Sheet / Dock Sheet
        st.markdown("### 📄 Load Sheet (Dock Instructions)")

        # Sort positions by loading order (back to front, bottom to top)
        load_order = sorted(positions, key=lambda p: (-p.x, p.z, p.y))

        sheet_data = []
        for i, pos in enumerate(load_order, 1):
            p = pos.pallet
            sheet_data.append({
                "Load Seq": i,
                "Pallet ID": p.id,
                "SKU": p.sku,
                "Customer": p.customer,
                "Stop": p.delivery_stop,
                "Position": f"({pos.x:.0f}, {pos.y:.0f}, {pos.z:.0f})",
                "Weight": f"{p.weight:.0f} lbs",
                "Stack": "Base" if pos.z < 0.1 else f"Level {pos.z//12:.0f}ft",
                "Notes": "⚠️ Fragile" if p.fragile else ("📦 Stackable" if p.stackable else "🔒 Non-stack")
            })

        df_sheet = pd.DataFrame(sheet_data)
        st.dataframe(df_sheet, use_container_width=True, height=300)

        # Download button for load sheet
        csv = df_sheet.to_csv(index=False)
        st.download_button(
            label="📥 Download Load Sheet (CSV)",
            data=csv,
            file_name="load_sheet.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Unplaced pallets warning
        if metrics['unplaced'] > 0:
            st.markdown("---")
            st.markdown("### ⚠️ Unplaced Pallets")
            st.markdown("<div class='warning-box'>The following pallets could not fit in the trailer. Consider a second trailer or adjusting the load.</div>", unsafe_allow_html=True)

            unplaced_df = pd.DataFrame([
                {"Pallet ID": p.id, "SKU": p.sku, "Customer": p.customer, 
                 "Weight": f"{p.weight:.0f} lbs", "Volume": f"{p.volume/1728:.1f} cu ft",
                 "Reason": "Overweight" if sum(pp.weight for pp in pallets) > trailer.max_weight else "Space constraint"}
                for p in metrics['unplaced_pallets']
            ])
            st.dataframe(unplaced_df, use_container_width=True)

    else:
        # Show placeholder before optimization
        st.markdown("---")
        st.info("👆 Click **'Optimize Load'** to see the AI-generated load plan with 3D visualization.")

        # Show empty trailer preview
        fig_empty = go.Figure()
        trailer_corners = [
            [0, 0, 0], [trailer.length, 0, 0], [trailer.length, trailer.width, 0], [0, trailer.width, 0],
            [0, 0, trailer.height], [trailer.length, 0, trailer.height], 
            [trailer.length, trailer.width, trailer.height], [0, trailer.width, trailer.height]
        ]
        edges = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]
        for edge in edges:
            fig_empty.add_trace(go.Scatter3d(
                x=[trailer_corners[edge[0]][0], trailer_corners[edge[1]][0]],
                y=[trailer_corners[edge[0]][1], trailer_corners[edge[1]][1]],
                z=[trailer_corners[edge[0]][2], trailer_corners[edge[1]][2]],
                mode='lines', line=dict(color='lightgray', width=2), hoverinfo='skip', showlegend=False
            ))
        fig_empty.update_layout(
            title="Empty Trailer - Waiting for Optimization",
            scene=dict(xaxis_title='Length', yaxis_title='Width', zaxis_title='Height', aspectmode='data'),
            height=500, showlegend=False
        )
        st.plotly_chart(fig_empty, use_container_width=True)

if __name__ == "__main__":
    main()
