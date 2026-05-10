"""
dashboard.py - AeroNet Lite Dashboard
Run: streamlit run src/dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
import time

from src.grid_model import create_sample_grid, AeroNetGrid, ZoneType, Drone, DroneType, Delivery
from src.layout_validator import LayoutValidator
from src.fleet_selector import FleetSelector, create_demand_grid
from src.astar_planner import AStarPlanner
from src.delivery_simulator import DeliverySimulator
from src.ml_pipeline import MLPipeline

st.set_page_config(
    page_title="AeroNet Lite",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

ZONE_COLORS = {
    "Residential": "#4CAF50",
    "Commercial":  "#2196F3",
    "Industrial":  "#9E9E9E",
    "Hospital":    "#F44336",
    "School":      "#FF9800",
    "Open Field":  "#E8F5E9",
}

GRID_SIZE = 10

def init_state():
    if "grid" not in st.session_state:
        random.seed(99)
        np.random.seed(42)
        st.session_state.grid = create_demand_grid()
        st.session_state.planner = AStarPlanner(st.session_state.grid)
        st.session_state.fleet = None
        st.session_state.drones = []
        st.session_state.deliveries = []
        st.session_state.ml_pipeline = None
        st.session_state.sim_step = 0
        st.session_state.event_log = []
        st.session_state.noflyCell = None
        st.session_state.disrupted = False
        st.session_state.sim_done = False
        st.session_state.current_path = None
        st.session_state.anim_step = 0
        st.session_state.anim_drones = []
        st.session_state.anim_deliveries = []
        st.session_state.anim_grid = None
        st.session_state.simulator = None

init_state()

def slog(msg: str):
    step = st.session_state.sim_step
    entry = f"Step {step:02d}: {msg}"
    st.session_state.event_log.append(entry)

def draw_hub(ax, cx, cy, size=0.30):
    body_w, body_h = size * 1.6, size * 1.2
    ax.add_patch(plt.Rectangle(
        (cx - body_w / 2, cy - body_h / 2), body_w, body_h,
        color="#FFD700", zorder=5, ec="#FFA500", lw=0.8
    ))
    for dx in (-body_w * 0.25, body_w * 0.25):
        ch_w, ch_h = body_w * 0.18, body_h * 0.55
        ax.add_patch(plt.Rectangle(
            (cx + dx - ch_w / 2, cy + body_h / 2), ch_w, ch_h,
            color="#FFA500", zorder=5, ec="#FF8C00", lw=0.6
        ))
    ax.text(cx, cy, "H", ha="center", va="center", fontsize=6,
            color="#0d1117", fontweight="bold", zorder=6)

def draw_nofly(ax, cx, cy, size=0.32):
    ax.add_patch(plt.Circle((cx, cy), size, color="#CC0000", zorder=5, ec="#FF0000", lw=1.0))
    ax.add_patch(plt.Circle((cx, cy), size, color="none", zorder=5, ec="#FF4444", lw=0.5))
    lw = size * 1.3
    ax.plot([cx - lw, cx + lw], [cy + lw, cy - lw], color="#FF4444", lw=1.8, zorder=6)
    ax.text(cx, cy, "X", ha="center", va="center", fontsize=7,
            color="white", fontweight="bold", zorder=7)

def draw_charging(ax, cx, cy, size=0.25):
    ax.add_patch(plt.Circle((cx, cy), size, color="#1a3a5c", zorder=4, ec="#00BFFF", lw=0.8))
    ax.text(cx, cy, "Z", ha="center", va="center", fontsize=7,
            color="#00BFFF", fontweight="bold", zorder=5)

def draw_medical(ax, cx, cy, size=0.25):
    ax.add_patch(plt.Circle((cx, cy), size, color="#500000", zorder=4, ec="#FF4444", lw=0.8))
    thick = size * 0.35
    ax.add_patch(plt.Rectangle(
        (cx - thick / 2, cy - size * 0.7), thick, size * 1.4,
        color="#FF4444", zorder=5
    ))
    ax.add_patch(plt.Rectangle(
        (cx - size * 0.7, cy - thick / 2), size * 1.4, thick,
        color="#FF4444", zorder=5
    ))

def draw_drone(ax, cx, cy, color, drone_id="", size=0.30, status="moving"):
    ax.add_patch(plt.Circle((cx, cy), size * 1.5, color=color, alpha=0.10, zorder=6))
    ax.add_patch(plt.Circle((cx, cy), size * 1.0, color=color, alpha=0.25, zorder=7))
    arm = size * 0.9
    for dx, dy in [(-arm, 0), (arm, 0), (0, -arm), (0, arm)]:
        ax.plot([cx, cx + dx], [cy, cy + dy], color=color, lw=1.2, alpha=0.7, zorder=7)
        ax.add_patch(plt.Circle((cx + dx, cy + dy), size * 0.22,
                                color=color, alpha=0.55, zorder=8))
    diamond_x = [cx, cx + size * 0.55, cx, cx - size * 0.55, cx]
    diamond_y = [cy + size * 0.55, cy, cy - size * 0.55, cy, cy + size * 0.55]
    ax.fill(diamond_x, diamond_y, color=color, alpha=0.85, zorder=9)
    ax.plot(diamond_x, diamond_y, color="white", lw=0.5, alpha=0.6, zorder=10)
    dot_color = "#00FF88" if status == "idle" else "#FFD700"
    ax.add_patch(plt.Circle((cx, cy), size * 0.22, color=dot_color, zorder=11))
    icon = "OK" if status == "idle" else "GO"
    ax.text(cx, cy + size * 1.75, f"{drone_id} {icon}", ha="center", va="bottom",
            fontsize=5, color=color, fontweight="bold", zorder=12)

def draw_pickup(ax, cx, cy, color="#FFD700"):
    s = 0.28
    ax.add_patch(plt.Rectangle((cx - s, cy - s * 0.8), s * 2, s * 1.6,
                                color="#8B4513", zorder=5, ec=color, lw=1.0))
    ax.add_patch(plt.Rectangle((cx - s, cy + s * 0.5), s * 2, s * 0.4,
                                color="#A0522D", zorder=5, ec=color, lw=0.6))
    ax.plot([cx, cx], [cy - s * 0.8, cy + s * 0.9], color=color, lw=1.0, zorder=6)
    ax.plot([cx - s, cx + s], [cy + s * 0.05, cy + s * 0.05], color=color, lw=1.0, zorder=6)
    ax.text(cx, cy - s * 1.1, "PKP", ha="center", va="top", fontsize=4.5,
            color=color, fontweight="bold", zorder=7)

def draw_dropoff(ax, cx, cy, color="#00BFFF"):
    s = 0.28
    ax.add_patch(plt.Rectangle((cx - s, cy - s), s * 2, s * 1.4,
                                color="#1a3a5c", zorder=5, ec=color, lw=1.0))
    roof_x = [cx - s * 1.15, cx, cx + s * 1.15]
    roof_y = [cy + s * 0.4, cy + s * 1.35, cy + s * 0.4]
    ax.fill(roof_x, roof_y, color="#0a2040", zorder=5)
    ax.plot(roof_x + [roof_x[0]], roof_y + [roof_y[0]], color=color, lw=0.8, zorder=6)
    ax.add_patch(plt.Rectangle((cx - s * 0.22, cy - s), s * 0.44, s * 0.7,
                                color=color, alpha=0.5, zorder=6))
    ax.text(cx, cy - s * 1.1, "DRP", ha="center", va="top", fontsize=4.5,
            color=color, fontweight="bold", zorder=7)

def draw_star(ax, cx, cy, color="#FFD700", size=0.30):
    import math
    n = 5
    outer, inner = size, size * 0.42
    xs, ys = [], []
    for i in range(n * 2):
        angle = math.pi / 2 + i * math.pi / n
        r = outer if i % 2 == 0 else inner
        xs.append(cx + r * math.cos(angle))
        ys.append(cy + r * math.sin(angle))
    xs.append(xs[0])
    ys.append(ys[0])
    ax.fill(xs, ys, color=color, zorder=10, alpha=0.9)
    ax.plot(xs, ys, color="white", lw=0.4, zorder=11)
    ax.text(cx, cy, "✓", ha="center", va="center", fontsize=6,
            color="#0d1117", fontweight="bold", zorder=12)

def draw_grid(grid, drones=None, title="City Grid"):
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = grid.get_cell(r, c)
            color = "#1a1a2e" if cell.no_fly else ZONE_COLORS.get(cell.zone.value, "#ffffff")
            ax.add_patch(plt.Rectangle((c, GRID_SIZE-1-r), 1, 1, color=color, ec="#0E1117", lw=0.8))
            cx, cy = c + 0.5, GRID_SIZE - 0.5 - r
            if cell.no_fly:
                draw_nofly(ax, cx, cy)
            elif cell.is_hub:
                draw_hub(ax, cx, cy)
            elif cell.is_charging:
                draw_charging(ax, cx, cy)
            elif cell.is_medical_pickup:
                draw_medical(ax, cx, cy)

    if drones:
        colors = plt.cm.tab10.colors
        drone_list = drones if isinstance(drones, list) else list(drones.values())
        for i, drone in enumerate(drone_list):
            if not drone.route:
                continue
            col = colors[i % len(colors)]
            xs = [c_ + 0.5 for (_, c_) in drone.route]
            ys = [GRID_SIZE - 0.5 - r_ for (r_, _) in drone.route]
            ax.plot(xs, ys, "-", color=col, lw=1.2, alpha=0.6)
            pr, pc = drone.current_location
            draw_drone(ax, pc + 0.5, GRID_SIZE - 0.5 - pr, col,
                       drone_id=drone.drone_id, status=drone.status)

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(range(GRID_SIZE+1))
    ax.set_yticks(range(GRID_SIZE+1))
    ax.set_xticklabels([str(i) if i < GRID_SIZE else "" for i in range(GRID_SIZE+1)], color="white", fontsize=7)
    ax.set_yticklabels([str(i) if i < GRID_SIZE else "" for i in range(GRID_SIZE+1)], color="white", fontsize=7)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
    ax.grid(True, color="#222", linewidth=0.4)
    return fig

def draw_demand_heatmap(grid):
    demand = np.array([[grid.get_cell(r,c).demand for c in range(GRID_SIZE)] for r in range(GRID_SIZE)])
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    im = ax.imshow(demand, cmap="YlOrRd", aspect="equal")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            ax.text(c, r, f"{demand[r,c]:.1f}", ha="center", va="center", fontsize=5, color="black")
    ax.set_title("Delivery Demand Heatmap", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    return fig

with st.sidebar:
    st.markdown("## 🚁 AeroNet Lite")
    st.markdown("**Autonomous Drone Delivery Simulator**")
    st.divider()

    st.markdown("### ⚙️ Simulation Controls")

    if st.button("🔄 Reset Simulation", use_container_width=True, key="reset_btn"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_state()
        st.rerun()

    st.divider()
    st.markdown("### 🗺️ Zone Legend")
    for zone, color in ZONE_COLORS.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
            f'<div style="width:16px;height:16px;background:{color};border-radius:3px"></div>'
            f'<span style="font-size:13px">{zone}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
                '<div style="width:16px;height:16px;background:#1a1a2e;border:1px solid #555;border-radius:3px;text-align:center;font-size:10px">X</div>'
                '<span style="font-size:13px">No-Fly Zone</span></div>',
                unsafe_allow_html=True)
    st.divider()
    st.caption("FAST-NUCES | BSDS AI SP2026")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🗺️ Grid & CSP", "🚁 Fleet (GA)", "📍 A* Routing", "🎬 Live Realtime", "⚠️ Disruption", "🧠 ML Pipeline", "📋 Event Log"
])

with tab1:
    st.markdown("## 📦 Module 1 — CSP Layout Validator")
    st.markdown("Validates the 10×10 city grid against 4 constraint rules.")

    col1, col2 = st.columns([1.4, 1])

    with col1:
        fig = draw_grid(st.session_state.grid, st.session_state.drones if st.session_state.drones else None, title="AeroNet Lite — City Grid")
        st.pyplot(fig, use_container_width=True)
        st.markdown("#### Delivery Demand Heatmap")
        fig_hm = draw_demand_heatmap(st.session_state.grid)
        st.pyplot(fig_hm, use_container_width=True)
        plt.close(fig_hm)

    with col2:
        st.markdown("### CSP Rules")
        validator = LayoutValidator(st.session_state.grid)
        validator.check_industrial_safety()
        validator.check_residential_coverage()
        validator.check_hub_charging_proximity()
        validator.check_medical_access()

        rules = [
            ("R1", "Industrial ↔ School/Hospital safety", "R1"),
            ("R2", "Residential ↔ Hub coverage (≤3 cells)", "R2"),
            ("R3", "Hub ↔ Charging pad (≤2 cells)", "R3"),
            ("R4", "Hospital ↔ Medical pickup (≤1 cell)", "R4"),
        ]

        for rule_id, desc, rid in rules:
            if rid in validator.passed_rules:
                st.success(f"✔ {rule_id}: {desc}")
            else:
                st.error(f"❌ {rule_id}: {desc}")

        passed = len(validator.passed_rules)
        st.divider()
        if passed == 4:
            st.success("### ✅ Layout VALID — All 4 rules passed")
            if not any("Layout validation" in e for e in st.session_state.event_log):
                st.session_state.sim_step = 1
                slog("Layout validation passed. All 4 CSP rules satisfied.")
        else:
            st.warning(f"### ⚠️ Layout has {4-passed} violation(s)")
            if not any("Layout validation" in e for e in st.session_state.event_log):
                st.session_state.sim_step = 1
                slog(f"Layout validation failed. {4-passed} rule(s) violated.")

        st.markdown("#### Grid Stats")
        zones = {}
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                z = st.session_state.grid.get_cell(r, c).zone.value
                zones[z] = zones.get(z, 0) + 1
        zone_df = pd.DataFrame(zones.items(), columns=["Zone", "Cells"])
        st.dataframe(zone_df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("## 🚁 Module 2 — Fleet Selector (Genetic Algorithm)")
    st.markdown("Optimises drone fleet composition under a fixed budget.")

    c1, c2, c3 = st.columns(3)
    with c1:
        budget_val = st.number_input("Budget ($)", value=10000, step=1000, min_value=5000, key="budget_input")
    with c2:
        pop_size = st.slider("GA Population", 10, 100, 60, key="pop_slider")
    with c3:
        gens = st.slider("Generations", 10, 200, 80, key="gen_slider")

    if st.button("▶ Run GA Fleet Selection", use_container_width=True, type="primary", key="run_ga_btn"):
        with st.spinner("Running Genetic Algorithm..."):
            selector = FleetSelector(st.session_state.grid, budget_val)
            light, heavy, fitness = selector.genetic_algorithm_select(verbose=False)
            st.session_state.fleet = {
                "light_drones": light,
                "heavy_drones": heavy,
                "total_cost": light * 1000 + heavy * 1800,
                "budget_remaining": budget_val - (light * 1000 + heavy * 1800),
                "coverage_pct": 100,
                "fitness_score": fitness,
            }
            st.session_state.drones = selector.create_fleet(light, heavy)
            st.session_state.sim_step = 3
            slog(f"Fleet selected: {light} Light + {heavy} Heavy drones")
        st.success("Fleet selected!")

    if st.session_state.fleet:
        f = st.session_state.fleet
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💡 Light Drones", f["light_drones"], f"${f['light_drones']*1000:,}")
        m2.metric("🏋️ Heavy Drones", f["heavy_drones"], f"${f['heavy_drones']*1800:,}")
        m3.metric("💰 Total Cost", f"${f['total_cost']:,}", f"${f['budget_remaining']:,} remaining")
        m4.metric("📦 Coverage", f"{f['coverage_pct']}%", f"Fitness: {f['fitness_score']:.4f}")

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        fig2.patch.set_facecolor("#0E1117")
        ax2.set_facecolor("#0E1117")
        bars = ax2.bar(
            ["Light Drones", "Heavy Drones"],
            [f["light_drones"] * 1000, f["heavy_drones"] * 1800],
            color=["#4CAF50", "#2196F3"],
            width=0.5,
        )
        ax2.axhline(budget_val, color="red", ls="--", lw=1.5, label=f"Budget ${budget_val:,}")
        ax2.set_ylabel("Cost ($)", color="white")
        ax2.tick_params(colors="white")
        ax2.set_title("Fleet Cost Breakdown", color="white", fontweight="bold")
        ax2.legend(facecolor="#1a1a2e", labelcolor="white")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#333")
        for bar in bars:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200,
                f"${bar.get_height():,.0f}",
                ha="center", color="white", fontsize=9,
            )
        st.pyplot(fig2)

        st.markdown("#### 🚁 Drone Fleet")
        drone_data = [
            {"ID": d.drone_id, "Type": d.drone_type.value, "Status": d.status, "Battery": f"{d.battery_level:.0f}%"}
            for d in st.session_state.drones
        ]
        st.dataframe(pd.DataFrame(drone_data), use_container_width=True, hide_index=True)
    else:
        st.info("Run GA fleet selection to see results.")

with tab3:
    st.markdown("## 📍 Module 3 — A* Delivery Path Planner")
    st.markdown("Plans Hub → Pickup → Drop-off → Hub routes using A* search.")

    if not st.session_state.drones:
        st.warning("Please run Fleet Selection first (Tab 2).")
    else:
        if st.button("▶ Generate & Assign Deliveries", use_container_width=True, type="primary", key="gen_deliveries_btn"):
            with st.spinner("Generating deliveries and planning A* routes..."):
                hubs = st.session_state.grid.get_hub_locations()
                residential = st.session_state.grid.get_cells_by_zone(ZoneType.RESIDENTIAL)
                commercial = st.session_state.grid.get_cells_by_zone(ZoneType.COMMERCIAL)
                medical = st.session_state.grid.get_medical_pickup_locations()

                st.session_state.deliveries = []
                dropoff_pool = residential if residential else (commercial if commercial else hubs)

                for i in range(8):
                    if medical and (i % 3 == 0):
                        pickup = random.choice(medical)
                        dropoff = random.choice(dropoff_pool)
                        weight = round(random.uniform(0.5, 1.5), 1)
                        priority = 1
                    else:
                        pickup = random.choice(hubs)
                        dropoff = random.choice(dropoff_pool)
                        weight = round(random.uniform(0.5, 1.0), 1)
                        priority = 2

                    delivery = Delivery(
                        delivery_id=i + 1,
                        pickup_location=pickup,
                        dropoff_location=dropoff,
                        weight_kg=weight,
                        priority=priority,
                        status="pending",
                    )
                    st.session_state.deliveries.append(delivery)

                for d in st.session_state.drones:
                    d.status = "idle"
                    d.route = []
                    d.route_index = 0
                    d.assigned_delivery = None

                pending = [d for d in st.session_state.deliveries if d.status == "pending"]
                available = [d for d in st.session_state.drones if d.status == "idle"]
                assigned = 0

                for i, delivery in enumerate(pending[: len(available)]):
                    drone = available[i]
                    if not drone.can_carry(delivery.weight_kg):
                        continue
                    hub = drone.current_location
                    route, cost, success = st.session_state.planner.plan_delivery_route(
                        hub, delivery.pickup_location, delivery.dropoff_location, verbose=False
                    )
                    if success:
                        drone.assigned_delivery = delivery
                        drone.route = route
                        drone.route_index = 0
                        drone.status = "moving"
                        delivery.assigned_drone_id = drone.drone_id
                        delivery.status = "assigned"
                        assigned += 1
                        st.session_state.sim_step = 5
                        slog(f"Delivery {delivery.delivery_id} assigned to Drone {drone.drone_id} | pickup {delivery.pickup_location} -> dropoff {delivery.dropoff_location}")

                slog(f"Deliveries generated: {assigned} assigned to drones")
                st.session_state.sim_step = 5
            st.success(f"{assigned} deliveries assigned!")

        if st.session_state.deliveries:
            col1, col2 = st.columns([1.4, 1])
            with col1:
                fig = draw_grid(st.session_state.grid, st.session_state.drones, title="Drone Routes — A* Paths")
                st.pyplot(fig, use_container_width=True)

            with col2:
                st.markdown("#### 📦 Delivery Assignments")
                total_d = len(st.session_state.deliveries)
                done_d  = sum(1 for d in st.session_state.deliveries if d.status == "completed")
                if done_d > 0:
                    st.success(f"✅ {done_d}/{total_d} deliveries completed")
                else:
                    st.info(f"📋 {total_d} deliveries assigned — run animation (Tab 4) to complete them")
                del_data = []
                for deliv in st.session_state.deliveries:
                    drone = next(
                        (d for d in st.session_state.drones if d.drone_id == deliv.assigned_drone_id), None
                    )
                    route_len = len(drone.route) if drone and drone.route else 0
                    del_data.append({
                        "ID": deliv.delivery_id,
                        "Pickup": str(deliv.pickup_location),
                        "Dropoff": str(deliv.dropoff_location),
                        "Weight": f"{deliv.weight_kg}kg",
                        "Drone": deliv.assigned_drone_id or "—",
                        "Route Len": route_len,
                        "Status": deliv.status,
                    })
                st.dataframe(pd.DataFrame(del_data), use_container_width=True, hide_index=True)

                st.markdown("#### 📊 Route Stats")
                routed = [d for d in st.session_state.drones if d.route]
                if routed:
                    lengths = [len(d.route) for d in routed]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg Route", f"{np.mean(lengths):.1f} cells")
                    c2.metric("Longest", f"{max(lengths)} cells")
                    c3.metric("Shortest", f"{min(lengths)} cells")

with tab4:
    st.markdown("## 🎬 Real-Time Live Animation")
    st.markdown("Watch drones move in real-time with automatic rerouting when no-fly zones are activated.")

    if not st.session_state.drones or not st.session_state.deliveries:
        st.warning("Please complete: Fleet Selection (Tab 2) → Generate Deliveries (Tab 3) first.")
    else:
        DRONE_COLORS = [
            "#FF5252", "#FF9800", "#FFEB3B", "#69F0AE", "#40C4FF",
            "#E040FB", "#FF6D00", "#00E5FF", "#76FF03", "#F50057",
        ]

        if "anim_initialized" not in st.session_state or not st.session_state.anim_initialized:
            st.session_state.anim_step = 0
            st.session_state.anim_completed = 0
            st.session_state.anim_drones = []
            for d in st.session_state.drones:
                nd = copy.deepcopy(d)
                nd.route_index = 0
                if nd.route:
                    nd.current_location = nd.route[0]
                nd.status = "moving" if nd.route else "idle"
                st.session_state.anim_drones.append(nd)
            st.session_state.anim_deliveries = copy.deepcopy(st.session_state.deliveries)
            st.session_state.anim_grid = copy.deepcopy(st.session_state.grid)
            st.session_state.anim_initialized = True

        def advance_one_step():
            completions_this_step = 0
            cells_per_step = 2

            for drone in st.session_state.anim_drones:
                if drone.status != "moving" or not drone.route:
                    continue

                steps_left = len(drone.route) - 1 - drone.route_index
                if steps_left > 0:
                    advance = min(cells_per_step, steps_left)
                    drone.route_index += advance
                    drone.current_location = drone.route[drone.route_index]
                    drone.battery_level = max(0, drone.battery_level - advance * 2)

                if drone.assigned_delivery:
                    dropoff = drone.assigned_delivery.dropoff_location
                    if tuple(drone.current_location) == tuple(dropoff):
                        for d in st.session_state.anim_deliveries:
                            if (d.delivery_id == drone.assigned_delivery.delivery_id and d.status != "completed"):
                                d.status = "completed"
                                drone.assigned_delivery.status = "completed"
                                completions_this_step += 1
                                for orig in st.session_state.deliveries:
                                    if orig.delivery_id == d.delivery_id:
                                        orig.status = "completed"
                                        break
                                break

                if drone.route_index >= len(drone.route) - 1:
                    drone.status = "idle"
                    drone.assigned_delivery = None

            st.session_state.anim_completed += completions_this_step
            st.session_state.anim_step += 1

            if st.session_state.anim_step >= 20:
                for d in st.session_state.anim_deliveries:
                    if d.status == "assigned":
                        d.status = "delayed"
                        for orig in st.session_state.deliveries:
                            if orig.delivery_id == d.delivery_id:
                                orig.status = "delayed"
                                break
                for drone in st.session_state.anim_drones:
                    if drone.assigned_delivery and not drone.route:
                        for d in st.session_state.anim_deliveries:
                            if d.delivery_id == drone.assigned_delivery.delivery_id and d.status == "assigned":
                                d.status = "failed"
                                for orig in st.session_state.deliveries:
                                    if orig.delivery_id == d.delivery_id:
                                        orig.status = "failed"
                                        break

            return completions_this_step

        def reroute_around_no_fly(row, col):
            rerouted = 0
            for drone in st.session_state.anim_drones:
                if drone.status != "moving" or not drone.route:
                    continue
                remaining = drone.route[drone.route_index:]
                if (row, col) not in remaining:
                    continue
                if not drone.assigned_delivery:
                    continue

                del_obj = next(
                    (d for d in st.session_state.anim_deliveries if d.delivery_id == drone.assigned_delivery.delivery_id), None
                )
                if del_obj is None:
                    continue

                target = del_obj.pickup_location if del_obj.status == "assigned" else del_obj.dropoff_location

                planner = AStarPlanner(st.session_state.anim_grid)
                path, cost, success = planner.find_alternative_path(
                    drone.current_location, target, blocked_cells={(row, col)}, verbose=False
                )
                if not success:
                    continue

                remaining_route = path[1:]

                if target == del_obj.pickup_location:
                    route2, _, ok2 = planner.find_path(del_obj.pickup_location, del_obj.dropoff_location)
                    hubs = st.session_state.anim_grid.get_hub_locations()
                    route3, _, ok3 = planner.find_path(del_obj.dropoff_location, hubs[0]) if hubs else ([], 0, False)
                    if ok2 and ok3:
                        drone.route = remaining_route + route2[1:] + route3[1:]
                    elif ok2:
                        drone.route = remaining_route + route2[1:]
                    else:
                        drone.route = remaining_route
                else:
                    hubs = st.session_state.anim_grid.get_hub_locations()
                    route3, _, ok3 = planner.find_path(del_obj.dropoff_location, hubs[0]) if hubs else ([], 0, False)
                    if ok3:
                        drone.route = remaining_route + route3[1:]
                    else:
                        drone.route = remaining_route

                drone.route_index = 0
                rerouted += 1
                slog(f"Drone {drone.drone_id} rerouted around ({row},{col})")
            return rerouted

        def draw_animated_grid():
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#0d1117")

            ZONE_BG = {
                "Residential": "#0a2010", "Commercial": "#0a1030",
                "Industrial": "#1a1a1a", "Hospital": "#200a0a",
                "School": "#1a1400", "Open Field": "#081408",
            }
            abbrev = {
                "Residential": "R", "Commercial": "C", "Industrial": "I",
                "Hospital": "H", "School": "S", "Open Field": ".",
            }

            completed_ids = {d.delivery_id for d in st.session_state.anim_deliveries if d.status == "completed"}
            completed_dropoffs = set(
                (d.dropoff_location[0], d.dropoff_location[1])
                for d in st.session_state.anim_deliveries if d.status == "completed"
            )

            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    cell = st.session_state.anim_grid.get_cell(r, c)
                    color = "#300000" if cell.no_fly else ZONE_BG.get(cell.zone.value, "#111")
                    ax.add_patch(plt.Rectangle((c, GRID_SIZE-1-r), 1, 1, color=color, ec="#1a1f2e", lw=0.6))
                    cx_c, cy_c = c + 0.5, GRID_SIZE - 0.5 - r
                    ax.text(cx_c, cy_c, abbrev.get(cell.zone.value, "?"),
                            ha="center", va="center", fontsize=6.5, color="#ffffff18")
                    if (r, c) in completed_dropoffs:
                        draw_star(ax, cx_c, cy_c - 0.30, size=0.22)
                    if cell.is_hub:
                        draw_hub(ax, cx_c, cy_c, size=0.28)
                    if cell.no_fly:
                        draw_nofly(ax, cx_c, cy_c, size=0.30)

            for i, drone in enumerate(st.session_state.anim_drones):
                if not drone.route:
                    continue
                col = DRONE_COLORS[i % len(DRONE_COLORS)]
                xs = [c_ + 0.5 for (_, c_) in drone.route]
                ys = [GRID_SIZE - 0.5 - r_ for (r_, _) in drone.route]
                ax.plot(xs, ys, "-", color=col, lw=0.9, alpha=0.2, zorder=3)
                if drone.route_index > 0:
                    ax.plot(xs[:drone.route_index+1], ys[:drone.route_index+1],
                            "-", color=col, lw=1.8, alpha=0.6, zorder=4)

            for i, drone in enumerate(st.session_state.anim_drones):
                col = DRONE_COLORS[i % len(DRONE_COLORS)]
                if drone.assigned_delivery:
                    del_id = drone.assigned_delivery.delivery_id
                    del_obj = next((d for d in st.session_state.anim_deliveries if d.delivery_id == del_id), None)
                    if del_obj and del_id not in completed_ids:
                        pr, pc = del_obj.pickup_location
                        draw_pickup(ax, pc + 0.5, GRID_SIZE - 0.5 - pr, color=col)
                        dr2, dc2 = del_obj.dropoff_location
                        draw_dropoff(ax, dc2 + 0.5, GRID_SIZE - 0.5 - dr2, color=col)

            for i, drone in enumerate(st.session_state.anim_drones):
                if drone.status == "failed":
                    continue
                col = DRONE_COLORS[i % len(DRONE_COLORS)]
                pr, pc = drone.current_location
                draw_drone(ax, pc + 0.5, GRID_SIZE - 0.5 - pr, col,
                           drone_id=drone.drone_id, status=drone.status)

            ax.set_xlim(0, GRID_SIZE)
            ax.set_ylim(0, GRID_SIZE)
            ax.set_xticks(range(GRID_SIZE + 1))
            ax.set_yticks(range(GRID_SIZE + 1))
            ax.set_xticklabels([str(i) if i < GRID_SIZE else "" for i in range(GRID_SIZE+1)], color="#666", fontsize=7)
            ax.set_yticklabels([str(i) if i < GRID_SIZE else "" for i in range(GRID_SIZE+1)], color="#666", fontsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1a1f2e")
            ax.grid(True, color="#1a1f2e", linewidth=0.5)

            actv = sum(1 for d in st.session_state.anim_drones if d.status == "moving")
            total_deliveries = len(st.session_state.anim_deliveries)
            done_n = len(completed_ids)
            ax.set_title(
                f"Real-Time Drone Animation  |  Step {st.session_state.anim_step}/20  |"
                f"  Active: {actv}  |  Completed: {done_n}/{total_deliveries}",
                color="white", fontsize=11, fontweight="bold", pad=10,
            )
            return fig

        status_ph = st.empty()
        grid_ph   = st.empty()

        left, right = st.columns([1, 2.5])

        with left:
            st.markdown("### Controls")
            speed = st.select_slider(
                "Animation Speed",
                options=["Slow", "Normal", "Fast", "Instant"],
                value="Normal", key="anim_speed"
            )
            delay_map = {"Slow": 1.2, "Normal": 0.6, "Fast": 0.35, "Instant": 0.3}

            st.divider()
            st.markdown("### No-Fly Zone")
            nf_r = st.number_input("No-fly Row", 0, 9, 4, key="anr")
            nf_c = st.number_input("No-fly Col", 0, 9, 6, key="anc")

            if st.button("ACTIVATE NO-FLY (Real-time Reroute)", use_container_width=True, key="activate_nofly_btn"):
                cell = st.session_state.anim_grid.get_cell(nf_r, nf_c)
                if cell:
                    cell.no_fly = True
                rerouted = reroute_around_no_fly(nf_r, nf_c)
                st.success(f"No-fly zone at ({nf_r},{nf_c}) — {rerouted} drone(s) rerouted")
                st.rerun()

            st.divider()

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Reset Animation", use_container_width=True, key="reset_anim_btn"):
                    st.session_state.anim_initialized = False
                    st.rerun()
            with col_b:
                if st.button("Play All 20", use_container_width=True, type="primary", key="play_all_btn"):
                    actual_delay = max(delay_map[speed], 0.3)
                    for step in range(st.session_state.anim_step + 1, 21):
                        completions = advance_one_step()

                        if step == 11:
                            cell = st.session_state.anim_grid.get_cell(nf_r, nf_c)
                            if cell:
                                cell.no_fly = True
                            reroute_around_no_fly(nf_r, nf_c)
                            st.session_state.sim_step = 11
                            slog(f"No-fly cell activated at ({nf_r},{nf_c})")
                            status_ph.warning(f"Step {step}: No-fly activated at ({nf_r},{nf_c}) — Rerouting drones!")
                        elif step == 18:
                            target_drone = next(
                                (d for d in st.session_state.anim_drones if d.status in ("moving", "idle")), None
                            )
                            if target_drone:
                                st.session_state.sim_step = 18
                                slog(f"Battery anomaly detected for Drone {target_drone.drone_id} — battery critically low")
                                status_ph.error(f"Step 18: Battery anomaly detected for Drone {target_drone.drone_id}!")
                            else:
                                status_ph.info(f"Step {step}/20 — drones en route...")
                        elif completions > 0:
                            status_ph.success(f"Step {step}: {completions} delivery(s) done! Total: {st.session_state.anim_completed}")
                        else:
                            status_ph.info(f"Step {step}/20 — drones en route...")

                        fig = draw_animated_grid()
                        grid_ph.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        time.sleep(actual_delay)

                    total_del = len(st.session_state.anim_deliveries)
                    n_completed = sum(1 for d in st.session_state.anim_deliveries if d.status == "completed")
                    n_delayed   = sum(1 for d in st.session_state.anim_deliveries if d.status == "delayed")
                    n_failed    = sum(1 for d in st.session_state.anim_deliveries if d.status == "failed")
                    st.session_state.sim_step = 20
                    slog(f"Simulation complete. {n_completed} completed, {n_delayed} delayed, {n_failed} failed.")
                    status_ph.success(f"Step 20 — Simulation complete! {n_completed} completed | {n_delayed} delayed | {n_failed} failed")
                    st.rerun()

            st.markdown("---")
            st.markdown("### Live Stats")
            total_del = len(st.session_state.anim_deliveries)
            completed_now = sum(1 for d in st.session_state.anim_deliveries if d.status == "completed")
            delayed_now   = sum(1 for d in st.session_state.anim_deliveries if d.status == "delayed")
            failed_now    = sum(1 for d in st.session_state.anim_deliveries if d.status == "failed")
            st.metric("Current Step", f"{st.session_state.anim_step}/20")
            st.metric("Completed", f"{completed_now}/{total_del}")
            st.metric("Active Drones", sum(1 for d in st.session_state.anim_drones if d.status == "moving"))
            if delayed_now > 0:
                st.warning(f"⏳ {delayed_now} delayed")
            if failed_now > 0:
                st.error(f"❌ {failed_now} failed")
            if st.session_state.anim_step >= 20:
                st.divider()
                st.markdown("### Step 20 — Final Summary")
                s1, s2, s3 = st.columns(3)
                s1.metric("Completed", completed_now)
                s2.metric("Delayed",   delayed_now)
                s3.metric("Failed",    failed_now)

            st.divider()
            st.markdown("**Legend**")
            st.markdown("**[Diamond]** Drone (active/moving)")
            st.markdown("**[Green dot]** Drone (idle/done)")
            st.markdown("**[PKP box]** Pickup location")
            st.markdown("**[DRP house]** Dropoff destination")
            st.markdown("**[Gold star]** Delivery completed")
            st.markdown("**[H rect]** Drone hub base")
            st.markdown("**[X circle]** No-fly zone")

        with right:
            fig = draw_animated_grid()
            grid_ph.pyplot(fig, use_container_width=True)
            plt.close(fig)

            col_c, col_d = st.columns(2)
            with col_c:
                if st.button("Advance 1 Step", use_container_width=True, key="advance_step_anim"):
                    completions = advance_one_step()
                    if completions > 0:
                        st.toast(f"{completions} delivery(s) completed!", icon="🎉")
                    st.rerun()
            with col_d:
                if st.button("Refresh View", use_container_width=True, key="refresh_view_anim"):
                    st.rerun()

            st.markdown("#### Drone Status")
            tbl = []
            for d in st.session_state.anim_drones:
                prog = int(d.route_index / max(len(d.route) - 1, 1) * 100) if d.route and len(d.route) > 1 else 100
                if d.status == "moving":
                    status_icon = "[GO]"
                    status_text = "MOVING"
                elif d.status == "idle":
                    status_icon = "[OK]"
                    status_text = "IDLE"
                else:
                    status_icon = "[!!]"
                    status_text = d.status.upper()

                delivery_status = "—"
                if d.assigned_delivery:
                    real_del = next((x for x in st.session_state.anim_deliveries if x.delivery_id == d.assigned_delivery.delivery_id), None)
                    if real_del and real_del.status == "completed":
                        delivery_status = "DONE"
                    else:
                        delivery_status = f"D{d.assigned_delivery.delivery_id}"
                else:
                    delivery_status = "DONE" if d.status == "idle" else "—"

                tbl.append({
                    "ID": d.drone_id,
                    "Type": d.drone_type.value,
                    "Status": f"{status_icon} {status_text}",
                    "Progress": f"{prog}%",
                    "Battery": f"{d.battery_level:.0f}%",
                    "Delivery": delivery_status,
                })
            st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

with tab5:
    st.markdown("## ⚠️ Module 4 — Disruption Handler")
    st.markdown("Activate a no-fly zone and watch drones reroute in real-time.")

    if not st.session_state.drones or not st.session_state.deliveries:
        st.warning("Please run Fleet Selection (Tab 2) and Generate Deliveries (Tab 3) first.")
    else:
        col_ctrl, col_map = st.columns([1, 1.5])

        with col_ctrl:
            st.markdown("### No-Fly Zone")
            nf_row = st.number_input("No-fly Row", 0, 9, 4, key="nf_row_d")
            nf_col = st.number_input("No-fly Col", 0, 9, 6, key="nf_col_d")

            if st.button("Activate No-Fly & Reroute", use_container_width=True, type="secondary", key="activate_nofly_disrupt"):
                with st.spinner("Activating no-fly zone and rerouting..."):
                    cell = st.session_state.grid.get_cell(nf_row, nf_col)
                    if cell:
                        cell.no_fly = True

                    for drone in st.session_state.drones:
                        if drone.status != "moving" or not drone.route:
                            continue
                        remaining = drone.route[drone.route_index:]
                        if (nf_row, nf_col) not in remaining:
                            continue
                        if not drone.assigned_delivery:
                            continue
                        target = drone.assigned_delivery.pickup_location if drone.assigned_delivery.status == "assigned" else drone.assigned_delivery.dropoff_location
                        path, cost, success = st.session_state.planner.find_alternative_path(
                            drone.current_location, target, blocked_cells={(nf_row, nf_col)}, verbose=False,
                        )
                        if success:
                            remaining_route = path[1:]
                            if target == drone.assigned_delivery.pickup_location:
                                route2, _, ok2 = st.session_state.planner.find_path(
                                    drone.assigned_delivery.pickup_location, drone.assigned_delivery.dropoff_location,
                                )
                                hubs = st.session_state.grid.get_hub_locations()
                                route3, _, ok3 = st.session_state.planner.find_path(drone.assigned_delivery.dropoff_location, hubs[0]) if hubs else ([], 0, False)
                                if ok2 and ok3:
                                    drone.route = remaining_route + route2[1:] + route3[1:]
                                elif ok2:
                                    drone.route = remaining_route + route2[1:]
                                else:
                                    drone.route = remaining_route
                            else:
                                hubs = st.session_state.grid.get_hub_locations()
                                route3, _, ok3 = st.session_state.planner.find_path(drone.assigned_delivery.dropoff_location, hubs[0]) if hubs else ([], 0, False)
                                if ok3:
                                    drone.route = remaining_route + route3[1:]
                                else:
                                    drone.route = remaining_route
                            drone.route_index = 0
                            slog(f"Drone {drone.drone_id} rerouted around ({nf_row},{nf_col})")
                    slog(f"No-fly zone activated at ({nf_row},{nf_col})")
                st.success(f"No-fly zone set at ({nf_row},{nf_col})")
                st.rerun()

            st.markdown("---")
            st.markdown("### Step Control")

            if st.button("Advance 1 Step", use_container_width=True, key="advance_step_disrupt"):
                st.session_state.sim_step += 1
                completed_count = 0
                for drone in st.session_state.drones:
                    if drone.status == "moving" and drone.route_index < len(drone.route) - 1:
                        drone.route_index += 1
                        drone.current_location = drone.route[drone.route_index]
                        drone.battery_level -= 2
                    elif drone.status == "moving":
                        if drone.assigned_delivery:
                            drone.assigned_delivery.status = "completed"
                            completed_count += 1
                            slog(f"Drone {drone.drone_id} completed Delivery {drone.assigned_delivery.delivery_id}")
                        drone.status = "idle"
                        drone.assigned_delivery = None
                if completed_count > 0:
                    st.success(f"{completed_count} delivery(s) completed!")
                st.rerun()

            st.markdown("---")
            completed = sum(1 for d in st.session_state.deliveries if d.status == "completed")
            active = sum(1 for d in st.session_state.drones if d.status == "moving")
            st.markdown(
                f"""
                <div style="display:flex;gap:1rem;margin-top:1rem">
                  <div style="flex:1;background:#1a1a2e;padding:0.8rem;border-radius:8px;text-align:center">
                    <div style="color:#4ade80;font-size:1.8rem;font-weight:bold">{completed}</div>
                    <div style="color:#888;font-size:0.7rem">Completed</div>
                  </div>
                  <div style="flex:1;background:#1a1a2e;padding:0.8rem;border-radius:8px;text-align:center">
                    <div style="color:#22d3ee;font-size:1.8rem;font-weight:bold">{active}</div>
                    <div style="color:#888;font-size:0.7rem">Active Drones</div>
                  </div>
                  <div style="flex:1;background:#1a1a2e;padding:0.8rem;border-radius:8px;text-align:center">
                    <div style="color:#fbbf24;font-size:1.8rem;font-weight:bold">{st.session_state.sim_step}</div>
                    <div style="color:#888;font-size:0.7rem">Step</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_map:
            fig = draw_grid(st.session_state.grid, st.session_state.drones, title=f"Live Grid — Step {st.session_state.sim_step}")
            st.pyplot(fig, use_container_width=True)

with tab6:
    st.markdown("## 🧠 Module 5 — ML Pipeline")

    sub1, sub2 = st.tabs(["📈 Demand Forecasting", "🔴 Anomaly Detection"])

    with sub1:
        st.markdown("### Demand Forecasting")
        st.markdown("Trained on Kaggle Bike Sharing Dataset")

        if st.button("▶ Load Demand Model", use_container_width=True, type="primary", key="load_demand_btn"):
            with st.spinner("Loading trained model..."):
                st.session_state.ml_pipeline = MLPipeline()
                st.session_state.ml_pipeline.load_models(verbose=False)
                st.success("Demand model loaded!")

        if st.session_state.ml_pipeline and st.session_state.ml_pipeline.demand_model:
            hour = st.slider("Hour of Day", 0, 23, 12, key="demand_hour")
            if st.button("Predict Demand", use_container_width=True, key="predict_demand_btn"):
                demand = st.session_state.ml_pipeline.predict_demand(hour)
                st.metric("Predicted Demand", f"{demand:.0f} units")

                hours = list(range(24))
                demands = [st.session_state.ml_pipeline.predict_demand(h) for h in hours]
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_facecolor("#0E1117")
                ax.set_facecolor("#0E1117")
                ax.plot(hours, demands, color="#00d4ff", linewidth=2)
                ax.fill_between(hours, demands, alpha=0.1, color="#00d4ff")
                ax.set_xlabel("Hour", color="white")
                ax.set_ylabel("Demand", color="white")
                ax.set_title("Demand Forecast by Hour", color="white")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")
                st.pyplot(fig)
        else:
            st.info("Click 'Load Demand Model' to see results.")

    with sub2:
        st.markdown("### Anomaly Detection")
        st.markdown("Trained on ALFA UAV Telemetry Dataset")

        if st.button("▶ Load Anomaly Model", use_container_width=True, type="primary", key="load_anomaly_btn"):
            with st.spinner("Loading trained model..."):
                if st.session_state.ml_pipeline is None:
                    st.session_state.ml_pipeline = MLPipeline()
                st.session_state.ml_pipeline.load_models(verbose=False)
                st.success("Anomaly model loaded!")

        if st.session_state.ml_pipeline and st.session_state.ml_pipeline.anomaly_model:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                test_battery = st.slider("Battery Level (%)", 0, 100, 75, key="anomaly_battery")
            with col_b:
                test_speed = st.slider("Speed (m/s)", 0, 50, 12, key="anomaly_speed")
            with col_c:
                test_deviation = st.slider("Route Deviation", 0.0, 10.0, 0.5, key="anomaly_deviation")

            if st.button("Detect Anomaly", use_container_width=True, type="primary", key="detect_anomaly_btn"):
                anomaly_type, confidence, recommendation = st.session_state.ml_pipeline.analyze_drone_telemetry(
                    drone_id="TEST", battery_level=test_battery, current_speed=test_speed,
                    altitude_change=0, route_deviation=test_deviation
                )
                if anomaly_type == "failure":
                    st.error(f"⚠️ {anomaly_type.upper()} detected!")
                    slog(f"Battery anomaly detected for Drone TEST — confidence {confidence:.1%}")
                elif anomaly_type == "traj":
                    st.warning(f"⚠️ {anomaly_type.upper()} detected!")
                    slog(f"Route anomaly detected for Drone TEST — confidence {confidence:.1%}")
                else:
                    st.success(f"✅ {anomaly_type}")
                st.metric("Confidence", f"{confidence:.1%}")
                st.info(f"Recommendation: {recommendation}")
        else:
            st.info("Click 'Load Anomaly Model' to see results.")

with tab7:
    st.markdown("## 📋 Simulation Event Log")

    all_events = st.session_state.event_log
    if not all_events:
        st.info("No events yet. Run the simulation to generate events.")
    else:
        st.markdown(f"**{len(all_events)} events recorded**")

        search = st.text_input("🔍 Filter events", placeholder="e.g. rerouted, anomaly, completed...", key="event_search")
        filtered = [e for e in all_events if search.lower() in e.lower()] if search else all_events

        log_html = (
            '<div style="background:#0d1117;padding:12px;border-radius:8px;'
            'font-family:monospace;font-size:13px;max-height:500px;overflow-y:auto">'
        )
        for entry in reversed(filtered):
            color = "#ffffff"
            if "completed" in entry.lower():
                color = "#4CAF50"
            elif "rerouted" in entry.lower():
                color = "#FF9800"
            elif "anomaly" in entry.lower() or "failed" in entry.lower():
                color = "#F44336"
            elif "no-fly" in entry.lower():
                color = "#F44336"
            log_html += f'<div style="color:{color};margin:2px 0;padding:2px 0;border-bottom:1px solid #1a1a2a">{entry}</div>'
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)

        if st.button("Download Event Log", key="download_log_btn"):
            log_text = "\n".join(all_events)
            st.download_button("Download .txt", log_text, "aeronet_event_log.txt", "text/plain", key="download_log_file")

st.divider()
st.markdown(
    '<div style="text-align:center;color:#555;font-size:12px">'
    'AeroNet Lite • Real-Time Drone Delivery Simulation • BSDS AI Semester Project SP2026 • FAST-NUCES Rawalpindi'
    '</div>',
    unsafe_allow_html=True,
)