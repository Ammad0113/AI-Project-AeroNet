
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import numpy as np
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import math

from grid_model import AeroNetGrid, ZoneType, Drone, Delivery
from fleet_selector import FleetSelector, create_demand_grid
from delivery_simulator import DeliverySimulator


BG = "#0E1117"
PANEL = "#1a1a2e"
SURFACE = "#1E1E2E"
BORDER = "#333333"
GRID_LINE = "#2a2a3a"

TEXT_HI = "#E0E4F0"
TEXT_MID = "#9498AA"
TEXT_DIM = "#5C6075"

TEAL = "#4ECDC4"
GOLD = "#FFD700"
SAGE = "#5BAD8F"
BLUSH = "#D96B7A"
LAVENDER = "#8B82C4"
SLATE_BLUE = "#5B8DB8"

ZONE_COLORS_DICT = {
    "Residential": "#4CAF50",
    "Commercial": "#2196F3",
    "Industrial": "#9E9E9E",
    "Hospital": "#F44336",
    "School": "#FF9800",
    "Open Field": "#E8F5E9",
}

ZONE_ICONS = {
    ZoneType.RESIDENTIAL: "H",
    ZoneType.COMMERCIAL: "C",
    ZoneType.INDUSTRIAL: "I",
    ZoneType.HOSPITAL: "M",
    ZoneType.SCHOOL: "S",
    ZoneType.OPEN_FIELD: "O",
}

_AERO_HEATMAP = LinearSegmentedColormap.from_list(
    "aero_heat",
    ["#0E1117", "#1E3D48", "#1B5468", "#2A7A8C", "#4ECDC4", "#A8E6E3"],
)
_AERO_ANOMALY = LinearSegmentedColormap.from_list(
    "aero_anomaly",
    ["#0E1117", "#1F2D28", "#2A5C45", "#5BAD8F", "#E6A817", "#C4614D", "#D96B7A"],
)


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
    ax.text(cx, cy, "+", ha="center", va="center", fontsize=7,
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


class EnhancedGridVisualizer:

    def __init__(self, grid: AeroNetGrid, figsize=(14, 10)):
        self.grid = grid
        self.figsize = figsize
        self.setup_style()

    def setup_style(self):
        plt.rcParams.update({
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": TEXT_MID,
            "axes.titlecolor": TEXT_HI,
            "text.color": TEXT_HI,
            "xtick.color": TEXT_DIM,
            "ytick.color": TEXT_DIM,
            "grid.color": GRID_LINE,
            "grid.linewidth": 0.4,
            "legend.facecolor": SURFACE,
            "legend.edgecolor": BORDER,
            "legend.labelcolor": TEXT_MID,
            "font.family": "sans-serif",
            "figure.dpi": 120,
        })

    def plot_master_dashboard(self, drones: Dict[str, Drone] = None,
                              deliveries: List[Delivery] = None,
                              title: str = "AeroNet Lite — Command Center"):

        fig = plt.figure(figsize=(18, 13))
        fig.patch.set_facecolor(BG)

        hax = fig.add_axes([0, 0.955, 1, 0.045])
        hax.set_facecolor(SURFACE)
        hax.set_xlim(0, 1)
        hax.set_ylim(0, 1)
        hax.axis("off")
        hax.text(0.014, 0.46, title, color=TEXT_HI, fontsize=12, va="center")
        ts = datetime.now().strftime("%Y-%m-%d   %H:%M")
        hax.text(0.987, 0.46, ts, color=TEXT_DIM, fontsize=8, va="center", ha="right")

        gs = gridspec.GridSpec(3, 4, figure=fig,
                               width_ratios=[2.8, 2.8, 0.85, 1.7],
                               hspace=0.36, wspace=0.28,
                               left=0.03, right=0.97,
                               top=0.94, bottom=0.07)

        ax1 = fig.add_subplot(gs[0:2, 0:2])
        self.plot_zone_map(ax=ax1, show_legend=False)
        ax1.set_title("Grid Zone Map", loc="left", fontsize=9, color=TEXT_MID, pad=8)

        ax_leg = fig.add_subplot(gs[0:2, 2])
        ax_leg.set_facecolor(BG)
        ax_leg.axis("off")
        self._draw_legend(ax_leg)
        ax_leg.set_title("Legend", loc="left", fontsize=9, color=TEXT_MID, pad=8)

        ax2 = fig.add_subplot(gs[0, 3])
        self.plot_drone_positions(drones or {}, ax=ax2)
        ax2.set_title("Active Drones", loc="left", fontsize=9, color=TEXT_MID, pad=8)

        ax3 = fig.add_subplot(gs[1, 3])
        self.plot_demand_heatmap(ax=ax3)
        ax3.set_title("Demand Heatmap", loc="left", fontsize=9, color=TEXT_MID, pad=8)

        ax4 = fig.add_subplot(gs[2, :])
        ax4.set_facecolor(SURFACE)
        for spine in ax4.spines.values():
            spine.set_edgecolor(BORDER)
            spine.set_linewidth(0.6)
        ax4.axis("off")
        self._add_statistics_panel(ax4, drones, deliveries)

        return fig

    def plot_zone_map(self, ax=None, show_legend=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            fig.patch.set_facecolor(BG)

        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
            spine.set_linewidth(0.6)

        for row in range(self.grid.size):
            for col in range(self.grid.size):
                cell = self.grid.get_cell(row, col)
                color = ZONE_COLORS_DICT.get(cell.zone.value, "#ffffff")
                y = self.grid.size - 1 - row

                rect = patches.FancyBboxPatch(
                    (col + 0.06, y + 0.06), 0.88, 0.88,
                    boxstyle="round,pad=0.06",
                    facecolor=color, edgecolor=BORDER,
                    linewidth=0.9, zorder=1,
                )
                ax.add_patch(rect)

                icon = ZONE_ICONS.get(cell.zone, "?")
                ax.text(col + 0.5, y + 0.5, icon,
                        ha="center", va="center", fontsize=10,
                        color="white", zorder=2, fontweight="bold")

        self._mark_enhanced_locations(ax)

        ax.set_xlim(0, self.grid.size)
        ax.set_ylim(0, self.grid.size)
        ax.set_xticks(range(self.grid.size + 1))
        ax.set_yticks(range(self.grid.size + 1))
        labels = [str(i) for i in range(self.grid.size + 1)]
        ax.set_xticklabels(labels, fontsize=7, color=TEXT_DIM)
        ax.set_yticklabels(labels[::-1], fontsize=7, color=TEXT_DIM)
        ax.grid(True, color=GRID_LINE, linewidth=0.4, linestyle="-", alpha=1, zorder=0)
        ax.set_aspect("equal")

        return ax

    def plot_demand_heatmap(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            fig.patch.set_facecolor(BG)

        demand_matrix = np.zeros((self.grid.size, self.grid.size))
        for row in range(self.grid.size):
            for col in range(self.grid.size):
                cell = self.grid.get_cell(row, col)
                demand_matrix[row, col] = cell.demand if cell else 0

        im = ax.imshow(demand_matrix, cmap="YlOrRd", interpolation="bicubic", origin="upper", aspect="equal")
        ax.set_xticks(range(self.grid.size))
        ax.set_yticks(range(self.grid.size))
        ax.set_xticklabels([str(i) for i in range(self.grid.size)], fontsize=7, color=TEXT_DIM)
        ax.set_yticklabels([str(i) for i in range(self.grid.size)], fontsize=7, color=TEXT_DIM)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=TEXT_DIM, labelsize=7)
        cbar.set_label("Demand Units", color=TEXT_MID, fontsize=8)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_MID)
        cbar.outline.set_edgecolor(BORDER)

        return ax

    def plot_demand_heatmap_3d(self):
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(15, 10))
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(BG)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(BORDER)
        ax.yaxis.pane.set_edgecolor(BORDER)
        ax.zaxis.pane.set_edgecolor(BORDER)
        ax.grid(True, color=GRID_LINE, linewidth=0.3, alpha=0.8)

        X, Y = np.meshgrid(range(self.grid.size), range(self.grid.size))
        Z = np.zeros((self.grid.size, self.grid.size))
        for row in range(self.grid.size):
            for col in range(self.grid.size):
                cell = self.grid.get_cell(row, col)
                Z[row, col] = cell.demand if cell else 0

        surf = ax.plot_surface(X, Y, Z, cmap="YlOrRd", linewidth=0, antialiased=True, alpha=0.85)
        ax.contour(X, Y, Z, zdir="z", offset=0, cmap="YlOrRd", alpha=0.4)

        for lbl in (ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels() + ax.zaxis.get_ticklabels()):
            lbl.set_color(TEXT_MID)
            lbl.set_fontsize(7)

        ax.set_xlabel("Column", fontsize=9, color=TEXT_MID, labelpad=8)
        ax.set_ylabel("Row", fontsize=9, color=TEXT_MID, labelpad=8)
        ax.set_zlabel("Demand", fontsize=9, color=TEXT_MID, labelpad=8)
        ax.set_title("3D Demand Distribution", fontsize=14, color=TEXT_HI, pad=18)

        cbar = fig.colorbar(surf, ax=ax, shrink=0.45, aspect=12, pad=0.08)
        cbar.set_label("Demand Units", color=TEXT_MID, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=TEXT_DIM, labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_MID)
        cbar.outline.set_edgecolor(BORDER)

        return fig

    def plot_drone_positions(self, drones: Dict[str, Drone], ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            fig.patch.set_facecolor(BG)

        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
            spine.set_linewidth(0.6)

        if not drones:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.58, "○", ha="center", va="center", fontsize=28, color=BORDER, transform=ax.transAxes)
            ax.text(0.5, 0.38, "No active drones", ha="center", va="center", fontsize=8, color=TEXT_DIM, transform=ax.transAxes)
            return ax

        for row in range(self.grid.size):
            for col in range(self.grid.size):
                cell = self.grid.get_cell(row, col)
                color = ZONE_COLORS_DICT.get(cell.zone.value, "#ffffff")
                y = self.grid.size - 1 - row
                rect = patches.FancyBboxPatch(
                    (col + 0.05, y + 0.05), 0.90, 0.90,
                    boxstyle="round,pad=0.05",
                    facecolor=color, edgecolor=BORDER,
                    linewidth=0.7, zorder=1,
                )
                ax.add_patch(rect)

        colors = plt.cm.tab10.colors
        for i, (drone_id, drone) in enumerate(drones.items()):
            col = drone.current_location[1]
            row = self.grid.size - 1 - drone.current_location[0]
            drone_color = colors[i % len(colors)]
            draw_drone(ax, col + 0.5, row + 0.5, drone_color, drone_id=drone.drone_id, status=drone.status)

        ax.set_xlim(0, self.grid.size)
        ax.set_ylim(0, self.grid.size)
        ax.set_xticks(range(self.grid.size + 1))
        ax.set_yticks(range(self.grid.size + 1))
        labels = [str(i) for i in range(self.grid.size + 1)]
        ax.set_xticklabels(labels, fontsize=7, color=TEXT_DIM)
        ax.set_yticklabels(labels[::-1], fontsize=7, color=TEXT_DIM)
        ax.grid(True, color=GRID_LINE, linewidth=0.4, linestyle="-", alpha=1)
        ax.set_aspect("equal")

        return ax

    def plot_performance_dashboard(self, simulation_stats: Dict):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.patch.set_facecolor(BG)
        for ax in axes.flatten():
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=TEXT_DIM)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            ax.title.set_color(TEXT_HI)
            ax.xaxis.label.set_color(TEXT_MID)
            ax.yaxis.label.set_color(TEXT_MID)
            ax.grid(True, color=GRID_LINE, alpha=0.3)

        fig.suptitle("Performance Analytics", fontsize=13, color=TEXT_HI, y=0.98)

        success_data = [
            simulation_stats.get("completed", 0),
            simulation_stats.get("failed", 0),
            simulation_stats.get("delayed", 0),
        ]
        axes[0, 0].pie(success_data, labels=["Completed", "Failed", "Delayed"],
                       autopct="%1.1f%%", colors=[SAGE, BLUSH, GOLD],
                       wedgeprops={"linewidth": 1.2, "edgecolor": BG},
                       startangle=90, textprops={"color": TEXT_MID, "fontsize": 8})
        axes[0, 0].set_title("Delivery Status", pad=12)

        batteries = simulation_stats.get("battery_levels", [])
        if batteries:
            axes[0, 1].hist(batteries, bins=20, color=TEAL, edgecolor=BG, linewidth=0.5, alpha=0.80)
            mean_val = np.mean(batteries)
            axes[0, 1].axvline(mean_val, color=GOLD, linewidth=1.2, linestyle="--", label=f"Mean: {mean_val:.1f}%")
            axes[0, 1].set_xlabel("Battery Level (%)", fontsize=8)
            axes[0, 1].set_ylabel("Frequency", fontsize=8)
            axes[0, 1].set_title("Drone Battery Distribution", pad=12)
            axes[0, 1].legend(fontsize=8)

        days = list(range(1, len(simulation_stats.get("demand_history", [])) + 1))
        if days:
            demand_h = simulation_stats.get("demand_history", [])
            capacity_h = simulation_stats.get("capacity_history", [])
            axes[1, 0].plot(days, demand_h, color=BLUSH, linewidth=1.6, marker="o", markersize=3, label="Demand")
            axes[1, 0].fill_between(days, demand_h, alpha=0.10, color=BLUSH)
            axes[1, 0].plot(days, capacity_h, color=TEAL, linewidth=1.6, marker="s", markersize=3, label="Capacity")
            axes[1, 0].fill_between(days, capacity_h, alpha=0.08, color=TEAL)
            axes[1, 0].set_xlabel("Time Step", fontsize=8)
            axes[1, 0].set_ylabel("Units", fontsize=8)
            axes[1, 0].set_title("Demand vs Delivery Capacity", pad=12)
            axes[1, 0].legend(fontsize=8)

        anomaly_data = simulation_stats.get("anomaly_matrix", np.zeros((10, 10)))
        im = axes[1, 1].imshow(anomaly_data, cmap=_AERO_ANOMALY, interpolation="bicubic")
        axes[1, 1].set_title("Anomaly Heatmap", pad=12)
        cbar = plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        cbar.set_label("Anomaly Score", color=TEXT_MID, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=TEXT_DIM, labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_MID)
        cbar.outline.set_edgecolor(BORDER)

        plt.tight_layout()
        return fig

    def _mark_enhanced_locations(self, ax):
        for row, col in self.grid.get_hub_locations():
            y = self.grid.size - 0.5 - row
            draw_hub(ax, col + 0.5, y, size=0.28)

        for row, col in self.grid.get_charging_locations():
            y = self.grid.size - 0.5 - row
            draw_charging(ax, col + 0.5, y, size=0.25)

        for row, col in self.grid.get_medical_pickup_locations():
            y = self.grid.size - 0.5 - row
            draw_medical(ax, col + 0.5, y, size=0.25)

    def _draw_legend(self, ax):
        zone_entries = [
            ("Residential", "#4CAF50"),
            ("Commercial", "#2196F3"),
            ("Hospital", "#F44336"),
            ("School", "#FF9800"),
            ("Industrial", "#9E9E9E"),
            ("Open Field", "#E8F5E9"),
        ]
        special_entries = [
            ("⬟", "#FFD700", "Hub"),
            ("+", "#00BFFF", "Charging"),
            ("†", "#FF4444", "Medical"),
            ("X", "#CC0000", "No-fly"),
        ]

        x0, y0 = 0.05, 0.92
        row_h = 0.09

        for i, (label, color) in enumerate(zone_entries):
            y = y0 - i * row_h
            sw = patches.FancyBboxPatch((x0, y - 0.03), 0.12, 0.06,
                                        boxstyle="round,pad=0.008",
                                        facecolor=color, edgecolor="none",
                                        transform=ax.transAxes, zorder=3)
            ax.add_patch(sw)
            ax.text(x0 + 0.18, y, label, ha="left", va="center", fontsize=7.5,
                    color=TEXT_MID, transform=ax.transAxes)

        divider_y = y0 - len(zone_entries) * row_h + 0.03
        ax.plot([0.05, 0.95], [divider_y, divider_y], color=BORDER, linewidth=0.6, transform=ax.transAxes)

        base_y = divider_y - 0.08
        for j, (marker, color, label) in enumerate(special_entries):
            y = base_y - j * row_h
            sw = patches.FancyBboxPatch((x0, y - 0.03), 0.12, 0.06,
                                        boxstyle="round,pad=0.008",
                                        facecolor="#1a1a2e", edgecolor=BORDER,
                                        transform=ax.transAxes, zorder=3)
            ax.add_patch(sw)
            ax.text(x0 + 0.06, y, marker, ha="center", va="center", fontsize=7,
                    color=color, transform=ax.transAxes)
            ax.text(x0 + 0.18, y, label, ha="left", va="center", fontsize=7.5,
                    color=TEXT_MID, transform=ax.transAxes)

    def _add_statistics_panel(self, ax, drones, deliveries):
        drone_list = list((drones or {}).values())
        delivery_list = deliveries or []

        total_drones = len(drone_list)
        active_drones = sum(1 for d in drone_list if d.status == "moving")
        idle_drones = sum(1 for d in drone_list if d.status == "idle")
        avg_battery = np.mean([d.battery_level for d in drone_list]) if drone_list else 0.0
        total_del = len(delivery_list)
        completed = sum(1 for d in delivery_list if d.status == "completed")
        in_progress = sum(1 for d in delivery_list if d.status == "assigned")
        pending = sum(1 for d in delivery_list if d.status == "pending")
        hubs = len(self.grid.get_hub_locations())
        chargers = len(self.grid.get_charging_locations())
        med_pkp = len(self.grid.get_medical_pickup_locations())
        no_fly = sum(1 for r in range(self.grid.size) for c in range(self.grid.size) if self.grid.get_cell(r, c) and self.grid.get_cell(r, c).no_fly)

        col_starts = [0.01, 0.34, 0.67]
        sections = [
            ("Drone Fleet", [("Total Drones", f"{total_drones}"), ("Active", f"{active_drones}"), ("Idle", f"{idle_drones}"), ("Avg Battery", f"{avg_battery:.1f}%")]),
            ("Deliveries", [("Total", f"{total_del}"), ("Completed", f"{completed}"), ("In Progress", f"{in_progress}"), ("Pending", f"{pending}")]),
            ("Infrastructure", [("Hubs", f"{hubs}"), ("Charging Pads", f"{chargers}"), ("Medical", f"{med_pkp}"), ("No-Fly Zones", f"{no_fly}")]),
        ]

        for (header, rows), x0 in zip(sections, col_starts):
            ax.text(x0, 0.95, header, transform=ax.transAxes, fontsize=8, color=TEXT_MID, va="top")
            for i, (label, value) in enumerate(rows):
                y = 0.74 - i * 0.175
                ax.text(x0 + 0.002, y, label + ":", transform=ax.transAxes, fontsize=8, color=TEXT_DIM, va="top")
                ax.text(x0 + 0.275, y, value, transform=ax.transAxes, fontsize=8, color=TEXT_HI, va="top", ha="right")

        ts = datetime.now().strftime("Updated %Y-%m-%d %H:%M:%S")
        ax.text(0.5, 0.03, ts, transform=ax.transAxes, fontsize=6.5, color=TEXT_DIM, ha="center")


def quick_visualization():
    print("AeroNet Lite — Visualization Suite")

    grid = create_demand_grid()

    print("Creating drone fleet...")
    selector = FleetSelector(grid, budget=10000)
    light, heavy, _ = selector.brute_force_select(verbose=False)
    drones = selector.create_fleet(light, heavy)
    print(f"  Created {len(drones)} drones")

    print("Creating deliveries...")
    sim = DeliverySimulator(grid, budget=10000)
    sim.drones = {d.drone_id: d for d in drones}
    sim.generate_deliveries(6)
    sim.assign_deliveries()
    print(f"  Created {len(sim.deliveries)} deliveries")

    visualizer = EnhancedGridVisualizer(grid)

    print("Rendering Master Dashboard with data...")
    fig1 = visualizer.plot_master_dashboard(drones=sim.drones, deliveries=sim.deliveries)
    plt.show()

    print("Rendering 3-D Demand Map...")
    fig2 = visualizer.plot_demand_heatmap_3d()
    plt.show()

    print("Done.")


if __name__ == "__main__":
    quick_visualization()