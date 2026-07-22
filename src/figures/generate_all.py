# src/figures/generate_all.py
"""Master Runner to generate all 9 publication-quality figures for the project.

Generates:
- Figure F1: Project Architecture (computational block layout flow chart)
- Figure F2: Combined Sparsity vs Antisymmetry Error (Discovery)
- Figure F3: Combined Sparsity vs Epistasis Error (Validation)
- Figure F4: Antisymmetry vs Epistasis Error (Causal Coupling)
- Figure F5: Reliability Score Distribution
- Figure F6: TKT Reliability Landscape Scatter Map
- Figure F7: TKT Industrial Fitness Score Distribution
- Figure F8: Top 15 Prioritized TKT Engineering Candidates
- Figure F9: The multi-panel 'Killer Figure'
Exports each figure as both high-resolution PNG and vector PDF.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.infrastructure.config_loader import ConfigLoader, AppConfig
from src.infrastructure.logger import get_logger

from src.figures.discovery import DiscoveryFigureGenerator
from src.figures.validation import ValidationFigureGenerator
from src.figures.integrated import IntegratedFigureGenerator
from src.figures.reliability import ReliabilityFigureGenerator
from src.figures.tkt_landscape import TktLandscapeFigureGenerator
from src.figures.candidate_discovery import CandidateDiscoveryFigureGenerator
from src.figures.killer_figure import KillerFigureGenerator

class MasterFigureRunner:
    """Master runner coordinating the sequential generation and export of all figures."""

    def __init__(self, config_path: str = "configs") -> None:
        self.config_loader = ConfigLoader(base_path=config_path)
        self.config: AppConfig = self.config_loader.load_config()
        self.logger = get_logger(
            name="master_figure_runner",
            log_dir=self.config.paths.logs_dir,
            level=self.config.logging.level,
        )

    def generate_figure_f1_architecture(self, out_dir: str) -> list[str]:
        """Generates Figure F1: Project System Architecture Flow Chart."""
        self.logger.info("Generating Figure F1: Project System Architecture flow chart...")
        
        fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
        ax.axis("off")
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 8)

        # Style options
        box_style = dict(boxstyle="round,pad=0.5", facecolor="#ebf5fb", edgecolor="#2980b9", linewidth=1.5)
        box_style_val = dict(boxstyle="round,pad=0.5", facecolor="#fef9e7", edgecolor="#f1c40f", linewidth=1.5)
        box_style_app = dict(boxstyle="round,pad=0.5", facecolor="#eaf2f8", edgecolor="#8e44ad", linewidth=1.5)

        # Helper to draw colored boxes
        def draw_box(x, y, text, style, fontsize=10):
            ax.text(x, y, text, bbox=style, fontsize=fontsize, fontweight="bold", ha="center", va="center")

        # Helper to draw arrows
        def draw_arrow(x1, y1, x2, y2, color="#7f8c8d"):
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.0, shrinkA=5, shrinkB=5)
            )

        # Render stages: Discovery (blue), Validation (yellow), Application (purple)
        # Stage titles
        ax.text(1.8, 7.5, "DISCOVERY PHASE (Sparsity Models)", fontsize=11, fontweight="bold", color="#1c5980", ha="center")
        ax.text(5.5, 7.5, "VALIDATION PHASE (Model Error Coupling)", fontsize=11, fontweight="bold", color="#7f6000", ha="center")
        ax.text(9.2, 7.5, "APPLICATION PHASE (Industrial Enzyme)", fontsize=11, fontweight="bold", color="#5e2273", ha="center")

        # Discovery Panel
        draw_box(1.8, 6.5, "Source Datasets\n(Megascale-D Mutation counts)\n[S001]", box_style)
        draw_box(1.8, 5.0, "Multi-Dimensional Sparsity\n(Empirical, Evolutionary, Structural)\n[D101, D102, D103]", box_style)
        draw_box(1.8, 3.5, "Combined Sparsity\n(Unified metric average)\n[D104]", box_style)
        draw_box(1.8, 2.0, "Experiment 1: Antisymmetry\n(Thermodynamic inconsistency)\n[D202]", box_style)

        # Arrows Discovery
        draw_arrow(1.8, 6.0, 1.8, 5.5)
        draw_arrow(1.8, 4.5, 1.8, 4.0)
        draw_arrow(1.8, 3.0, 1.8, 2.5)

        # Validation Panel
        draw_box(5.5, 6.5, "Experiment 2: Epistasis\n(Double mutant interactions)\n[D301]", box_style_val)
        draw_box(5.5, 4.5, "Epistasis Prediction Error\n(Model failure rates)\n[D302]", box_style_val)
        draw_box(5.5, 2.5, "Integrated Error Analysis\n(Combined Sparsity vs. Error Causal Path)\n[D401]", box_style_val)

        # Arrows Validation
        draw_arrow(5.5, 6.0, 5.5, 5.0)
        draw_arrow(5.5, 4.0, 5.5, 3.0)

        # Cross-Arrows linking Discovery and Validation
        draw_arrow(1.8, 1.5, 5.5, 2.0) # Antisymmetry to integrated
        draw_arrow(5.5, 2.0, 1.8, 2.0) # Bidirectional line
        draw_arrow(1.8, 3.0, 5.5, 2.5) # Combined Sparsity to Integrated

        # Application Panel
        draw_box(9.2, 6.5, "Reliability score Framework\n(Reliability = 1 - Sparsity)\n[D402]", box_style_app)
        draw_box(9.2, 4.8, "TKT Mutation Landscape\n(All 11,400 single substitutions)\n[D502]", box_style_app)
        draw_box(9.2, 3.1, "Active-Site Proximity Controls\n(Controls for spatial biases)\n[D503]", box_style_app)
        draw_box(9.2, 1.4, "Industrial Fitness Score\n(Priortized wet-lab candidates)\n[D602]", box_style_app)

        # Arrows Application
        draw_arrow(9.2, 6.0, 9.2, 5.3)
        draw_arrow(9.2, 4.3, 9.2, 3.6)
        draw_arrow(9.2, 2.6, 9.2, 1.9)

        # Link validation to application
        draw_arrow(5.5, 2.0, 9.2, 6.5, color="#8e44ad")

        ax.set_title("Figure F1: Trustworthy AI Protein Engineering Master Project Architecture", fontsize=14, fontweight="bold", pad=20, ha="center")
        plt.tight_layout()

        png_path = os.path.join(out_dir, "F1_project_architecture.png")
        pdf_path = os.path.join(out_dir, "F1_project_architecture.pdf")
        
        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path, format="pdf")
        plt.close()
        
        self.logger.info(f"Figure F1 saved successfully to {png_path} and {pdf_path}")
        return [png_path, pdf_path]

    def run_all(self, output_dir: str = None) -> None:
        """Executes all figure submodules sequentially to generate Figures F1 through F9."""
        out_dir = output_dir or os.path.join(os.getcwd(), self.config.paths.figures_dir)
        os.makedirs(out_dir, exist_ok=True)

        self.logger.info(f"Master Figures Generator starting. Saving all high-res outputs to: {out_dir}")
        print(f"Generating all high-resolution figures in {out_dir}...")

        # Figure F1: Project System Architecture
        self.generate_figure_f1_architecture(out_dir)

        # Figure F2: Combined Sparsity vs Antisymmetry (Discovery)
        f2_gen = DiscoveryFigureGenerator(config_path=os.path.dirname(self.config_loader._config_paths[0]) if hasattr(self.config_loader, '_config_paths') else "configs")
        f2_gen.generate_figure(output_dir=out_dir)

        # Figure F3: Combined Sparsity vs Epistasis (Validation)
        f3_gen = ValidationFigureGenerator(config_path=os.path.dirname(self.config_loader._config_paths[0]) if hasattr(self.config_loader, '_config_paths') else "configs")
        f3_gen.generate_figure(output_dir=out_dir)

        # Figure F4: Antisymmetry vs Epistasis Error (Integrated)
        f4_gen = IntegratedFigureGenerator(config_path=os.path.dirname(self.config_loader._config_paths[0]) if hasattr(self.config_loader, '_config_paths') else "configs")
        f4_gen.generate_figure(output_dir=out_dir)

        # Figure F5: Reliability Score Distribution
        f5_gen = ReliabilityFigureGenerator(config_path=os.path.dirname(self.config_loader._config_paths[0]) if hasattr(self.config_loader, '_config_paths') else "configs")
        f5_gen.generate_figure(output_dir=out_dir)

        # Figures F6 & F7: TKT Reliability Landscape and Industrial Fitness Score Distribution
        f6_7_gen = TktLandscapeFigureGenerator(config_path=os.path.dirname(self.config_loader._config_paths[0]) if hasattr(self.config_loader, '_config_paths') else "configs")
        f6_7_gen.generate_figures(output_dir=out_dir)

        # Figure F8: Top candidate horizontal bar plot
        f8_gen = CandidateDiscoveryFigureGenerator(config_path=os.path.dirname(self.config_loader._config_paths[0]) if hasattr(self.config_loader, '_config_paths') else "configs")
        f8_gen.generate_figure(output_dir=out_dir)

        # Figure F9: The flagship 'Killer Figure'
        f9_gen = KillerFigureGenerator(config_path=os.path.dirname(self.config_loader._config_paths[0]) if hasattr(self.config_loader, '_config_paths') else "configs")
        f9_gen.generate_figure(output_dir=out_dir)

        print(f"SUCCESS: Generated Figures F1 to F9 in both high-res PNG and vector PDF.")
        self.logger.info("All publication-quality figures successfully compiled.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Master script to generate all 9 publication figures.")
    parser.add_argument("--config", type=str, default="configs", help="Path to config directory.")
    parser.add_argument("--output", type=str, default=None, help="Directory to save generated figures.")
    args = parser.parse_args()

    try:
        runner = MasterFigureRunner(config_path=args.config)
        runner.run_all(output_dir=args.output)
        print("Done")
    except Exception as e:
        print(f"ERROR: Master Figure Generation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
