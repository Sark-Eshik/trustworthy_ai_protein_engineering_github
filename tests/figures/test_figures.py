# tests/figures/test_figures.py
"""Unit and integration tests for the Publication Figures module."""

import os
import pytest
import pandas as pd
import numpy as np

from src.figures.discovery import DiscoveryFigureGenerator
from src.figures.validation import ValidationFigureGenerator
from src.figures.integrated import IntegratedFigureGenerator
from src.figures.reliability import ReliabilityFigureGenerator
from src.figures.tkt_landscape import TktLandscapeFigureGenerator
from src.figures.candidate_discovery import CandidateDiscoveryFigureGenerator
from src.figures.killer_figure import KillerFigureGenerator
from src.figures.generate_all import MasterFigureRunner

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory fixture for saving figures."""
    out_dir = tmp_path / "test_figures"
    out_dir.mkdir()
    return str(out_dir)

def test_discovery_figure_generator(temp_output_dir):
    """Test that DiscoveryFigureGenerator builds PNG/PDF and performs Athlete Exercises."""
    generator = DiscoveryFigureGenerator()
    
    # Test file output
    png_path, pdf_path = generator.generate_figure(output_dir=temp_output_dir)
    assert os.path.exists(png_path)
    assert os.path.exists(pdf_path)
    assert png_path.endswith(".png")
    assert pdf_path.endswith(".pdf")

    # Test error handling on missing file
    with pytest.raises(FileNotFoundError):
        generator.generate_figure(results_path="non_existent_results.parquet", output_dir=temp_output_dir)

def test_validation_figure_generator(temp_output_dir):
    """Test that ValidationFigureGenerator builds PNG/PDF and performs Athlete Exercises."""
    generator = ValidationFigureGenerator()
    
    png_path, pdf_path = generator.generate_figure(output_dir=temp_output_dir)
    assert os.path.exists(png_path)
    assert os.path.exists(pdf_path)
    assert png_path.endswith(".png")
    assert pdf_path.endswith(".pdf")

    with pytest.raises(FileNotFoundError):
        generator.generate_figure(results_path="non_existent_results.parquet", output_dir=temp_output_dir)

def test_integrated_figure_generator(temp_output_dir):
    """Test that IntegratedFigureGenerator builds PNG/PDF."""
    generator = IntegratedFigureGenerator()
    
    png_path, pdf_path = generator.generate_figure(output_dir=temp_output_dir)
    assert os.path.exists(png_path)
    assert os.path.exists(pdf_path)
    assert png_path.endswith(".png")
    assert pdf_path.endswith(".pdf")

    with pytest.raises(FileNotFoundError):
        generator.generate_figure(results_path="non_existent_results.parquet", output_dir=temp_output_dir)

def test_reliability_figure_generator(temp_output_dir):
    """Test that ReliabilityFigureGenerator builds PNG/PDF."""
    generator = ReliabilityFigureGenerator()
    
    png_path, pdf_path = generator.generate_figure(output_dir=temp_output_dir)
    assert os.path.exists(png_path)
    assert os.path.exists(pdf_path)
    assert png_path.endswith(".png")
    assert pdf_path.endswith(".pdf")

    with pytest.raises(FileNotFoundError):
        generator.generate_figure(results_path="non_existent_results.parquet", output_dir=temp_output_dir)

def test_tkt_landscape_figure_generator(temp_output_dir):
    """Test that TktLandscapeFigureGenerator builds PNG/PDF."""
    generator = TktLandscapeFigureGenerator()
    
    f6_png, f6_pdf, f7_png, f7_pdf = generator.generate_figures(output_dir=temp_output_dir)
    assert os.path.exists(f6_png)
    assert os.path.exists(f6_pdf)
    assert os.path.exists(f7_png)
    assert os.path.exists(f7_pdf)

    with pytest.raises(FileNotFoundError):
        generator.generate_figures(analysis_path="non_existent.parquet", output_dir=temp_output_dir)

def test_candidate_discovery_figure_generator(temp_output_dir):
    """Test that CandidateDiscoveryFigureGenerator builds PNG/PDF."""
    generator = CandidateDiscoveryFigureGenerator()
    
    png_path, pdf_path = generator.generate_figure(output_dir=temp_output_dir)
    assert os.path.exists(png_path)
    assert os.path.exists(pdf_path)

    with pytest.raises(FileNotFoundError):
        generator.generate_figure(csv_path="non_existent.csv", output_dir=temp_output_dir)

def test_killer_figure_generator(temp_output_dir):
    """Test that KillerFigureGenerator builds PNG/PDF."""
    generator = KillerFigureGenerator()
    
    png_path, pdf_path = generator.generate_figure(output_dir=temp_output_dir)
    assert os.path.exists(png_path)
    assert os.path.exists(pdf_path)

def test_master_figure_runner(temp_output_dir):
    """Test that MasterFigureRunner builds all Figures F1 to F9 sequentially."""
    runner = MasterFigureRunner()
    runner.run_all(output_dir=temp_output_dir)

    expected_files = [
        "F1_project_architecture.png", "F1_project_architecture.pdf",
        "F2_combined_sparsity_vs_antisymmetry_error.png", "F2_combined_sparsity_vs_antisymmetry_error.pdf",
        "F3_combined_sparsity_vs_epistasis_error.png", "F3_combined_sparsity_vs_epistasis_error.pdf",
        "F4_antisymmetry_vs_epistasis_error.png", "F4_antisymmetry_vs_epistasis_error.pdf",
        "F5_reliability_distribution.png", "F5_reliability_distribution.pdf",
        "F6_tkt_reliability_landscape.png", "F6_tkt_reliability_landscape.pdf",
        "F7_tkt_industrial_fitness_landscape.png", "F7_tkt_industrial_fitness_landscape.pdf",
        "F8_top_candidate_mutations.png", "F8_top_candidate_mutations.pdf",
        "F9_killer_figure.png", "F9_killer_figure.pdf"
    ]

    for filename in expected_files:
        path = os.path.join(temp_output_dir, filename)
        assert os.path.exists(path), f"Expected figure {filename} was not generated by MasterFigureRunner!"
