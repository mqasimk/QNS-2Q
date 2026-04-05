"""Shared test fixtures and path configuration for QNS-2Q tests."""

import sys
import os

# Add src/ to the Python path so we can import modules directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def time_vector():
    """A standard time vector for testing (1000 points over [0, 4e-6])."""
    return np.linspace(0, 4e-6, 1000)


@pytest.fixture
def small_time_vector():
    """A small time vector for fast tests (100 points over [0, 1e-6])."""
    return np.linspace(0, 1e-6, 100)


@pytest.fixture
def frequency_grid():
    """A standard frequency grid for testing."""
    return jnp.linspace(-1e8, 1e8, 501)


@pytest.fixture
def small_frequency_grid():
    """A small frequency grid for fast tests."""
    return jnp.linspace(-5e7, 5e7, 101)
