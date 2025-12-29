"""
Test Script for Pulse Sequence Construction

This script verifies the implementation of the pulse sequence generation
functions (CDD, mqCDD) from ID_opt_v3.py by generating and plotting their
corresponding switching functions for qubit 1, qubit 2, and the 1-2 interaction.
"""

import matplotlib.pyplot as plt
import numpy as np
from ID_opt_v3 import cddn, mqCDD, make_tk12

def plot_switching_functions(pulse_times_q1, pulse_times_q2, pulse_times_q12, T_seq, title):
    """
    Plots the switching functions y(t) for a pair of pulse sequences and their interaction.
    
    Args:
        pulse_times_q1 (array): Pulse times for qubit 1.
        pulse_times_q2 (array): Pulse times for qubit 2.
        pulse_times_q12 (array): Pulse times for the 1-2 interaction.
        T_seq (float): The total duration of the sequence.
        title (str): The title for the plot.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=16)
    
    def get_switching_function(pulse_times, T, num_points=2000):
        t_grid = np.linspace(0, T, num_points)
        y = np.ones_like(t_grid)
        # pulse_times includes 0 and T. Internal pulses are at indices 1:-1
        internal_pulses = pulse_times[1:-1]
        for t_pulse in internal_pulses:
            y[t_grid >= t_pulse] *= -1
        return t_grid, y

    # Qubit 1
    t1, y1 = get_switching_function(pulse_times_q1, T_seq)
    axs[0].step(t1, y1, 'r-', where='post')
    axs[0].set_title("Qubit 1 Switching Function")
    axs[0].set_ylabel("y1(t)")
    axs[0].set_ylim(-1.2, 1.2)
    axs[0].grid(True, alpha=0.3)
    
    # Qubit 2
    t2, y2 = get_switching_function(pulse_times_q2, T_seq)
    axs[1].step(t2, y2, 'b-', where='post')
    axs[1].set_title("Qubit 2 Switching Function")
    axs[1].set_ylabel("y2(t)")
    axs[1].set_ylim(-1.2, 1.2)
    axs[1].grid(True, alpha=0.3)

    # 1-2 Interaction
    t12, y12 = get_switching_function(pulse_times_q12, T_seq)
    axs[2].step(t12, y12, 'g-', where='post')
    axs[2].set_title("1-2 Interaction Switching Function")
    axs[2].set_ylabel("y12(t)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylim(-1.2, 1.2)
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    return fig

if __name__ == "__main__":
    T_base = 1.0  # Use a simple time unit for clarity

    # --- Test 1: Basic CDD Sequence ---
    cdd_order = 2
    pt_cdd_q1 = np.array(cddn(0., T_base, cdd_order))
    pt_cdd_q2 = np.array(cddn(0., T_base, cdd_order))
    pt_cdd_q12 = make_tk12(pt_cdd_q1, pt_cdd_q2)
    
    fig1 = plot_switching_functions(pt_cdd_q1, pt_cdd_q2, pt_cdd_q12, T_base, f"CDD-{cdd_order} Sequence")
    fig1.savefig("test_cdd_construction.pdf")
    print("Saved CDD test plot to test_cdd_construction.pdf")

    # --- Test 2: Basic mqCDD Sequence ---
    n_mqcdd, m_mqcdd = 1, 2
    pt_mq_q1_list, pt_mq_q2_list = mqCDD(T_base, n_mqcdd, m_mqcdd)
    pt_mq_q1 = np.array(pt_mq_q1_list)
    pt_mq_q2 = np.array(pt_mq_q2_list)
    pt_mq_q12 = make_tk12(pt_mq_q1, pt_mq_q2)

    fig2 = plot_switching_functions(pt_mq_q1, pt_mq_q2, pt_mq_q12, T_base, f"mqCDD(n={n_mqcdd}, m={m_mqcdd}) Sequence")
    fig2.savefig("test_mqcdd_construction.pdf")
    print("Saved mqCDD test plot to test_mqcdd_construction.pdf")

    # --- Test 3: Asymmetric CDD Sequence ---
    cdd_order_1 = 1
    cdd_order_2 = 3
    pt_cdd_asym_q1 = np.array(cddn(0., T_base, cdd_order_1))
    pt_cdd_asym_q2 = np.array(cddn(0., T_base, cdd_order_2))
    pt_cdd_asym_q12 = make_tk12(pt_cdd_asym_q1, pt_cdd_asym_q2)
    
    fig3 = plot_switching_functions(pt_cdd_asym_q1, pt_cdd_asym_q2, pt_cdd_asym_q12, T_base, f"Asymmetric CDD({cdd_order_1}, {cdd_order_2}) Sequence")
    fig3.savefig("test_asymmetric_cdd_construction.pdf")
    print("Saved Asymmetric CDD test plot to test_asymmetric_cdd_construction.pdf")

    plt.show()
