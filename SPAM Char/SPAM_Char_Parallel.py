from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import List, Optional, Sequence, Tuple, Any
from qiskit_ibm_provider import IBMProvider
from qiskit.providers.backend import Backend
from qiskit_experiments.framework import BaseExperiment, Options, AnalysisResultData
from qiskit_ibm_runtime import Options as IBMOptions
from qiskit.providers.fake_provider import FakeLagos
from qiskit_aer.noise import NoiseModel
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.framework import (
    BaseAnalysis,
    CompositeAnalysis,
    ExperimentData,
    AnalysisResultData
)
import numpy as np
import matplotlib.pyplot as plt
import os

class SPAMParams(BaseExperiment):
    """Custom experiment class template."""
    def __init__(self,
                 physical_qubits: Sequence[int],
                 backend: Optional[Backend] = None,
                 **kwargs):
        self.num_exps = kwargs['num_exps']
        #self.physical_qubits = physical_qubits
        analysis = SPAMParamsAnalysis()
        """Initialize the experiment."""
        super().__init__(physical_qubits,
                        backend = backend,
                        analysis = analysis)
    def circuits(self) -> List[QuantumCircuit]:
        """Generate the list of circuits to be run."""
        circuits = []
        # Generate circuits and populate metadata here
        for i in range(self.num_exps):
            circ = QuantumCircuit(1,1)
            circ.measure(0,0)
            circuits.append(circ)
        for i in range(self.num_exps):
            circ = QuantumCircuit(1,1)
            circ.x(0)
            circ.measure(0,0)
            circuits.append(circ)
        for i in range(self.num_exps):
            qreg_q = QuantumRegister(1, 'q')
            creg_c = ClassicalRegister(1, 'c')
            creg_mcm = ClassicalRegister(1, 'mcm')
            circ = QuantumCircuit(qreg_q, creg_mcm, creg_c)
            circ.measure(qreg_q, creg_mcm)
            circ.barrier()
            circ.measure(qreg_q, creg_c)
            circuits.append(circ)
        return circuits

    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Set default experiment options here."""
        options = super()._default_experiment_options()
        #options.update_options(
        #    dummy_option = None,
        #)
        return options

class SPAMParamsAnalysis(BaseAnalysis):
    def __init__(self):
        super().__init__()

    def _run_analysis(self, experiment_data: ExperimentData):
        data = experiment_data.data()
        p0to1_arr = np.zeros((int(len(data)/3), 1))
        for i in range(int(len(data)/3)):
            if data[i]['counts']['0'] == sum(data[i]['counts'].values()):
                p0to1_arr[i] = 0
            else:
                p0to1_arr[i] = 1 - (data[i]['counts']['0'])/sum(data[i]['counts'].values())
        p1to0_arr = np.zeros((int(len(data)/3), 1))
        for i in range(int(len(data)/3), int(2*len(data)/3)):
            if data[i]['counts']['1'] == sum(data[i]['counts'].values()):
                p1to0_arr[i-int(len(data)/3)] = 0
            else:
                p1to0_arr[i-int(len(data)/3)] = 1 - (data[i]['counts']['1'])/sum(data[i]['counts'].values())
        p0to0to0_arr = np.zeros((int(len(data)/3), 1))
        for i in range(int(2*len(data)/3), len(data)):
            # check if a particular key exists in dictionary
            if '10' not in data[i]['counts']:
                c0 = data[i]['counts']['00']
            else:
                c0 = data[i]['counts']['00']+data[i]['counts']['10']
            p0to0to0_arr[i-int(2*len(data)/3)] = data[i]['counts']['00']/c0
        alpha_arr = 1 - (p0to1_arr + p1to0_arr)
        delta_arr = p1to0_arr - p0to1_arr
        alpha_m_arr = np.sqrt(2.*p0to0to0_arr*(1.+alpha_arr+delta_arr)-2.*alpha_arr*(1.+delta_arr)-(1.+delta_arr)**2)
        alpha_sp_arr = alpha_arr/alpha_m_arr
        analysis_results = [AnalysisResultData(name='alpha', value=alpha_arr),
                            AnalysisResultData(name='delta', value=delta_arr),
                            AnalysisResultData(name='alpha_m', value=alpha_m_arr),
                            AnalysisResultData(name='alpha_sp', value=alpha_sp_arr)]
        figures = []
        return analysis_results, figures




# Make a noise model
fake_backend = FakeLagos()
noise_model = NoiseModel.from_backend(fake_backend)

# Set options to include the noise model
options = IBMOptions()
options.simulator = {
    "noise_model": noise_model,
    "basis_gates": fake_backend.configuration().basis_gates,
    "coupling_map": fake_backend.configuration().coupling_map,
    "seed_simulator": 123
}


# Set number of shots, optimization_level and resilience_level
options.execution.shots = 4000
options.optimization_level = 0
options.resilience_level = 0

IBMProvider.save_account(token="4fd31868f9a515dcfab559da15732676ca598d027e6ef67b42fd93f506cad6dca269609c7519717a74c99644eae00d7eee8ca97d1f3a4b83c30a6978d628ebbb", overwrite=True)
provider = IBMProvider(instance="ibm-q-ornl/ornl/phy147")
backend_name = 'ibm_osaka'
backend = provider.get_backend(backend_name)


# create a quantum circuit with n qubits and n classical bits
qubit_set = [0,1,2,3,4,5,6]
num_qubits = len(qubit_set)#backend.num_qubits
num_exps = 100
num_classical_bits = num_qubits
foldername = backend_name+'_joint_'+str(num_qubits)+'q_SPAMjobs_1'

exp_list = []
for i in qubit_set:
    exp_list.append(SPAMParams(physical_qubits=(i,), backend=backend, num_exps=num_exps, analysis=SPAMParamsAnalysis()))
parallel_exp = ParallelExperiment(exp_list)
# run the experiment
parallel_run = parallel_exp.run(backend=backend).block_for_results()
# save the experiment's job id for later

# sort data by qubit
alpha_arr = np.zeros((num_exps, num_qubits))
alpha_m_arr = np.zeros((num_exps, num_qubits))
alpha_sp_arr = np.zeros((num_exps, num_qubits))
delta_arr = np.zeros((num_exps, num_qubits))

for i in range(num_qubits):
    alpha_arr[:, i] = np.ndarray.flatten(parallel_run.child_data(i).analysis_results('alpha').value)
    alpha_m_arr[:, i] = np.ndarray.flatten(parallel_run.child_data(i).analysis_results('alpha_m').value)
    alpha_sp_arr[:, i] = np.ndarray.flatten(parallel_run.child_data(i).analysis_results('alpha_sp').value)
    delta_arr[:, i] = np.ndarray.flatten(parallel_run.child_data(i).analysis_results('delta').value)

# create a new folder in the current directory called x-qubit_SPAMjobs

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, foldername)
if not os.path.exists(final_directory):
    os.makedirs(final_directory)

# save the experiment data to the folder
np.save(foldername + "/alpha_arr", alpha_arr)
np.save(foldername + "/alpha_m_arr", alpha_m_arr)
np.save(foldername + "/alpha_sp_arr", alpha_sp_arr)
np.save(foldername + "/delta_arr", delta_arr)
np.save(foldername + "/qubit_set", np.array(qubit_set))

# plot and save the results to the folder
x_axis = np.array(qubit_set)

plt.boxplot(alpha_arr, meanline=True)
plt.title("SPAM-Error")
plt.xlabel("Qubit Number")
plt.ylabel(r"$\alpha$")
plt.ylim(0.8, 1.2)
plt.savefig(foldername + "/SPAM-Error.png")
plt.show()

plt.boxplot(alpha_sp_arr, meanline=True)
plt.title("SP-Error")
plt.xlabel("Qubit Number")
plt.ylabel(r"$\alpha_{SP}$")
plt.ylim(0.8, 1.2)
plt.savefig(foldername + "/SP-Error.png")
plt.show()

plt.boxplot(alpha_m_arr, meanline=True)
plt.title("M-Error")
plt.xlabel("Qubit Number")
plt.ylabel(r"$\alpha_M$")
plt.ylim(0.8, 1.2)
plt.savefig(foldername + "/M-Error.png")
plt.show()

plt.boxplot(delta_arr, meanline=True)
plt.title("Delta")
plt.xlabel("Qubit Number")
plt.ylabel(r"$\delta$")
plt.savefig(foldername + "/Delta.png")

