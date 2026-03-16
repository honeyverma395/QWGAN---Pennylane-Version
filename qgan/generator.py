"""Generator module — PennyLane rewrite.

Replaces: generator.py, qcircuit.py, qgates.py, optimizer.py, data_reshapers.py

The generator is a variational quantum circuit (ansatz) that acts on one half
of a maximally-entangled state (Choi representation).  PennyLane handles gate
definitions, circuit simulation and automatic differentiation, so the five
files listed above collapse into this single module.

Like before:
    - Choi representation: I \otimes G applied to |\Phi^+> (with optional ancilla |0>).
    - Three ansatz modes: ZZ_Z_X, ZZ_YY_XX_Z, and custom.
    - Full ancilla support: topologies (disconnected, ansatz, bridge, total, fake)
      and post-processing modes (pass, project, trace) via ancilla.py.

Typical usage (inside training.py):
    gen = Generator()
    state = gen.get_total_gen_state()          # forward pass
    gen.update_gen(dis, final_target_state)     # backward + optimiser step
"""

import os
import pickle
from typing import Optional

import numpy as np
import pennylane as qml

from config import CFG
from qgan.ancilla import get_final_gen_state_for_discriminator
from qgan.cost_functions import braket
from qgan.discriminator import Discriminator
from data.data_managers import print_and_log


# -- GATE TERM DEFINITIONS ---------------------------------
# Maps term strings to (PennyLane gate, n_qubits)
# Same logic as in target.py
_1Q_GATES = {
    "X": qml.RX,
    "Y": qml.RY,
    "Z": qml.RZ,
}

_2Q_GATES = {
    "XX": qml.IsingXX,
    "YY": qml.IsingYY,
    "ZZ": qml.IsingZZ,
}

# Predefined ansatz 
_PREDEFINED_ANSATZ = {
    "ZZ_Z_X":     ["X", "Z", "ZZ"],
    "ZZ_YY_XX_Z": ["Z", "XX", "YY", "ZZ"],
}


def _get_ansatz_terms() -> list[str]:
    """Get the list of gate terms for the current ansatz config."""
    if CFG.gen_ansatz == "custom":
        if not CFG.custom_ansatz_terms:
            raise ValueError("custom ansatz requires custom_ansatz_terms to be set.")
        return CFG.custom_ansatz_terms
    if CFG.gen_ansatz in _PREDEFINED_ANSATZ:
        return _PREDEFINED_ANSATZ[CFG.gen_ansatz]
    raise ValueError(
        f"Unknown ansatz: {CFG.gen_ansatz}. "
        f"Expected one of: {list(_PREDEFINED_ANSATZ.keys())} or 'custom'."
    )


#-- DEVICE ---------------------------------
def _make_device(total_wires: int):
    """Create a PennyLane device for the full Choi + generator register."""
    return qml.device("default.qubit", wires=total_wires)



# -- UNIFIED ANSATZ  (handles all three modes: ZZ_Z_X, ZZ_YY_XX_Z, custom) ---------------------------------
class Ansatz:
    """Applies gates inside a QNode context.

    All ansatz types follow the same pattern per layer:
        For each term in the term list:
            - 1q terms (X, Y, Z): apply rotation on each system qubit
            - 2q terms (XX, YY, ZZ): apply Ising gate between neighbouring system qubits
        Then ancilla gates (1q or 2q couplings) if configured.
    """

    @staticmethod
    def apply(params, gen_wires: list[int], ancilla_wire: Optional[int]):
        """Apply the ansatz defined in CFG to the generator wires."""
        terms = _get_ansatz_terms()
        system_wires = [w for w in gen_wires if w != ancilla_wire]
        n_sys = len(system_wires)
        idx = 0

        for _ in range(CFG.gen_layers):
            # -- System gates from term list --
            for term in terms:
                if term in _1Q_GATES:
                    gate_fn = _1Q_GATES[term]
                    for w in system_wires:
                        gate_fn(params[idx], wires=w);  idx += 1
                elif term in _2Q_GATES:
                    gate_fn = _2Q_GATES[term]
                    for i in range(n_sys - 1):
                        gate_fn(params[idx], wires=[system_wires[i], system_wires[i + 1]])
                        idx += 1
                else:
                    raise ValueError(f"Unknown gate term: {term}")

            # -- Ancilla 1q gates --
            if ancilla_wire is not None and CFG.do_ancilla_1q_gates:
                for term in terms:
                    if term in _1Q_GATES:
                        _1Q_GATES[term](params[idx], wires=ancilla_wire);  idx += 1

            # -- Ancilla 2q couplings --
            if ancilla_wire is not None:
                idx = Ansatz._apply_ancilla_couplings(
                    params, idx, terms, system_wires, ancilla_wire
                )

    @staticmethod
    def _apply_ancilla_couplings(params, idx: int, terms: list[str],
                                  system_wires: list[int],
                                  ancilla_wire: int) -> int:
        """Apply ancilla 2q couplings according to topology.

        Uses the same 2q gate types from the term list.
        Returns updated param index.
        """
        n_sys = len(system_wires)
        topo = CFG.ancilla_topology

        # Resolve connect_to wire
        connect_to = CFG.ancilla_connect_to
        if connect_to is not None and isinstance(connect_to, int) and connect_to < n_sys:
            connect_wire = system_wires[connect_to]
        else:
            connect_wire = system_wires[-1]

        # Collect the 2q gate functions from the term list
        gates_2q = [_2Q_GATES[t] for t in terms if t in _2Q_GATES]

        def _apply_coupling(wire_a, wire_b):
            nonlocal idx
            for gate_fn in gates_2q:
                gate_fn(params[idx], wires=[wire_a, wire_b]);  idx += 1

        if topo == "total":
            for w in system_wires:
                _apply_coupling(w, ancilla_wire)

        if topo == "bridge":
            _apply_coupling(system_wires[0], ancilla_wire)

        if topo in ("bridge", "ansatz"):
            _apply_coupling(connect_wire, ancilla_wire)

        if topo == "fake" and n_sys > 2:
            _apply_coupling(system_wires[0], connect_wire)

        return idx


#-- PARAMETER COUNTING ---------------------------------
def count_params(n_system: int, has_ancilla: bool) -> int:
    """Total number of trainable angles for the current ansatz + config."""
    terms = _get_ansatz_terms()
    n = 0

    # Count 1q and 2q gate types in the term list
    n_1q_terms = sum(1 for t in terms if t in _1Q_GATES)
    n_2q_terms = sum(1 for t in terms if t in _2Q_GATES)

    for _ in range(CFG.gen_layers):
        # System 1q gates: each 1q term × n_system qubits
        n += n_1q_terms * n_system

        # System 2q gates: each 2q term × (n_system - 1) pairs (non-cyclical)
        n += n_2q_terms * (n_system - 1)

        # Ancilla 1q gates
        if has_ancilla and CFG.do_ancilla_1q_gates:
            n += n_1q_terms

        # Ancilla 2q couplings
        if has_ancilla:
            n += _count_ancilla_coupling_params(n_2q_terms, n_system)

    return n


def _count_ancilla_coupling_params(n_2q_terms: int, n_system: int) -> int:
    """Count params added by one layer of ancilla couplings."""
    topo = CFG.ancilla_topology
    n = 0
    if topo == "total":
        n += n_system * n_2q_terms
    if topo == "bridge":
        n += n_2q_terms
    if topo in ("bridge", "ansatz"):
        n += n_2q_terms
    if topo == "fake" and n_system > 2:
        n += n_2q_terms
    return n


#-- WIRE LAYOUT ---------------------------------
def _wire_layout():
    """Compute wire indices for each register.

    Layout:  [ choi_register | gen_system | (ancilla) ]
    """
    s = CFG.system_size
    choi_wires = list(range(s))
    gen_system_wires = list(range(s, 2 * s))

    if CFG.extra_ancilla:
        ancilla_wire = 2 * s
        gen_wires = gen_system_wires + [ancilla_wire]
        total_wires = 2 * s + 1
    else:
        ancilla_wire = None
        gen_wires = gen_system_wires
        total_wires = 2 * s

    return choi_wires, gen_wires, ancilla_wire, total_wires


#-- QNODE ---------------------------------
def _build_qnode():
    """Build the QNode for the generator circuit.

    1) Prepare maximally-entangled state (Choi) via H + CNOT.
    2) Apply ansatz on generator register.
    3) Return full statevector.
    """
    choi_wires, gen_wires, ancilla_wire, total_wires = _wire_layout()
    dev = _make_device(total_wires)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(params):
        gen_system_wires = [w for w in gen_wires if w != ancilla_wire]
        for c_wire, g_wire in zip(choi_wires, gen_system_wires):
            qml.Hadamard(wires=c_wire)
            qml.CNOT(wires=[c_wire, g_wire])

        Ansatz.apply(params, gen_wires, ancilla_wire)
        return qml.state()

    return circuit



# -- GENERATOR CLASS ---------------------------------
class Generator:
    """Generator for the Quantum WGAN (PennyLane version).

        gen.get_total_gen_state()      -> full Choi statevector
        gen.update_gen(dis, target)    -> one optimisation step
        gen.load_model_params(path)    -> restore from checkpoint
        gen.params                     -> current angles (np.ndarray)
    """

    def __init__(self):
        # -- save/load compatibility --
        self.size: int = CFG.system_size + (1 if CFG.extra_ancilla else 0)
        self.ancilla: bool = CFG.extra_ancilla
        self.ancilla_topology: str = CFG.ancilla_topology
        self.ansatz: str = CFG.gen_ansatz
        self.layers: int = CFG.gen_layers
        self.target_size: int = CFG.system_size
        self.target_hamiltonian: str = CFG.target_hamiltonian

        # -- parameters --
        n_params = count_params(CFG.system_size, CFG.extra_ancilla)
        self.params: np.ndarray = self._init_params(n_params)
        self.n_params: int = n_params

        # -- circuit --
        self.circuit = _build_qnode()

        # -- optimiser --
        self.optimizer = qml.MomentumOptimizer(
            stepsize=CFG.l_rate,
            momentum=CFG.momentum_coeff,
        )

        # cached state
        self.total_gen_state: np.ndarray = self.get_total_gen_state()

    # -- pickle support ---------------------------------
    def __getstate__(self):
        """Exclude non-picklable objects (QNode, optimizer) from serialisation."""
        state = self.__dict__.copy()
        state.pop('circuit', None)
        state.pop('optimizer', None)
        return state

    def __setstate__(self, state):
        """Restore from pickle: rebuild QNode and optimizer."""
        self.__dict__.update(state)
        self.circuit = _build_qnode()
        self.optimizer = qml.MomentumOptimizer(
            stepsize=CFG.l_rate,
            momentum=CFG.momentum_coeff,
        )

    # -- parameter initialisation ---------------------------------
    def _init_params(self, n_params: int) -> np.ndarray:
        """Random uniform in [0, 2 \pi), respecting start_ancilla_gates_randomly."""
        params = np.random.uniform(0, 2 * np.pi, n_params)

        if CFG.extra_ancilla and not CFG.start_ancilla_gates_randomly:
            ancilla_indices = self._get_ancilla_param_indices()
            params[ancilla_indices] = 0.0

        return params

    def _get_ancilla_param_indices(self) -> list[int]:
        """Identify which param indices correspond to ancilla-only gates."""
        terms = _get_ansatz_terms()
        n_1q_terms = sum(1 for t in terms if t in _1Q_GATES)
        n_2q_terms = sum(1 for t in terms if t in _2Q_GATES)
        n_sys = CFG.system_size

        indices = []
        idx = 0

        for _ in range(CFG.gen_layers):
            # System 1q + 2q gates (skip)
            idx += n_1q_terms * n_sys
            idx += n_2q_terms * (n_sys - 1)

            # Ancilla 1q gates
            if CFG.do_ancilla_1q_gates:
                for _ in range(n_1q_terms):
                    indices.append(idx);  idx += 1

            # Ancilla 2q couplings
            n_anc_2q = _count_ancilla_coupling_params(n_2q_terms, n_sys)
            for _ in range(n_anc_2q):
                indices.append(idx);  idx += 1

        return indices

    # -- forward pass ---------------------------------
    def get_total_gen_state(self) -> np.ndarray:
        """Run the circuit and return full statevector as column vector."""
        state = self.circuit(self.params)
        return np.asmatrix(np.array(state).reshape(-1, 1))

    # -- gradient + optimiser step ---------------------------------
    def update_gen(self, dis: Discriminator, final_target_state: np.ndarray):
        """One generator optimisation step (minimisation)."""
        grad = self._grad_theta(dis, final_target_state)
        self._apply_momentum_step(grad)
        self.total_gen_state = self.get_total_gen_state()

    def _apply_momentum_step(self, grad: np.ndarray):
        """Manual momentum update (minimisation).

        v_{t+1} = \mu ·v_t - \eta · \nabla
        \theta_{t+1} = \theta_t + v_{t+1}
        """
        if not hasattr(self, '_velocity'):
            self._velocity = -self.optimizer.stepsize * grad
        else:
            self._velocity = (
                self.optimizer.momentum * self._velocity
                - self.optimizer.stepsize * grad
            )
        self.params = self.params + self._velocity

    def _grad_theta(self, dis: Discriminator,
                    final_target_state: np.ndarray) -> np.ndarray:
        """Compute \partial Loss/ \partial \theta for all generator parameters.

        Loss = \psi_term - \phi_term - reg_term
        \psi depends only on target -> \partial \psi / \partial \theta = 0
        """
        A, B, _, phi = dis.get_dis_matrices_rep() #check discriminator.py

        total_gen_state = self.total_gen_state
        final_gen_state = get_final_gen_state_for_discriminator(total_gen_state) #check ancilla.py

        grad_g_phi = np.zeros(self.n_params)
        grad_g_reg = np.zeros(self.n_params)

        for i in range(self.n_params):
            total_gen_grad = self._param_shift_grad_state(i)
            final_gen_grad = get_final_gen_state_for_discriminator(total_gen_grad)

            # \phi term
            tmp_phi = (braket(final_gen_grad, phi, final_gen_state)
                       + braket(final_gen_state, phi, final_gen_grad))
            grad_g_phi[i] = np.real(np.ndarray.item(tmp_phi))

            # Regularisation (8 braket terms)
            # fmt: off
            t1 = braket(final_gen_grad, A, final_gen_state) * braket(final_target_state, B, final_target_state)
            t2 = braket(final_gen_state, A, final_gen_grad)  * braket(final_target_state, B, final_target_state)
            t3 = braket(final_gen_grad, B, final_target_state) * braket(final_target_state, A, final_gen_state)
            t4 = braket(final_gen_state, B, final_target_state) * braket(final_target_state, A, final_gen_grad)
            t5 = braket(final_gen_grad, A, final_target_state) * braket(final_target_state, B, final_gen_state)
            t6 = braket(final_gen_state, A, final_target_state) * braket(final_target_state, B, final_gen_grad)
            t7 = braket(final_gen_grad, B, final_gen_state)    * braket(final_target_state, A, final_target_state)
            t8 = braket(final_gen_state, B, final_gen_grad)    * braket(final_target_state, A, final_target_state)
            tmp_reg = CFG.lamb / np.e * (CFG.cst1 * (t1 + t2) - CFG.cst2 * (t3 + t4 + t5 + t6) + CFG.cst3 * (t7 + t8))
            # fmt: on
            grad_g_reg[i] = np.real(np.ndarray.item(tmp_reg))

        return -(grad_g_phi + grad_g_reg)

    def _param_shift_grad_state(self, param_idx: int,
                                shift: float = np.pi / 2) -> np.ndarray:
        """\partial |\psi(\theta)>/\partial \theta_i 
        via parameter-shift rule on statevector.
        """
        params_plus = self.params.copy()
        params_plus[param_idx] += shift

        params_minus = self.params.copy()
        params_minus[param_idx] -= shift

        state_plus = np.array(self.circuit(params_plus)).reshape(-1, 1)
        state_minus = np.array(self.circuit(params_minus)).reshape(-1, 1)

        return np.asmatrix((state_plus - state_minus) / (2 * np.sin(shift)))

    # -- SAVE / LOAD ---------------------------------
    # No big changes from the original code
    def load_model_params(self, file_path: str) -> bool:
        """Load generator parameters from a saved model.
        """
        if not os.path.exists(file_path):
            print_and_log("ERROR: Generator model file not found\n", CFG.log_path)
            return False
        try:
            with open(file_path, "rb") as f:
                saved = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as e:
            print_and_log(f"ERROR: Could not load generator model: {e}\n", CFG.log_path)
            return False

        # Compatibility checks
        cant_load = False
        if saved.target_size != self.target_size:
            print_and_log("ERROR: target size mismatch.\n", CFG.log_path);  cant_load = True
        if saved.target_hamiltonian != self.target_hamiltonian:
            print_and_log("ERROR: target hamiltonian mismatch.\n", CFG.log_path);  cant_load = True
        if saved.ansatz != self.ansatz:
            print_and_log("ERROR: ansatz mismatch.\n", CFG.log_path);  cant_load = True
        if saved.layers != self.layers:
            print_and_log("ERROR: layer count mismatch.\n", CFG.log_path);  cant_load = True
        if saved.ancilla and self.ancilla and getattr(saved, 'ancilla_topology', None) != self.ancilla_topology:
            print_and_log("ERROR: ancilla topology mismatch.\n", CFG.log_path);  cant_load = True
        if cant_load:
            return False

        # Detect format
        if hasattr(saved, 'params') and isinstance(saved.params, np.ndarray):
            return self._load_from_new_format(saved)
        elif hasattr(saved, 'qc'):
            return self._load_from_old_format(saved)
        else:
            print_and_log("ERROR: Unrecognised saved generator format.\n", CFG.log_path)
            return False

    def _load_from_new_format(self, saved: "Generator") -> bool:
        if saved.size == self.size and saved.ancilla == self.ancilla:
            if len(saved.params) != self.n_params:
                print_and_log("ERROR: param count mismatch.\n", CFG.log_path)
                return False
            self.params = saved.params.copy()
            self.total_gen_state = self.get_total_gen_state()
            print_and_log("Generator parameters loaded (new format).\n", CFG.log_path)
            return True

        if saved.ancilla != self.ancilla and abs(saved.size - self.size) == 1:
            self._partial_load_params(saved.params, saved)
            self.total_gen_state = self.get_total_gen_state()
            print_and_log("Generator parameters partially loaded (ancilla diff).\n", CFG.log_path)
            return True

        print_and_log("ERROR: incompatible generator.\n", CFG.log_path)
        return False

    def _load_from_old_format(self, saved) -> bool:
        old_angles = np.array([g.angle for g in saved.qc.gates])

        if saved.size == self.size and saved.ancilla == self.ancilla:
            if len(old_angles) != self.n_params:
                print_and_log("ERROR: gate count mismatch.\n", CFG.log_path)
                return False
            self.params = old_angles.copy()
            self.total_gen_state = self.get_total_gen_state()
            print_and_log("Generator parameters loaded (old format → new).\n", CFG.log_path)
            return True

        if saved.ancilla != self.ancilla and abs(saved.size - self.size) == 1:
            saved_proxy = type('Proxy', (), {
                'params': old_angles, 'ancilla': saved.ancilla, 'size': saved.size,
            })()
            self._partial_load_params(old_angles, saved_proxy)
            self.total_gen_state = self.get_total_gen_state()
            print_and_log("Generator parameters partially loaded (old format, ancilla diff).\n", CFG.log_path)
            return True

        print_and_log("ERROR: incompatible old-format generator.\n", CFG.log_path)
        return False

    def _partial_load_params(self, saved_params: np.ndarray, saved) -> None:
        """Load system-only params when ancilla is added/removed."""
        my_ancilla_idx = set(self._get_ancilla_param_indices()) if self.ancilla else set()

        saved_ancilla_idx = set()
        if saved.ancilla:
            orig_extra = CFG.extra_ancilla
            orig_topo = CFG.ancilla_topology
            orig_1q = CFG.do_ancilla_1q_gates
            CFG.extra_ancilla = saved.ancilla
            CFG.ancilla_topology = getattr(saved, 'ancilla_topology', CFG.ancilla_topology)
            CFG.do_ancilla_1q_gates = getattr(saved, 'do_ancilla_1q_gates', CFG.do_ancilla_1q_gates)
            temp_gen = Generator.__new__(Generator)
            temp_gen.ansatz = self.ansatz
            saved_ancilla_idx = set(temp_gen._get_ancilla_param_indices())
            CFG.extra_ancilla = orig_extra
            CFG.ancilla_topology = orig_topo
            CFG.do_ancilla_1q_gates = orig_1q

        saved_system = [p for i, p in enumerate(saved_params) if i not in saved_ancilla_idx]
        my_system_idx = [i for i in range(self.n_params) if i not in my_ancilla_idx]

        n_copy = min(len(saved_system), len(my_system_idx))
        for j in range(n_copy):
            self.params[my_system_idx[j]] = saved_system[j]