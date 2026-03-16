"""Ancilla post-processing tools — PennyLane rewrite.

Public interface unchanged:
    get_max_entangled_state_with_ancilla_if_needed(size) -> (gen_state, target_state)
    project_ancilla_zero(state, renormalize)              -> (projected_state, prob)
    trace_out_ancilla(state)                              -> sampled_state
    get_final_gen_state_for_discriminator(state)           -> final_state

All functions return np.matrix column vectors for compatibility with
the discriminator and braket() which expect that format.
"""

import numpy as np
import pennylane as qml

from config import CFG


# -- MAXIMALLY ENTANGLED STATE PREPARATION ---------------------------------
def get_max_entangled_state_with_ancilla_if_needed(size: int):
    """Get the maximally entangled state for the system size, with ancilla if needed.

    Prepares |\Phi^+ > = (1/\sqrt{d}) \sum_i |i> |i>  on 2*size qubits,
    optionally tensored with |0> for the ancilla.

    Args:
        size (int): the system size (number of qubits per register).

    Returns:
        tuple[np.matrix, np.matrix]: (initial_state_for_gen, initial_state_for_target),
            both as column vectors. (tuple useful when we want to share data but not allow to change it)
    """
    # Build ||\Phi^+ >  on 2*size qubits using a circuit: H + CNOT on each pair
    n_choi = 2 * size
    dev = qml.device("default.qubit", wires=n_choi)

    @qml.qnode(dev, interface="numpy")
    def bell_state_circuit():
        for i in range(size):
            qml.Hadamard(wires=i)
            qml.CNOT(wires=[i, i + size])
        return qml.state()

    state = np.array(bell_state_circuit(), dtype=complex)

    # Add ancilla qubit |0> at the end if needed
    ancilla_zero = np.array([1, 0], dtype=complex)
    state_with_ancilla = np.kron(state, ancilla_zero)

    # Different conditions for gen and target:
    initial_state_for_gen = state_with_ancilla if CFG.extra_ancilla else state
    initial_state_for_target = (
        state_with_ancilla if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else state
    )

    return (
        np.asmatrix(initial_state_for_gen).T,
        np.asmatrix(initial_state_for_target).T,
    )


 
# -- PROJECT ANCILLA TO |0> ---------------------------------
def project_ancilla_zero(state: np.ndarray, renormalize: bool = True) -> tuple[np.ndarray, float]:
    """Project the last qubit onto |0> and optionally renormalize.

    Keeps only the amplitudes where the ancilla is in |0> (even indices),
    effectively applying < 0|_ancilla to the state.

    Args:
        state (np.ndarray): The quantum state vector (column vector).
        renormalize (bool): Whether to renormalize the projected state.

    Returns:
        np.matrix: The projected state vector with the ancilla qubit removed.
        float: The probability of the ancilla being in state |0>
    """
    state = np.asarray(state).flatten()

    # Keep only even indices (ancilla = |0>, e.g. |000>, |001>, |010>, |011>...)
    projected = state[::2]

    # Compute the norm (probability of ancilla being |0>)
    norm = np.linalg.norm(projected)

    if norm == 0:
        n_system_qubits = CFG.system_size * 2  # choi + system register
        return np.asmatrix(np.zeros((2**n_system_qubits, 1), dtype=complex)), 0.0

    # Renormalize according to config
    if renormalize:
        if CFG.ancilla_project_norm == "re-norm":
            projected = projected / norm
        elif CFG.ancilla_project_norm != "pass":
            raise ValueError(f"Unknown ancilla_project_norm: {CFG.ancilla_project_norm}")

    return np.asmatrix(projected.reshape(-1, 1)), norm**2


# -- TRACE OUT ANCILLA ---------------------------------
def trace_out_ancilla(state: np.ndarray) -> np.ndarray:
    """Trace out the last qubit and return a sampled pure state. 
    (Ask Guille if is legal or we can do it better, do it without the stochasticity)

    Computes the reduced density matrix by tracing out the ancilla,
    then samples a pure state from the eigenbasis weighted by eigenvalues.

    Args:
        state (np.ndarray): The quantum state vector (column vector).

    Returns:
        np.matrix: A sampled pure state after tracing out the ancilla.
    """
    state = np.asarray(state).flatten()
    n_total = len(state)
    n_qubits = int(np.log2(n_total))
    n_system = n_qubits - 1  # all qubits except ancilla

    # Build full density matrix |\Phi >< \Phi|
    rho_full = np.outer(state, state.conj())

    # PennyLane to compute the reduced density matrix
    # by tracing out the last qubit (ancilla)
    rho_reduced = qml.math.reduce_dm(rho_full, indices=[n_system])

    # Sample a pure state from the reduced density matrix
    # (eigenvector weighted by eigenvalue probability)
    eigvals, eigvecs = np.linalg.eigh(rho_reduced)
    eigvals = np.maximum(eigvals, 0)  # clip numerical negatives
    eigvals = eigvals / np.sum(eigvals)  # renormalize

    idx = np.random.choice(len(eigvals), p=eigvals)
    sampled_state = eigvecs[:, idx]

    return np.asmatrix(sampled_state.reshape(-1, 1))


# -- ROUTER: GEN STATE -> DISCRIMINATOR ---------------------------------
def get_final_gen_state_for_discriminator(total_output_state: np.ndarray) -> np.ndarray:
    """Process the generator output state according to ancilla_mode.

    Routes to the appropriate post-processing:
        - "pass":    state goes directly to discriminator (ancilla included)
        - "project": project ancilla onto |0>, remove it
        - "trace":   trace out ancilla, sample pure state

    Args:
        total_output_state (np.ndarray): The output state from the generator.

    Returns:
        np.ndarray: The final state to be passed to the discriminator.
    """
    if not CFG.extra_ancilla:
        return total_output_state

    if CFG.ancilla_mode == "pass":
        return total_output_state
    if CFG.ancilla_mode == "project":
        projected, _ = project_ancilla_zero(total_output_state)
        return projected
    if CFG.ancilla_mode == "trace":
        return trace_out_ancilla(total_output_state)

    raise ValueError(f"Unknown ancilla_mode: {CFG.ancilla_mode}")