"""Cost and Fidelity Functions — PennyLane version.

Mathematical functions used to evaluate the QGAN:
    1. braket()  — Inner product <bra|O_1·O_2·...·O_n|ket>
    2. compute_cost()  — Wasserstein distance (for the learning)
    3. compute_fidelity()  — State overlap (for us)
    4. compute_fidelity_and_cost()  — Convenience wrapper for both

Since the generator states and discriminator matrices deliver numpy arrays, 
this module works identically with the original code and the PennyLane rewrite 
(no changes needed)
"""

import numpy as np

from config import CFG

# Seed numpy's RNG for reproducibility across runs.
np.random.seed()


def braket(*args) -> float:
    """Calculate the braket (inner product) between two quantum states.

    It computes:
        <bra| O_1 · O_2 · ... · O_n |ket>

    Used by:
        - compute_cost() for Wasserstein distance terms
        - compute_fidelity() for state overlap
        - generator._grad_theta() for gradient computation
        - discriminator._grad_alpha/beta() for gradient computation

    Args:
        args: The arguments can be either two vectors (bra and ket),
              three (bra, operator, ket) or bigger (bra, operator^N, ket).

    Returns:
        float: The inner product of the two vectors.
    """
    # Unpack: first arg is bra, last is ket, everything in between is operators
    bra, *ops, ket = args

    # Apply each operator to ket from left to right: |ket> -> O₁|ket> -> O₂·O₁|ket> -> ...
    for op in ops:
        ket = np.matmul(op, ket)

    # Finally compute < bra|result>
    return np.matmul(bra.getH(), ket)


def compute_cost(dis, final_target_state: np.ndarray, final_gen_state: np.ndarray) -> float:
    """Calculate the cost function (Wasserstein distance)

    The Wasserstein cost has three parts:
        Cost = \psi_term - \phi_term - reg_term

    Where:
        \psi_term = <target|\psi|target>     (real part of discriminator on target)
        \phi_term = <gen|\phi|gen>            (imaginary part of discriminator on generated)
        reg_term = regularisation        (enforces Lipschitz constraint)

    The discriminator MAXIMISES and the generator MINIMISES it.

    Args:
        dis (Discriminator): the discriminator.
        final_target_state (np.ndarray): the target state to input into the Discriminator.
        final_gen_state (np.ndarray): the gen state to input into the Discriminator.

    Returns:
        float: the cost function.
    """
    # Get the discriminator's current matrix representations:
    #   A = exp(-\phi/ \lambda),  B = exp(\psi/ \lambda),  \psi (real part),  \phi (imaginary part)
    A, B, psi, phi = dis.get_dis_matrices_rep() # defined in discriminator.py 

    # -- \psi term: how the real part of the discriminator scores the TARGET state
    psiterm = np.ndarray.item(braket(final_target_state, psi, final_target_state))

    # -- \phi term: how the imaginary part of the discriminator scores the GENERATED state 
    phiterm = np.ndarray.item(braket(final_gen_state, phi, final_gen_state))

    # -- Regularisation term: enforces Lipschitz continuity of the discriminator
    # Without this, the discriminator could grow unboundedly.
    # fmt: off
    term1 = braket(final_gen_state, A, final_gen_state) * braket(final_target_state, B, final_target_state)
    term2 = braket(final_target_state, A, final_gen_state) * braket(final_gen_state, B, final_target_state)
    term3 = braket(final_gen_state, A, final_target_state) * braket(final_target_state, B, final_gen_state)
    term4 = braket(final_target_state, A, final_target_state) * braket(final_gen_state, B, final_gen_state)
    regterm = np.ndarray.item(CFG.lamb / np.e * (CFG.cst1 * term1 - CFG.cst2 * (term2 + term3) + CFG.cst3 * term4))
    # fmt: on

    # -- Final cost: \psi_term - \phi_term - reg_term 
    # Take the real part to discard any tiny imaginary residuals from numerical noise
    loss = np.real(psiterm - phiterm - regterm)
    return loss


def compute_fidelity(final_target_state: np.ndarray, final_gen_state: np.ndarray) -> float:
    """Calculate the fidelity between target state and gen state.

    Fidelity = |<target|gen>|^2 (for pure-state), 

    Fidelity = (Tr\sqrt{(\sqrt{\rho}·\sigma·\sqrt{\rho})})^2, (mixed states)
    
    Args:
        final_target_state (np.ndarray): The final target state of the system.
        final_gen_state (np.ndarray): The final gen state of the system.

    Returns:
        float: the fidelity between the target state and the gen state.
    """
    braket_result = braket(final_target_state, final_gen_state)
    return np.abs(np.ndarray.item(braket_result)) ** 2
    # return np.abs(np.asscalar(np.matmul(target_state.getH(), total_final_state))) ** 2


def compute_fidelity_and_cost(dis, final_target_state: np.ndarray, final_gen_state: np.ndarray) -> tuple[float, float]:
    """Calculate the fidelity and cost function.

    Convenience wrapper that computes both metrics in one call.
    Used in the training loop every CFG.save_fid_and_loss_every_x_iter iterations
    to track progress without duplicating code.

    Args:
        dis (Discriminator): the discriminator.
        final_target_state (np.ndarray): the target state.
        final_gen_state (np.ndarray): the gen state.

    Returns:
        tuple[float, float]: the fidelity and cost function.
    """
    fidelity = compute_fidelity(final_target_state, final_gen_state)
    cost = compute_cost(dis, final_target_state, final_gen_state)
    return fidelity, cost