"""Discriminator module — PennyLane-compatible version.

The discriminator is NOT a quantum circuit, it is a classical parametrisation
of two Hermitian operators (\psi, \phi) built from tensor products of Pauli matrices.
That's why it stays as pure numpy/scipy linear algebra.

Changes from original:
    - Removed dependency on MomentumOptimizer (optimizer.py eliminated).
      Momentum update is now in this file.
    - Removed dependency on tools.qobjects.qgates (qgates.py eliminated).
      Pauli matrices defined locally.
    - Everything else (gradient logic, load/save) unchanged, just adapted to Pennylane.
"""

import os
import pickle
from copy import deepcopy

import numpy as np
from scipy.linalg import expm

from config import CFG
from qgan.cost_functions import braket
from data.data_managers import print_and_log


# -- PAULI MATRICES ---------------------------------
# The four single-qubit Pauli matrices form a basis for all 2×2 Hermitian
# operators. Any Hermitian operator on one qubit can be written as a linear
# combination:  H = a·I + b·X + c·Y + d·Z  with real coefficients a, b, c, d.
# For multi-qubit systems, we take tensor (Kronecker) products of these.
I = np.eye(2, dtype=complex)      
X = np.array([[0, 1], [1, 0]], dtype=complex)   
Y = np.array([[0, -1j], [1j, 0]], dtype=complex) 
Z = np.array([[1, 0], [0, -1]], dtype=complex)    


# -- MOMENTUM OPTIMIZER  ---------------------------------
class _MomentumState:
    """Tracks velocity for SGD with momentum on a single parameter array.

    Standard momentum SGD update rule:
        v_{t+1} = \mu·v_t + sign· \eta · \nabla  
        \theta_{t+1} = \theta_t + v_{t+1}

    where:
        - \eta is the learning rate (step size),
        - \mu is the momentum coefficient (how much of the previous velocity to retain),
        - sign = +1 for maximisation (gradient ascent), -1 for minimisation (gradient descent).

    Momentum helps smooth out noisy gradients and accelerate convergence
    by accumulating a running average of past gradient directions.
    """

    def __init__(self, eta: float = CFG.l_rate, mu: float = CFG.momentum_coeff):
        self.eta = eta   # Learning rate: controls the magnitude of each update step
        self.mu = mu     # Momentum coefficient: fraction of previous velocity retained
        self.v = None    # Velocity tensor

    def step(self, params: np.ndarray, grad: np.ndarray,
             maximise: bool = True) -> np.ndarray:
        """One momentum update step.

        On the first call, the velocity is initialised directly from the
        scaled gradient (no momentum term yet). On subsequent calls, the
        velocity blends the previous velocity with the new gradient.

        Args:
            params: Current parameters (any shape).
            grad: Gradient (same shape as params).
            maximise: If True, ascend (+grad). If False, descend (-grad).

        Returns:
            Updated parameters (same shape).
        """
        sign = 1.0 if maximise else -1.0

        if self.v is None:
            # First step: no previous velocity to blend with
            self.v = sign * self.eta * grad
        else:
            # Subsequent steps: blend old velocity with new gradient
            self.v = self.mu * self.v + sign * self.eta * grad

        return params + self.v


# -- DISCRIMINATOR ---------------------------------
# Wasserstein cost constants loaded from the global config.
# These control the relative weight of different terms in the
# regularisation expression that enforces the Lipschitz constraint.
#   cst1, cst2, cst3: prefactors for the three cross-terms in the penalty
#   lamb (\lambda): regularisation strength / temperature parameter
cst1, cst2, cst3, lamb = CFG.cst1, CFG.cst2, CFG.cst3, CFG.lamb

class Discriminator:
    """Discriminator class for the Quantum Wasserstein GAN.

    The discriminator is parametrised by two sets of real coefficients
    (\alpha, \beta), each of shape (N_qubits, 4), where 4 corresponds
    to the Pauli basis {I, X, Y, Z}.

    From these coefficients two full Hermitian operators are built:

    Representation with coeff. in front of each Hermitian operator (N × 4):
        - alpha: coefficients for the real part (\psi), I, X, Y, Z per qubit.
        - beta:  coefficients for the imaginary part (\phi).

    Matrix representation spanning the full space (2^N × 2^N):
        - \psi = \otimes_i \sum_j \alpha_{ij}· \sigma_j
        - \phi = \otimes_i \sum_j \beta_{ij} · \sigma_j

    The Wasserstein distance is estimated from expectation values of \psi
    and \phi on the target and generated states. A regularisation term
    based on matrix exponentials enforces the Lipschitz constraint:

    For computing the gradients:
        - A = exp(-\phi/ \lambda)
        - B = exp(\psi/ \lambda)
    """

    def __init__(self):
        # The discriminator operates on a composite Hilbert space:
        # Total number of qubits the discriminator acts on:
        self.size: int = (
            CFG.system_size * 2
            + (1 if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else 0)
        )

        # Pauli basis used to decompose per-qubit Hermitian operators
        self.herm: list = [I, X, Y, Z]

        # Randomly initialise the \alpha and \beta coefficient arrays
        self._init_params_alpha_beta()

        # Momentum optimisers for \alpha (-> \psi) and \beta (-> \phi), both MAXIMISE
        # The discriminator wants to maximise the Wasserstein distance estimate,
        # so both parameter sets are updated via gradient ascent.
        self.optimizer_psi = _MomentumState()
        self.optimizer_phi = _MomentumState()

        # Metadata for save/load compatibility checks
        # These allow us to verify that a saved model is compatible
        # with the current configuration before loading its parameters.
        self.ancilla: bool = CFG.extra_ancilla
        self.ancilla_mode: str = CFG.ancilla_mode
        self.target_size: int = CFG.system_size
        self.target_hamiltonian: str = CFG.target_hamiltonian

    def _init_params_alpha_beta(self):
        """Random initialisation of \alpha and \beta in [-1, 1].

        Each row corresponds to one qubit, and each column to a Pauli
        operator (I, X, Y, Z). 
        """
        self.alpha: np.ndarray = -1 + 2 * np.random.random((self.size, 4))
        self.beta: np.ndarray = -1 + 2 * np.random.random((self.size, 4))

    # matrix representations ---------------------------------
    def get_psi_and_phi(self) -> tuple[np.ndarray, np.ndarray]:
        """Build \psi and \phi matrices via Kronecker product of per-qubit Hermitians.

        For each qubit i, a local 2×2 Hermitian is formed:
            H_i = \sum_j coeff[i][j] · \sigma_j

        Then the full multi-qubit operator is the tensor product:
            \psi = H_0 \otimes H_1 \otimes ... \otimes H_{N-1}

        Returns:
            (\psi, \phi) each of shape (2^N, 2^N).
        """
        # Start with scalar 1; successive Kronecker products build up the
        # full 2^N × 2^N matrices incrementally.
        psi, phi = 1, 1
        for i in range(self.size):
            # Build per-qubit operators by summing over the Pauli basis
            psi_i = np.zeros((2, 2), dtype=complex)
            phi_i = np.zeros((2, 2), dtype=complex)
            for j, herm_j in enumerate(self.herm):
                psi_i += self.alpha[i][j] * herm_j
                phi_i += self.beta[i][j] * herm_j
            # Kronecker product extends the operator to the next qubit
            psi = np.kron(psi, psi_i)
            phi = np.kron(phi, phi_i)
        return psi, phi

    def get_dis_matrices_rep(self) -> tuple:
        """Compute A = exp(-\phi/ \lambda) and B = exp(\psi/ \lambda).

        These matrix exponentials appear in the Wasserstein dual formulation's
        regularisation term. They enforce the Lipschitz constraint on the
        discriminator: without regularisation, the discriminator could grow
        \psi and \phi without bound, making the Wasserstein estimate diverge.

        Returns:
            (A, B, \phi, \psi): the exponentials and the raw Hermitian operators.
        """
        psi, phi = self.get_psi_and_phi()
        A = expm(float(-1 / lamb) * phi)  # exp(-\phi / \lambda)
        B = expm(float(1 / lamb) * psi)   # exp(+\psi / \lambda)
        return A, B, psi, phi

    # -- parameter update ---------------------------------
    def update_dis(self, final_target_state: np.ndarray,
                   final_gen_state: np.ndarray):
        """Update \alpha and \beta using momentum SGD (maximisation).

        This is the discriminator's training step. Given the current target
        state (from real data) and the generator's output state, it:
            1. Builds the current A, B matrices from the parameters.
            2. Computes gradients of the Wasserstein cost w.r.t. \alpha and \beta.
            3. Applies momentum SGD to update both parameter sets.

        Gradients are computed for both \alpha and \beta before either
        is updated, to avoid one update affecting the other's gradient.
        This ensures the gradients are evaluated at the same point in parameter space.
        """
        A, B, _, _ = self.get_dis_matrices_rep()

        # Compute gradients at the CURRENT parameter values
        grad_alpha = self._compute_grad(
            final_target_state, final_gen_state, A, B, "alpha"
        )
        grad_beta = self._compute_grad(
            final_target_state, final_gen_state, A, B, "beta"
        )

        # Apply momentum updates (maximise the Wasserstein distance estimate)
        new_alpha = self.optimizer_psi.step(self.alpha, grad_alpha, maximise=True)
        new_beta = self.optimizer_phi.step(self.beta, grad_beta, maximise=True)

        # Update after both gradients are computed
        self.alpha = new_alpha
        self.beta = new_beta

    # -- gradient computation ---------------------------------
    def _compute_grad(self, final_target_state, final_gen_state,
                      A, B, param: str) -> np.ndarray:
        """Gradient of the Wasserstein cost w.r.t. \alpha or \beta.

        The full gradient decomposes into three terms:
            1. grad_psi_term: derivative of Tr[\psi · \rho_target]
               (only non-zero for \alpha, since \psi depends on \alpha)
            2. grad_phi_term: derivative of Tr[\phi · \rho_gen]
               (only non-zero for \beta, since \phi depends on \beta)
            3. grad_reg_term: derivative of the Lipschitz regularisation penalty

        The final gradient is: grad_psi - grad_phi - grad_reg
    
        We iterate over each Hermitian type (I, X, Y, Z) separately because
        the per-qubit structure allows the gradient to be decomposed this way.

        Returns:
            np.ndarray of shape (self.size, 4).
        """
        zero_param = self.alpha if param == "alpha" else self.beta
        grad_psi_term = np.zeros_like(zero_param, dtype=complex)
        grad_phi_term = np.zeros_like(zero_param, dtype=complex)
        grad_reg_term = np.zeros_like(zero_param, dtype=complex)

        # Loop over each Pauli type: 0=I, 1=X, 2=Y, 3=Z
        for herm_type in range(len(self.herm)):
            if param == "alpha":
                gpsi, gphi, greg = self._grad_alpha(
                    final_target_state, final_gen_state, A, B, herm_type
                )
            else:
                gpsi, gphi, greg = self._grad_beta(
                    final_target_state, final_gen_state, A, B, herm_type
                )

            # Each of gpsi, gphi, greg is a list of length self.size
            # (one entry per qubit). Store them in the corresponding column.
            grad_psi_term[:, herm_type] = np.asarray(gpsi)
            grad_phi_term[:, herm_type] = np.asarray(gphi)
            grad_reg_term[:, herm_type] = np.asarray(greg)

        # Take the real part (the imaginary parts should be negligible
        # since \psi, \phi are Hermitian and the states are physical).
        return np.real(grad_psi_term - grad_phi_term - grad_reg_term)

    def _grad_alpha(self, final_target_state, final_gen_state,
                    A, B, herm_type):
        """Gradient step w.r.t. \alpha for a given Hermitian type (I/X/Y/Z).

        Since \psi depends on \alpha (and \phi does not), the \phi-term
        gradient is always zero for \alpha derivatives.

        For the \psi-term: d/d\alpha Tr[\psi · \rho_target] uses the
        chain rule through the Kronecker product structure, for each qubit i,
        the derivative replaces qubit i's linear combination with the bare
        Pauli \sigma_{herm_type}, keeping all other qubits unchanged.

        Args:
            final_target_state: Target (real data) quantum state vector.
            final_gen_state: Generator's output quantum state vector.
            A: exp(-\phi / \lambda).
            B: exp(+\psi / \lambda).
            herm_type: Index into {I, X, Y, Z} (0, 1, 2, 3).

        Returns:
            (gpsi, gphi, greg): Three lists, each of length self.size,
            containing per-qubit gradient contributions.
        """
        cs = 1 / lamb  # Scaling factor
        # Get the list of gradient matrices: one per qubit, where qubit i
        # has its Pauli combination replaced by the bare \sigma_{herm_type}
        grad_psi_list = self._grad_psi_or_phi(herm_type, respect_to="psi")
        gpsi, gphi, greg = [], [], []

        for grad_psi in grad_psi_list:
            # \psi term: <target | d\psi/d\alpha | target>
            gpsi.append(np.ndarray.item(
                braket(final_target_state, grad_psi, final_target_state)
            ))
            # \phi term: no dependence on \alpha, so gradient is zero
            gphi.append(0)

            # Regularisation term: four cross-braket products
            # fmt: off
            t1 = cs * braket(final_gen_state, A, final_gen_state) * braket(final_target_state, grad_psi, B, final_target_state)
            t2 = cs * braket(final_gen_state, grad_psi, B, final_target_state) * braket(final_target_state, A, final_gen_state)
            t3 = cs * braket(final_gen_state, A, final_target_state) * braket(final_target_state, grad_psi, B, final_gen_state)
            t4 = cs * braket(final_gen_state, grad_psi, B, final_gen_state) * braket(final_target_state, A, final_target_state)
            greg.append(np.ndarray.item(lamb / np.e * (cst1 * t1 - cst2 * (t2 + t3) + cst3 * t4)))
            # fmt: on

        return gpsi, gphi, greg

    def _grad_beta(self, final_target_state, final_gen_state,
                   A, B, herm_type):
        """Gradient step w.r.t. \beta for a given Hermitian type (I/X/Y/Z).

        Since \phi depends on \beta (and \psi does not), the \psi-term
        gradient is always zero for \beta derivatives.

        For the \phi-term: d/d\beta Tr[\phi · \rho_gen] follows the same
        Kronecker product chain rule as \alpha, but applied to \phi.

        The regularisation involves the derivative of A = exp(-\phi/\lambda),
        which introduces a factor of cs = -1/\lambda.

        Args:
            final_target_state: Target (real data) quantum state vector.
            final_gen_state: Generator's output quantum state vector.
            A: exp(-\phi / \lambda).
            B: exp(+\psi / \lambda).
            herm_type: Index into {I, X, Y, Z} (0, 1, 2, 3).

        Returns:
            (gpsi, gphi, greg): Three lists, each of length self.size,
            containing per-qubit gradient contributions.
        """
        cs = -1 / lamb  # Negative sign because A = exp(-\phi/\lambda)
        # Gradient matrices for \phi: same structure as \psi but using \beta coefficients
        grad_phi_list = self._grad_psi_or_phi(herm_type, respect_to="phi")
        gpsi, gphi, greg = [], [], []

        for grad_phi in grad_phi_list:
            # \psi term: no dependence on \beta, so gradient is zero
            gpsi.append(0)
            # \phi term: <gen | d\phi/d\beta | gen>
            gphi.append(np.ndarray.item(
                braket(final_gen_state, grad_phi, final_gen_state)
            ))

            # Regularisation term: same structure as \alpha case but with
            # d\phi replacing d\psi, and A replacing B in the brakets.
            # fmt: off
            t1 = cs * braket(final_gen_state, grad_phi, A, final_gen_state) * braket(final_target_state, B, final_target_state)
            t2 = cs * braket(final_gen_state, B, final_target_state) * braket(final_target_state, grad_phi, A, final_gen_state)
            t3 = cs * braket(final_gen_state, grad_phi, A, final_target_state) * braket(final_target_state, B, final_gen_state)
            t4 = cs * braket(final_gen_state, B, final_gen_state) * braket(final_target_state, grad_phi, A, final_target_state)
            greg.append(np.ndarray.item(lamb / np.e * (cst1 * t1 - cst2 * (t2 + t3) + cst3 * t4)))
            # fmt: on

        return gpsi, gphi, greg

    def _grad_psi_or_phi(self, herm_type: int, respect_to: str) -> list:
        """Gradient of \psi (or \phi) 

        Because \psi (or \phi) is built as a tensor product of per-qubit
        operators, the derivative w.r.t. the coefficient \alpha[i][herm_type]
        (or \beta[i][herm_type]) has a simple form: it's the same tensor
        product, but with qubit i's linear combination replaced by the bare
        Pauli matrix \sigma_{herm_type}.

        Args:
            herm_type: Which Pauli to differentiate w.r.t. (0=I, 1=X, 2=Y, 3=Z).
            respect_to: "psi" (use \alpha coefficients) or "phi" (use \beta).

        Returns:
            List of self.size matrices, each of shape (2^N, 2^N).
        """
        coefficients = self.alpha if respect_to == "psi" else self.beta

        grad_matrices = []
        for i in range(self.size):
            # Build the tensor product, treating qubit i specially
            matrix = 1
            for j in range(self.size):
                if i == j:
                    # Derivative qubit: replace with the bare Pauli
                    matrix_j = self.herm[herm_type]
                else:
                    # Non-derivative qubit: keep the full linear combination
                    # H_j = \sum_k coeff[j][k] · \sigma_k
                    matrix_j = np.zeros((2, 2), dtype=complex)
                    for k, herm_k in enumerate(self.herm):
                        matrix_j += coefficients[j][k] * herm_k
                matrix = np.kron(matrix, matrix_j)
            grad_matrices.append(matrix)

        return grad_matrices

    # -- save / load ---------------------------------
    # No big changes
    def load_model_params(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            print_and_log("Discriminator model file not found\n", CFG.log_path)
            return False
        try:
            with open(file_path, "rb") as f:
                saved_dis = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as e:
            print_and_log(
                f"ERROR: Could not load discriminator model: {e}\n", CFG.log_path
            )
            return False

        # Compatibility checks, these must match exactly, otherwise the
        # parameters would correspond to a different physical system.
        cant_load = False
        if saved_dis.target_size != self.target_size:
            print_and_log(
                "ERROR: target size mismatch.\n", CFG.log_path
            )
            cant_load = True
        if saved_dis.target_hamiltonian != self.target_hamiltonian:
            print_and_log(
                "ERROR: target hamiltonian mismatch.\n", CFG.log_path
            )
            cant_load = True
        if cant_load:
            return False

        # Exact match, all parameters can be transferred directly
        if saved_dis.size == self.size:
            self.alpha = deepcopy(saved_dis.alpha)
            self.beta = deepcopy(saved_dis.beta)
            print_and_log("Discriminator parameters loaded.\n", CFG.log_path)
            return True

        # \pm 1 qubit (ancilla difference), partial parameter transfer.
        # This handles the case where the saved model had an ancilla and
        # the current one doesn't, or vice versa.
        if abs(saved_dis.size - self.size) == 1:
            min_size = min(saved_dis.size, self.size)
            self.alpha[:min_size] = saved_dis.alpha[:min_size].copy()
            self.beta[:min_size] = saved_dis.beta[:min_size].copy()
            print_and_log(
                "Discriminator parameters partially loaded (\pm 1 qubit).\n",
                CFG.log_path,
            )
            return True

        # Size difference > 1: architectures are too different to transfer
        print_and_log(
            "ERROR: incompatible discriminator (size mismatch).\n", CFG.log_path
        )
        return False