"""Training module for the Quantum GAN — PennyLane version.

The training loop logic is unchanged from the original.
What changes:
    - Generator uses PennyLane QNodes and autograd (see generator.py).
    - Discriminator keeps its classical numpy/scipy internals but
      replaces MomentumOptimizer with an inline momentum update
      (since optimizer.py and data_reshapers.py are eliminated).
    - save_gen_final_params adapted for flat param array (no need of qc.gates).
"""

from datetime import datetime

import numpy as np

from config import CFG
from qgan.ancilla import (
    get_final_gen_state_for_discriminator,
    get_max_entangled_state_with_ancilla_if_needed,
)
from qgan.cost_functions import compute_fidelity_and_cost
from qgan.discriminator import Discriminator
from qgan.generator import Generator
from qgan.target import get_final_target_state
from data.data_managers import (
    print_and_log,
    save_fidelity_loss,
    save_model,
    save_gen_final_params,
)
from data.loading_helpers import load_models_if_specified
from plot_hub import plt_fidelity_vs_iter

np.random.seed()

class Training:
    def __init__(self):
        """Builds the training configuration.

        Prepares:
            1. Maximally entangled state (Choi) with ancilla if needed.
            2. Target state from the target Hamiltonian.
            3. Generator (PennyLane based).
            4. Discriminator (numpy/scipy).
        """
        # Prepare maximally entangled state (+ ancilla if needed)
        initial_state_total, initial_state_final = (
            get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
        )

        # Target state: (I \otimes U_target) |\Phi^+>
        self.final_target_state: np.matrix = get_final_target_state(initial_state_final)

        # Generator: variational quantum circuit (PennyLane)
        self.gen: Generator = Generator()

        # Discriminator: classical Hermitian operator
        self.dis: Discriminator = Discriminator()

    def run(self):
        """Run the training loop.

        For each iteration:
            1. Process ancilla on generator output.
            2. Update discriminator (maximise Wasserstein distance).
            3. Update generator (minimise Wasserstein distance).
            4. Log fidelity and loss periodically.

        Stops when max_fidelity is reached or max epochs is reached
        Saves models, fidelity history, and generator parameters at the end.
        """
        # -- Initialise training -----------------------------------
        print_and_log("\n" + CFG.show_data(), CFG.log_path)

        # Load models if a previous checkpoint is specified
        load_models_if_specified(self)

        fidelities_history, losses_history = [], []
        starttime = datetime.now()
        num_epochs: int = 0

        # -- Main training loop -----------------------------------
        while True:
            fidelities = []
            losses = []
            num_epochs += 1

            for epoch_iter in range(CFG.iterations_epoch):
                # --- Discriminator and Generator gradient descent
                # Process ancilla before passing to discriminator
                final_gen_state = get_final_gen_state_for_discriminator(
                    self.gen.total_gen_state
                )

                # -- Discriminator step(s): MAXIMISE Wasserstein distance
                for _ in range(CFG.steps_dis):
                    self.dis.update_dis(self.final_target_state, final_gen_state)

                # -- Generator step(s): MINIMISE Wasserstein distance
                for _ in range(CFG.steps_gen):
                    self.gen.update_gen(self.dis, self.final_target_state)

                # --- Periodically compute and save fidelity & loss
                if epoch_iter % CFG.save_fid_and_loss_every_x_iter == 0:
                    fid, loss = compute_fidelity_and_cost(
                        self.dis, self.final_target_state, final_gen_state
                    )
                    fidelities.append(fid)
                    losses.append(loss)

                # -- Periodically log -----------------------------------
                if epoch_iter % CFG.log_every_x_iter == 0:
                    info = (
                        f"\nepoch:{num_epochs:4d} | "
                        f"iters:{epoch_iter + 1:4d} | "
                        f"fidelity:{round(fid, 6):8f} | "
                        f"loss:{round(loss, 6):8f}"
                    )
                    print_and_log(info, CFG.log_path)

            # -- End of epoch: store history and plot -----------------------------------
            fidelities_history = np.append(fidelities_history, fidelities)
            losses_history = np.append(losses_history, losses)
            plt_fidelity_vs_iter(fidelities_history, losses_history, CFG, num_epochs)

            #-- Stopping conditions -----------------------------------
            if num_epochs >= CFG.epochs:
                print_and_log(
                    "\n==================================================\n",
                    CFG.log_path,
                )
                print_and_log(
                    f"\nThe number of epochs exceeds {CFG.epochs}.",
                    CFG.log_path,
                )
                break

            if fidelities[-1] > CFG.max_fidelity:
                print_and_log(
                    "\n==================================================\n",
                    CFG.log_path,
                )
                print_and_log(
                    f"\nThe fidelity {fidelities[-1]} exceeds the maximum {CFG.max_fidelity}.",
                    CFG.log_path,
                )
                break

        # -- End of training: save everything -----------------------------------
        # Fidelity and loss history
        save_fidelity_loss(fidelities_history, losses_history, CFG.fid_loss_path)

        # Generator and discriminator models (pickle)
        save_model(self.gen, CFG.model_gen_path)
        save_model(self.dis, CFG.model_dis_path)

        # Generator final parameters (plain text)
        save_gen_final_params(self.gen, CFG.gen_final_params_path)

        endtime = datetime.now()
        print_and_log(f"\nRun took: {endtime - starttime} time.", CFG.log_path)