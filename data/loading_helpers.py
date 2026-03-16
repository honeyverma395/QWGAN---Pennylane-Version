"""Model loading and warm-start helpers — PennyLane version.

Changes from original:
    - Warm start functions operate on gen.params (flat numpy array)
      instead of gen.qc.gates[i].angle.
    - No dependency on QuantumCircuit or QuantumGate.
    - Everything else (load logic, logging) unchanged.

This module is called by training.py at the start of each training run. 
If CFG.load_timestamp is set, it attempts to load a previously saved 
generator and discriminator from disk, and optionally perturbs the 
generator parameters (warm start) to escape local minima.

The warm start is used in the "common initial plateaus" experiment mode:
after finding a plateau, we reload those parameters and try again with 
small perturbations to see if we can escape the plateau.
"""

import math
import os

import numpy as np

from config import CFG
from qgan.generator import Generator
from data.data_managers import print_and_log


def load_models_if_specified(training_instance):
    """Load generator and discriminator parameters if a load_timestamp is provided.

    Modifies training_instance.gen and training_instance.dis
    by calling their load_model_params methods.

    Args:
        training_instance (Training): The training instance containing gen and dis.
    """
    # -- Skip if no timestamp -------------------------
    # If load_timestamp is None, this is a fresh training run, nothing to load.
    if not CFG.load_timestamp:
        print_and_log(
            "\nStarting training from scratch (no timestamp specified).\n",
            CFG.log_path,
        )
        print_and_log("==================================================\n", CFG.log_path)
        return

    # -- Build load paths -------------------------
    # Construct the paths where the previous models were saved.
    # The structure is: generated_data/<timestamp>/saved_model/<model_filename>
    # We reuse the same filenames (model-gen(hs).pkl, model-dis(hs).pkl)
    # but swap the base directory to the load_timestamp's directory.
    print_and_log(
        f"\nAttempting to load model parameters [{CFG.load_timestamp}].\n",
        CFG.log_path,
    )
    gen_model_filename = os.path.basename(CFG.model_gen_path)  # e.g. "model-gen(hs).pkl"
    dis_model_filename = os.path.basename(CFG.model_dis_path)  # e.g. "model-dis(hs).pkl"
    load_gen_path = os.path.join(
        "generated_data", CFG.load_timestamp, "saved_model", gen_model_filename
    )
    load_dis_path = os.path.join(
        "generated_data", CFG.load_timestamp, "saved_model", dis_model_filename
    )

    # -- Load generator -------------------------
    print_and_log(
        f"Attempting to load Generator parameters from: {load_gen_path}\n",
        CFG.log_path,
    )
    gen_loaded = training_instance.gen.load_model_params(load_gen_path)

    # -- Load discriminator -------------------------
    # Optimizer momentum is NOT loaded, it resets on load.
    # This is intentional: we want fresh momentum for the new run.
    print_and_log(
        f"\nAttempting to load Discriminator parameters from: {load_dis_path}\n",
        CFG.log_path,
    )
    dis_loaded = training_instance.dis.load_model_params(load_dis_path)

    # -- Final check -------------------------
    # Both must load successfully. If either fails, we abort rather than
    # training with a mismatched gen/dis pair
    if gen_loaded and dis_loaded:
        # Optionally perturb the loaded generator parameters (warm start).
        if CFG.type_of_warm_start != "none":
            apply_warm_start(training_instance)
        print_and_log(
            "Model parameter loading complete. Continuing training.\n",
            CFG.log_path,
        )
        print_and_log("==================================================\n", CFG.log_path)
    else:
        raise ValueError(
            "Incompatible or missing model parameters. "
            "Check the load paths or model compatibility."
        )


def perturb_all_gen_params_X_percent(gen: Generator):
    """Randomly perturb ALL generator parameters by a small amount.

    Each angle is shifted by a uniform random value in
    [-strength * 2\pi, +strength * 2\pi], then wrapped to [0, 2 \pi).

    Used for warm_start type "all".

    For example, with warm_start_strength = 0.1:
      - perturbation range = [-0.2\pi, +0.2\pi] ≈ [-0.628, +0.628] radians
      - Each parameter gets a small random nudge
      - The modulo ensures angles stay in [0, 2\pi)

    This is useful for exploring the neighbourhood of a plateau:
    all parameters move a little, potentially escaping the local minimum.

    Args:
        gen: Generator instance with a .params numpy array.
    """
    # Scale the perturbation
    perturbation_strength = CFG.warm_start_strength * 2 * math.pi

    # Generate one random noise value per parameter
    noise = np.random.uniform(
        -perturbation_strength, perturbation_strength, size=gen.params.shape
    )

    # Apply perturbation and wrap to [0, 2\pi)
    gen.params = (gen.params + noise) % (2 * math.pi)

    # Refresh the cached state after perturbation
    gen.total_gen_state = gen.get_total_gen_state()


def restart_X_percent_of_gen_params_randomly(gen: Generator):
    """Randomly reset a PERCENTAGE of generator parameters to new random values.

    The percentage is determined by CFG.warm_start_strength (0 to 1).
    Selected parameters are replaced with uniform random values in [0, 2 \pi).

    Used for warm_start type "some".

    For example, with warm_start_strength = 0.3 and 30 total params:
      - 30 * 0.3 = 9 parameters are randomly selected
      - Those 9 get completely new random values
      - The other 21 keep their loaded values

    This is more aggressive than "all": it completely resets a fraction. 
    Useful for breaking out of deep plateaus where small perturbations aren't enough.

    Args:
        gen: Generator instance with a .params numpy array.
    """
    num_params = len(gen.params)

    if CFG.warm_start_strength > 0.0:
        # How many parameters to reset (rounded up)
        num_perturb = math.ceil(num_params * CFG.warm_start_strength)

        # Pick random indices WITHOUT replacement (each param reset at most once)
        indices = np.random.choice(num_params, size=num_perturb, replace=False)

        # Replace selected params with fresh random values in [0, 2 \pi)
        gen.params[indices] = np.random.uniform(0, 2 * np.pi, size=num_perturb)

        # Refresh the cached state after perturbation
        gen.total_gen_state = gen.get_total_gen_state()


def apply_warm_start(training_instance):
    """Apply warm start to the generator if specified in configuration.

    Args:
        training_instance (Training): The training instance containing the generator.
    """
    print_and_log(
        "Warm start enabled. Randomly perturbing generator parameters.\n",
        CFG.log_path,
    )
    if CFG.type_of_warm_start == "all":
        perturb_all_gen_params_X_percent(training_instance.gen)
    elif CFG.type_of_warm_start == "some":
        restart_X_percent_of_gen_params_randomly(training_instance.gen)
    else:
        raise ValueError(f"Unknown type of warm start: {CFG.type_of_warm_start}")