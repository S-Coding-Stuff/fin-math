"""Development of MC Finite Difference Sensitivities."""
import numpy as np
from scipy.stats import norm
from engines.monte_carlo import MonteCarloPricing
from engines.qmc import QuasiMonteCarloPricing  # Assume this is a predefined module

class MCFDSensitivity:
    def __init__(self, model, fd_step=1e-4):
        self.model = model
        self.fd_step = fd_step

    def compute_sensitivities(self, params, num_paths=10000):
        base_price = self.model.price(params, num_paths)
        sensitivities = {}

        for param_name, param_value in params.items():
            bumped_params_up = params.copy()
            bumped_params_down = params.copy()

            bumped_params_up[param_name] += self.fd_step
            bumped_params_down[param_name] -= self.fd_step

            price_up = self.model.price(bumped_params_up, num_paths)
            price_down = self.model.price(bumped_params_down, num_paths)

            sensitivity = (price_up - price_down) / (2 * self.fd_step)
            sensitivities[param_name] = sensitivity

        return sensitivities