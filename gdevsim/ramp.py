"""Adapted from DEVSIM LLC:

Improvements:
    - clump contacts together
    - more step size adaptivity
"""
import devsim as ds
from dataclasses import dataclass
from typing import List
from pathlib import Path

from gdevsim.models.models import get_contact_bias_name


@dataclass
class RampParameters:
    contact_name: str
    biases: List[float]
    initial_step_size:  float | None = None
    maximum_step_size:  float | None = None
    step_size_scaling_down: float = 4
    step_size_scaling_up: float = 2
    min_step: float = 0.01
    max_iter: int = 20
    rel_error: float = 1E-10
    abs_error: float = 1E10
    save_intermediate_structures_root: str | None = "ramp"
    save_directory: Path | None = None
    intermediate_structures: List[Path] | None = None
    
    def get_intermediate_structures_filenames(self) -> List[str]:
        """Return list of filenames for intermediate structures"""
        return [f"{self.save_intermediate_structures_root}_{self.contact_name}_{bias}.ds" for bias in self.biases]
    
    def get_intermediate_structures_filepaths(self) -> List[Path]:
        """Return list of filepaths for intermediate structures"""
        return self.intermediate_structures
    
    def add_intermediate_structures_filepath(self, path):
        self.intermediate_structures.append(path)

    def clear_intermediate_structures_filepath(self):
        self.intermediate_structures = []
    

def rampbias(
    device,
    contacts,
    start_bias,
    end_bias,
    initial_step_size,
    maximum_step_size,
    step_size_scaling_down,
    step_size_scaling_up,
    min_step,
    max_iter,
    rel_error,
    abs_error,
):
    """
    Ramps bias with assignable callback function
    """
    if start_bias < end_bias:
        step_sign = 1
    else:
        step_sign = -1
    step_size = abs(initial_step_size)

    last_bias = start_bias
    while abs(last_bias - end_bias) > min_step:
        print(f"Last end {last_bias:e} {end_bias:e}")
        next_bias = last_bias + step_sign * step_size
        if next_bias < end_bias:
            next_step_sign = 1
        else:
            next_step_sign = -1

        if next_step_sign != step_sign:
            next_bias = end_bias
            print("setting to last bias %e" % (end_bias))
            print("setting next bias %e" % (next_bias))
        for contact in contacts:
            ds.set_parameter(
                device=device, name=contact, value=next_bias
            )
        try:
            ds.solve(
                type="dc",
                absolute_error=abs_error,
                relative_error=rel_error,
                maximum_iterations=max_iter,
            )
        except ds.error as msg:
            if str(msg).find("Convergence failure") != 0:
                raise
            for contact in contacts:
                ds.set_parameter(
                    device=device, name=get_contact_bias_name(contact), value=last_bias
                )
            step_size /= step_size_scaling_down
            print("setting new step size %e" % (step_size))
            if step_size < min_step:
                raise RuntimeError("Minimum step size too small")  # noqa: B904
            continue
        # If no convergence issue, increase the ramp rate again if it has been lowered:
        else:
            if step_size < maximum_step_size:
                step_size *= step_size_scaling_up
                step_size = min(step_size, maximum_step_size)
                print("setting new step size %e" % (step_size))

        print("Succeeded")
        last_bias = next_bias