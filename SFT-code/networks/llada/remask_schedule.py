import numpy as np


def uniform_schedule(gen_length, num_steps):
    """
    Generate a uniform schedule for the given generation length and number of steps.
    
    Args:
        gen_length (int): The total length of the generation.
        num_steps (int): The number of steps in the schedule.
        
    Returns:
        np.ndarray: An array of shape (num_steps,) with uniformly spaced values.
    """
    return np.linspace(gen_length // num_steps, gen_length, num_steps, dtype=int)


def log_schedule(gen_length, num_steps):
    """
    Generate a logarithmic schedule for the given generation length and number of steps.
    
    Args:
        gen_length (int): The total length of the generation.
        num_steps (int): The number of steps in the schedule.
        
    Returns:
        np.ndarray: An array of shape (num_steps,) with logarithmically spaced values.
    """
    step = np.arange(num_steps)
    gen = np.log(step + 2) * 256 / np.log(num_steps + 1)
    return np.clip(gen, 0, gen_length).astype(int)


if __name__ == "__main__":
    print("Log Schedule Example:")
    print(log_schedule(256, 64))
