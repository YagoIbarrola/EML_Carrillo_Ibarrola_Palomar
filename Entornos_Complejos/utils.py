import numpy as np
from matplotlib import pyplot as plt


def get_moving_avgs(arr, window: int, convolution_mode: str = "valid"):
    """
    Compute moving average using convolution.

    Args:
        arr: Sequence of numeric values
        window: Window size
        convolution_mode: 'valid', 'same', or 'full'

    Returns:
        Smoothed numpy array
    """
    return (
        np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode,
        )
        / window
    )


def plot_training_metrics(
    rewards,
    lengths,
    training_errors,
    rolling_length: int = 500,
):
    """
    Plot smoothed training statistics.

    Args:
        rewards: Episode reward history
        lengths: Episode length history
        training_errors: TD error history
        rolling_length: Moving average window
    """

    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))


    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        rewards,
        rolling_length,
        "valid",
    )
    axs[0].plot(reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        lengths,
        rolling_length,
        "valid",
    )
    axs[1].plot(length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Delta_Q")
    training_error_moving_average = get_moving_avgs(
        training_errors,
        rolling_length,
        "same",
    )
    axs[2].plot(training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    for i, ax in enumerate(axs):
        ax.tick_params(axis='x', labelrotation=30)

    plt.tight_layout()
    plt.show()