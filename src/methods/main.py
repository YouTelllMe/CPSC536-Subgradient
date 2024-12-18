from incremental_subgradient import IncrementalSubgradient
from modified_subgradient import ModifiedSubgradient
from subgradient import Subgradient
import numpy as np
from problem import N, X0, MAX_ITERATION
from helpers import Step
import matplotlib.pyplot as plt



if __name__ == "__main__":
    subgradient_const = Subgradient(X0, Step.CONSTANT_SIZE, MAX_ITERATION, 3300)
    subgradient_polyak = Subgradient(X0, Step.POLYAK, MAX_ITERATION, 3300)
    subgradient_dim = Subgradient(X0, Step.SQAURE_SUMMABLE_NON_SUMMABLE, MAX_ITERATION, 3300)

    inc_subgradient_dim = IncrementalSubgradient(X0, Step.SQAURE_SUMMABLE_NON_SUMMABLE, MAX_ITERATION)
    inc_subgradient_const = IncrementalSubgradient(X0, Step.CONSTANT_SIZE, MAX_ITERATION)

    modified_subgradient_const = ModifiedSubgradient(X0, Step.CONSTANT_SIZE, MAX_ITERATION, 3300, 5, 1.5)
    modified_subgradient_polyak = ModifiedSubgradient(X0, Step.POLYAK, MAX_ITERATION, 3300, 5, 1.5)
    modified_subgradient_dim = ModifiedSubgradient(X0, Step.SQAURE_SUMMABLE_NON_SUMMABLE, MAX_ITERATION, 3300, 5, 1.5)


    subgradient_polyak.run()
    subgradient_const.run()
    subgradient_dim.run()
    inc_subgradient_const.run()
    inc_subgradient_dim.run()
    modified_subgradient_polyak.run()
    modified_subgradient_const.run()
    modified_subgradient_dim.run()


    x = range(MAX_ITERATION)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(23, 10))
    colors = ["#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Olive
            "#17becf"]  # Cyan

    ax1.plot(x, subgradient_const.objectives, linewidth = 0.2, marker='.', label='Regular - Const', c=colors[0])
    ax1.plot(x, subgradient_dim.objectives, linewidth = 0.2, marker='.', label='Regular - Diminishing', c=colors[1])
    ax1.plot(x, subgradient_polyak.objectives, linewidth = 0.2, marker='.', label='Regular - Polyak', c=colors[2])
    
    ax2.plot(x, inc_subgradient_const.objectives, linewidth = 0.2, marker='.', label='Incremental - Const', c=colors[3])
    ax2.plot(x, inc_subgradient_dim.objectives, linewidth = 0.2, marker='.', label='Incremental - Diminishing', c=colors[4])

    ax3.plot(x, modified_subgradient_const.objectives, linewidth = 0.2, marker='.', label='Modified - Const', c=colors[5])
    ax3.plot(x, modified_subgradient_dim.objectives, linewidth = 0.2, marker='.', label='Modified - Diminishing', c=colors[6])
    ax3.plot(x, modified_subgradient_polyak.objectives, linewidth = 0.2, marker='.', label='Modified - Polyak', c=colors[7])

    ax4.plot(x, subgradient_const.objectives, linewidth = 0.2, marker='.', label='Regular - Const', c=colors[0])
    ax4.plot(x, subgradient_dim.objectives, linewidth = 0.2, marker='.', label='Regular - Diminishing', c=colors[1])
    ax4.plot(x, subgradient_polyak.objectives, linewidth = 0.2, marker='.', label='Regular - Polyak', c=colors[2])
    ax4.plot(x, inc_subgradient_const.objectives, linewidth = 0.2, marker='.', label='Incremental - Const', c=colors[3])
    ax4.plot(x, inc_subgradient_dim.objectives, linewidth = 0.2, marker='.', label='Incremental - Diminishing', c=colors[4])
    ax4.plot(x, modified_subgradient_const.objectives, linewidth = 0.2, marker='.', label='Modified - Const', c=colors[5])
    ax4.plot(x, modified_subgradient_dim.objectives, linewidth = 0.2, marker='.', label='Modified - Diminishing', c=colors[6])
    ax4.plot(x, modified_subgradient_polyak.objectives, linewidth = 0.2, marker='.', label='Modified - Polyak', c=colors[7])

    # Add grid and legend to each subplot
    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid()
        ax.legend()

    # Set figure-level titles and labels
    fig.suptitle("Task Assignment with Subgradient Methods")
    fig.supxlabel("Iterations k")
    fig.supylabel("Objective (Larger the Better)")

    # Save and display the figure
    plt.tight_layout()  # Adjust layout to fit suptitle
    plt.savefig("hello.png")
    plt.show()