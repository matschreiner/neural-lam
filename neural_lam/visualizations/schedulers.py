if __name__ == "__main__":
    # Run this code to visualize the learning rate schedule
    # Third-party
    import matplotlib.pyplot as plt

    model = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters())
    scheduler = WarmupCosineAnnealingLR(
        opt, warmup_steps=20, annealing_steps=100
    )

    lrs = []
    for _ in range(150):
        lrs.append(opt.param_groups[0]["lr"])
        scheduler.step()

    plt.plot(lrs)
    plt.vlines(20, 0, max(lrs), colors="k", linestyles="dashed")
    plt.vlines(120, 0, max(lrs), colors="k", linestyles="dashed")
    plt.text(21, max(lrs) / 2, "warmup ended", fontsize=10, color="k")
    plt.text(121, max(lrs) / 2, "annealing ended", fontsize=10, color="k")

    plt.hlines(max(lrs), 15, 25, colors="k", linestyles="dashed")
    plt.text(26, max(lrs), f"{max(lrs):.2e}", fontsize=10, color="k")
    plt.text(121, min(lrs), f"{min(lrs):.2e}", fontsize=10, color="k")

    plt.xlabel("Step")
    plt.ylabel("Learning Rate")

    plt.show()
