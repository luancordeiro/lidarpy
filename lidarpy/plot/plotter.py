import matplotlib.pyplot as plt


def plot_3graph_std(z,
                    alpha_avg,
                    beta_avg,
                    lidar_ratio_avg,
                    alpha_std=None,
                    beta_std=None,
                    lidar_ratio_std=None,
                    save_name=None):
    plt.rcParams.update({'font.family': 'serif', 'font.size': 22, 'font.weight': 'light'})
    plt.rc('legend', fontsize=16)

    alpha_label = r"Extinction ($\times 10^4$ m$^{-1}$)"
    beta_label = r"Backscatter ($\times 10^6$ m$^{-1}$sr$^{-1}$)"

    fmt = "-"
    alpha = 1 if alpha_std is None else 0.5

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    fig.set_figwidth(15)
    fig.set_figheight(9)

    ax1.plot(alpha_avg * 1e4,
             z,
             fmt,
             alpha=alpha,
             color="black")

    if alpha_std is not None:
        ax1.fill_betweenx(z,
                          (alpha_avg - alpha_std) * 1e4,
                          (alpha_avg + alpha_std) * 1e4,
                          color='green',
                          alpha=0.18)

    ax1.grid()
    ax1.grid(alpha=0.65)
    ax1.set_ylabel("Height (m)")
    ax1.set_xlabel(alpha_label)

    ax2.plot(beta_avg * 1e6,
             z,
             fmt,
             alpha=alpha,
             color="black")

    if beta_std is not None:
        ax2.fill_betweenx(z,
                          (beta_avg - beta_std) * 1e6,
                          (beta_avg + beta_std) * 1e6,
                          color='green',
                          alpha=0.18)

    ax2.grid()
    ax2.grid(alpha=0.65)
    ax2.set_xlabel(beta_label)

    ax3.plot(lidar_ratio_avg,
             z,
             fmt,
             alpha=alpha,
             color="black")

    if lidar_ratio_std is not None:
        ax3.fill_betweenx(z,
                          (lidar_ratio_avg - lidar_ratio_std),
                          (lidar_ratio_avg + lidar_ratio_std),
                          color='green',
                          alpha=0.18)

    ax3.grid()
    ax3.grid(alpha=0.65)
    ax3.set_xlabel(r"Lidar Ratio (sr)")

    if save_name is not None:
        plt.savefig(f"{save_name}")

    plt.show()


def compare_w_sol(x_comp, y_comp, x_exact, y_exact, ylabel):
    plt.figure(figsize=(12, 7))
    plt.plot(x_comp, y_comp, "-", linewidth=2.5, color="black", alpha=0.6, label="comp")
    plt.plot(x_exact, y_exact, "--", linewidth=2, color="red", alpha=0.7, label="exact")
    plt.legend()
    plt.xlabel("Height (m)")
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()
