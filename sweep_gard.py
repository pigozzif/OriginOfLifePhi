import os

from plotting import plot_diagnostics, load_data

sigmas = [3.0, 4.0, 4.5, 5.0, 6.0]
kf_kb_ratios = [1e2, 1e3, 1e4]


if __name__ == "__main__":
    for sigma in sigmas:
        for A in sigmas:
            for ratio in kf_kb_ratios:
                exp_name = ".".join([str(sigma).replace(".", ","),
                                     str(A).replace(".", ","),
                                     str(ratio).replace(".", ",")])
                kb = 1e-5
                kf = kb * ratio
                for seed in range(1):
                    os.system(f"python gard.py --seed={seed} --n_gen=1000 --kf={kf} --kb={kb} --A={-A} --sigma={sigma}")
                    print(seed)
                os.makedirs(f"figures/{exp_name}", exist_ok=True)
                try:
                    plot_diagnostics(data=load_data(),
                                     exp_name=os.path.join("figures", exp_name))
                except ValueError:
                    pass
