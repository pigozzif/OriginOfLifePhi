import os

sigmas = [3.0, 4.0, 4.5, 5.0, 6.0]
kf_kb_ratios = [1e2, 1e3, 1e4]


if __name__ == "__main__":
    for sigma in sigmas:
        for A in sigmas:
            for ratio in kf_kb_ratios:
                exp_name = ".".join([str(sigma), str(A), str(ratio)])
                kb = 1e-5
                kf = kb * ratio
                for seed in range(100):
                    os.system(f"python gard.py --seed={seed} --n_gen=1000 --kf={kf} --kb={kb} --A={A} --sigma={sigma}")
                    print(seed)
                os.makedirs(f"figures/{exp_name}", exist_ok=True)
