# OptStopRandNN

Maintained subset of the original `OptStopRandNN` code focused on:

- Black-Scholes experiments
- fractional Brownian motion experiments
- backward-induction methods: `LSM`, `NLSM`, `DOS`, `RLSM`, `RRLSM`
- tree baselines: `B`, `Trinomial`
- path samplers for Black-Scholes: `mc`, `mc_antithetic`, `sobol`, `sobol_scrambled`

## Installation

```sh
cd OptStopRandNN
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -e .
```

## Running

The main entry point is:

```sh
python optimal_stopping/run/run_algo.py --configs=... --nb_jobs=N
```

Useful flags:

- `--path_gen_seed=<int>`
- `--path_sampler=mc|mc_antithetic|sobol|sobol_scrambled`
- `--compute_greeks=True`
- `--greeks_method=central|regression`

## Example configs

Black-Scholes pricing:

```sh
python optimal_stopping/run/run_algo.py \
  --configs=table_spots_Dim_BS_MaxCallr0,table_spots_Dim_BS_MaxCallr0_do,table_spots_Dim_BS_MaxCallr0_bf \
  --nb_jobs=4
```

Black-Scholes convergence:

```sh
python optimal_stopping/run/run_algo.py \
  --configs=table_conv_study_BS_LND \
  --nb_jobs=4 \
  --path_sampler=sobol
```

Fractional Brownian motion:

```sh
python optimal_stopping/run/run_algo.py \
  --configs=table_RNN_DOS,table_RNN_DOS_PD,table_RNN_DOS_randRNN \
  --nb_jobs=4
```

Greeks:

```sh
python optimal_stopping/run/run_algo.py \
  --configs=table_greeks_1,table_greeks_1_2 \
  --nb_jobs=1 \
  --compute_greeks=True \
  --greeks_method=central \
  --fd_compute_gamma_via_PDE=True \
  --eps=1e-9 \
  --fd_freeze_exe_boundary=True
```
