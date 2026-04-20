# OptStopRandNN

Maintained subset of the original `OptStopRandNN` code focused on:

- Black-Scholes experiments
- fractional Brownian motion experiments
- backward-induction methods: `LSM`, `NLSM`, `DOS`, `RLSM`, `RRLSM`
- tree baselines: `B`, `Trinomial`
- path samplers for Black-Scholes: `mc`, `mc_antithetic`, `sobol`, `sobol_seq`, `sobol_bb`, `sobol_scrambled`, `sobol_scrambled_seq`, `sobol_scrambled_bb`

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
- `--path_sampler=mc|mc_antithetic|sobol|sobol_seq|sobol_bb|sobol_scrambled|sobol_scrambled_seq|sobol_scrambled_bb`
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

## Experiment commands

The commands below cover the usual workflow for reproducing pricing and Greek experiments, then exporting tables and plots from the resulting metrics CSVs.

Find the latest metrics files:

```sh
cd OptStopRandNN
ls -t output/metrics_draft/*.csv | head
```

Run the built-in Black-Scholes Greek sweep with the helper script:

```sh
cd OptStopRandNN
bash scripts/run_black_scholes_greeks.sh \
  --method both \
  --jobs 1 \
  --algos LSM,NLSM,RLSM,RLSMSoftplus \
  --path-samplers mc,mc_antithetic,sobol_seq,sobol_bb,sobol_scrambled_seq,sobol_scrambled_bb
```

Run the same central-difference Greek sweep directly:

```sh
cd OptStopRandNN
python optimal_stopping/run/run_algo.py \
  --configs=table_greeks_1,table_greeks_1_2 \
  --algos=LSM,NLSM,RLSM,RLSMSoftplus \
  --path_samplers=mc,mc_antithetic,sobol_seq,sobol_bb,sobol_scrambled_seq,sobol_scrambled_bb \
  --nb_jobs=1 \
  --compute_greeks=True \
  --greeks_method=central \
  --fd_compute_gamma_via_PDE=True \
  --eps=1e-9 \
  --fd_freeze_exe_boundary=True
```

Run the regression-based Greek sweep directly:

```sh
cd OptStopRandNN
python optimal_stopping/run/run_algo.py \
  --configs=table_greeks_1,table_greeks_1_2 \
  --algos=LSM,NLSM,RLSM,RLSMSoftplus \
  --path_samplers=mc,mc_antithetic,sobol_seq,sobol_bb,sobol_scrambled_seq,sobol_scrambled_bb \
  --nb_jobs=1 \
  --compute_greeks=True \
  --greeks_method=regression \
  --reg_eps=5 \
  --eps=1e-9 \
  --poly_deg=9 \
  --fd_freeze_exe_boundary=True
```

Run the binomial benchmark used by the model-comparison exporter:

```sh
cd OptStopRandNN
python optimal_stopping/run/run_algo.py \
  --configs=table_greeks_binomial \
  --algos=B \
  --nb_jobs=1 \
  --compute_greeks=True \
  --greeks_method=central \
  --fd_compute_gamma_via_PDE=True \
  --eps=1e-9
```

Export compact runtime and price tables from one metrics CSV:

```sh
cd OptStopRandNN
python optimal_stopping/utilities/export_runtime_compact_tables.py \
  --csv-filename <run_id>.csv
```

Export the 1D Greek model-comparison tables from one metrics CSV. This requires the CSV to include binomial (`B`) rows:

```sh
cd OptStopRandNN
python optimal_stopping/utilities/export_greeks_model_comparison.py \
  --csv-filename <run_id>.csv
```

Plot price and Greeks from one saved metrics CSV:

```sh
cd OptStopRandNN
python optimal_stopping/utilities/plot_greeks_from_csv.py \
  --csv-path output/metrics_draft/<run_id>.csv \
  --algo RLSMSoftplus \
  --greeks-method central \
  --path-samplers mc,mc_antithetic,sobol_seq,sobol_bb,sobol_scrambled_seq,sobol_scrambled_bb \
  --x-col spot \
  --group-cols volatility,maturity \
  --agg median
```

Create one spot-curve PDF per sampler from one metrics CSV:

```sh
cd OptStopRandNN
python optimal_stopping/utilities/plot_greeks_curves_by_method.py \
  --csv-path output/metrics_draft/<run_id>.csv \
  --algo RLSMSoftplus \
  --greeks-method central \
  --methods mc,mc_antithetic,sobol_seq,sobol_bb,sobol_scrambled_seq,sobol_scrambled_bb
```

Aggregate across all draft metrics CSVs and build the runtime-vs-error efficiency frontier:

```sh
cd OptStopRandNN
python optimal_stopping/utilities/plot_greeks_efficiency.py \
  --algo RLSMSoftplus \
  --greeks-method regression \
  --methods mc,mc_antithetic,sobol_seq,sobol_bb,sobol_scrambled_seq,sobol_scrambled_bb \
  --read-which 0
```

Aggregate across all draft metrics CSVs and plot Greeks against maturity for several moneyness slices:

```sh
cd OptStopRandNN
python optimal_stopping/utilities/plot_greeks_maturity_slices.py \
  --config table_greeks_plots \
  --algo RLSMSoftplus \
  --greeks-method regression \
  --volatility 0.2 \
  --methods mc,mc_antithetic,sobol_seq,sobol_bb,sobol_scrambled_seq,sobol_scrambled_bb \
  --moneyness-slices 0.90,1.00,1.10 \
  --read-which 0
```
