from dataclasses import dataclass
import typing
from typing import Iterable

import numpy as np

FigureType = typing.NewType("FigureType", str)
TablePrice = FigureType("TablePrice")
TableDuration = FigureType("TableDuration")
PricePerNbPaths = FigureType("PricePerNbPaths")


class Seed:
    def __init__(self):
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed

    def get_seed(self):
        return self.seed


path_gen_seed = Seed()
QMC_PATH_SAMPLERS = (
    "mc",
    "mc_antithetic",
    "sobol_seq",
    "sobol_bb",
    "sobol_scrambled_seq",
    "sobol_scrambled_bb",
)


@dataclass
class _DefaultConfig:
    algos: Iterable[str] = ("NLSM", "LSM", "DOS", "RLSM")
    path_samplers: Iterable[str] = ("mc",)
    dividends: Iterable[float] = (0.0,)
    nb_dates: Iterable[int] = (10,)
    drift: Iterable[float] = (0.02,)
    mean: Iterable[float] = (0.01,)
    speed: Iterable[float] = (2.0,)
    correlation: Iterable[float] = (-0.3,)
    hurst: Iterable[float] = (0.75,)
    stock_models: Iterable[str] = ("BlackScholes",)
    strikes: Iterable[float] = (100.0,)
    maturities: Iterable[float] = (1.0,)
    nb_paths: Iterable[int] = (20000,)
    nb_runs: int = 10
    nb_stocks: Iterable[int] = (1,)
    payoffs: Iterable[str] = ("MaxCall",)
    spots: Iterable[float] = (100.0,)
    volatilities: Iterable[float] = (0.2,)
    hidden_size: Iterable[int] = (20,)
    nb_epochs: Iterable[int] = (30,)
    factors: Iterable[Iterable[float]] = ((1.0, 1.0, 1.0),)
    ridge_coeff: Iterable[float] = (1.0,)
    train_ITM_only: Iterable[bool] = (True,)
    use_path: Iterable[bool] = (False,)
    use_payoff_as_input: Iterable[bool] = (False,)
    representations: Iterable[str] = ("TablePriceDuration",)


@dataclass
class _DimensionTable(_DefaultConfig):
    algos: Iterable[str] = ("NLSM", "RLSM")
    nb_stocks: Iterable[int] = (5, 10, 50, 100, 500, 1000, 2000)


@dataclass
class _SmallDimensionTable(_DefaultConfig):
    algos: Iterable[str] = ("LSM",)
    nb_stocks: Iterable[int] = (5, 10, 50, 100)


@dataclass
class _VerySmallDimensionTable(_DefaultConfig):
    nb_stocks: Iterable[int] = (5, 10, 20)
    algos: Iterable[str] = ("NLSM", "RLSM", "LSM", "DOS")


# Black-Scholes tables
table_spots_Dim_BS_MaxCallr0 = _DimensionTable(
    spots=(80.0, 100.0, 120.0), drift=(0.0,), use_payoff_as_input=(True, False)
)
table_spots_Dim_BS_MaxCallr0_do = _DimensionTable(
    algos=("DOS",), spots=(80.0, 100.0, 120.0), drift=(0.0,), use_payoff_as_input=(True, False)
)
table_spots_Dim_BS_MaxCallr0_bf = _SmallDimensionTable(
    spots=(80.0, 100.0, 120.0), drift=(0.0,), use_payoff_as_input=(True, False)
)
table_spots_Dim_MaxCallr0_ref = _DimensionTable(
    spots=(80.0, 100.0, 120.0), drift=(0.0,), algos=("EOP",), use_payoff_as_input=(False,)
)
table_spots_Dim_BS_MaxCallr0_gt1 = _DimensionTable(
    spots=(80.0, 100.0, 120.0),
    algos=("NLSM", "RLSM", "LSM", "DOS", "EOP", "B"),
    ridge_coeff=(1.0, np.nan, None),
    drift=(0.0,),
    use_payoff_as_input=(True, False),
)

table_smallDim_BS_GeoPut = _VerySmallDimensionTable(
    payoffs=("GeometricPut",),
    nb_stocks=(5, 10, 20, 50, 100),
    algos=("LSM",),
    stock_models=("BlackScholes",),
    use_payoff_as_input=(True, False),
)
table_smallDim_BS_GeoPut_ref1 = _VerySmallDimensionTable(
    payoffs=("Put1Dim",), algos=("B",), nb_runs=1, nb_stocks=(1,), nb_dates=(10000,), volatilities=(0.2,)
)
table_smallDim_BS_GeoPut_ref2 = _VerySmallDimensionTable(
    payoffs=("Put1Dim",), algos=("B",), nb_runs=1, nb_stocks=(5,), nb_dates=(10000,), volatilities=(0.2,)
)
table_smallDim_BS_GeoPut_ref3 = _VerySmallDimensionTable(
    payoffs=("Put1Dim",), algos=("B",), nb_runs=1, nb_stocks=(10,), nb_dates=(10000,), volatilities=(0.2,)
)
table_smallDim_BS_GeoPut_ref4 = _VerySmallDimensionTable(
    payoffs=("Put1Dim",), algos=("B",), nb_runs=1, nb_stocks=(20,), nb_dates=(10000,), volatilities=(0.2,)
)
table_smallDim_BS_GeoPut_ref5 = _VerySmallDimensionTable(
    payoffs=("Put1Dim",), algos=("B",), nb_runs=1, nb_stocks=(50,), nb_dates=(10000,), volatilities=(0.2,)
)
table_smallDim_BS_GeoPut_ref6 = _VerySmallDimensionTable(
    payoffs=("Put1Dim",), algos=("B",), nb_runs=1, nb_stocks=(100,), nb_dates=(10000,), volatilities=(0.2,)
)
table_smallDim_BS_GeoPut_ref7 = _VerySmallDimensionTable(
    payoffs=("Put1Dim",), algos=("B",), nb_runs=1, nb_stocks=(500,), nb_dates=(10000,), volatilities=(0.2,)
)
table_smallDim_BS_GeoPut_ref8 = _VerySmallDimensionTable(
    payoffs=("Put1Dim",), algos=("B",), nb_runs=1, nb_stocks=(1000,), nb_dates=(10000,), volatilities=(0.2,)
)
table_smallDim_BS_GeoPut_ref9 = _VerySmallDimensionTable(
    payoffs=("Put1Dim",), algos=("B",), nb_runs=1, nb_stocks=(2000,), nb_dates=(10000,), volatilities=(0.2,)
)
table_GeoPut_payoffs_gt1 = _DimensionTable(
    payoffs=("GeometricPut",),
    algos=("NLSM", "RLSM", "LSM", "DOS", "B"),
    nb_stocks=(5, 10, 20, 50, 100),
    nb_dates=(10, 10000),
    dividends=(0.0,),
    volatilities=(0.2,),
    drift=(0.02,),
    use_payoff_as_input=(True, False),
)

table_Dim_BS_BasktCallr0 = _DimensionTable(
    payoffs=("BasketCall",), algos=("NLSM", "RLSM", "DOS"), drift=(0.0,), use_payoff_as_input=(True, False)
)
table_Dim_BS_BasktCallr0_bf = _SmallDimensionTable(
    payoffs=("BasketCall",), drift=(0.0,), use_payoff_as_input=(True, False)
)
table_spots_Dim_BasktCallr0_ref = _DimensionTable(drift=(0.0,), algos=("EOP",), payoffs=("BasketCall",))
table_BasketCall_payoffsr0_gt = _DimensionTable(
    payoffs=("BasketCall",),
    algos=("NLSM", "RLSM", "LSM", "DOS", "EOP"),
    nb_stocks=(5, 10, 20, 50, 100, 500, 1000, 2000),
    drift=(0.0,),
    use_payoff_as_input=(False,),
)
table_BasketCall_payoffsr0_gt1 = _DimensionTable(
    payoffs=("BasketCall",),
    algos=("NLSM", "RLSM", "LSM", "DOS", "EOP"),
    nb_stocks=(5, 10, 20, 50, 100, 500, 1000, 2000),
    drift=(0.0,),
    use_payoff_as_input=(True, False),
)

# Many-date Black-Scholes tables
table_manyDates_BS_MaxCallr0_1 = _VerySmallDimensionTable(
    algos=("NLSM", "RLSM", "LSM", "DOS"), nb_stocks=(10, 50), nb_dates=(50, 100), drift=(0.0,), use_payoff_as_input=(True, False)
)
table_manyDates_BS_MaxCallr0_2 = _VerySmallDimensionTable(
    algos=("NLSM", "RLSM", "DOS"), nb_stocks=(100, 500), nb_dates=(50, 100), drift=(0.0,), use_payoff_as_input=(True, False)
)
table_manyDates_BS_MaxCallr0_ref = _VerySmallDimensionTable(
    algos=("EOP",), nb_stocks=(10, 50, 100, 500), nb_dates=(50, 100), drift=(0.0,), use_payoff_as_input=(False,)
)
table_manyDates_BS_MaxCallr0_gt1 = _VerySmallDimensionTable(
    algos=("NLSM", "RLSM", "LSM", "DOS", "EOP"),
    nb_stocks=(10, 50, 100, 500),
    nb_dates=(10, 50, 100),
    hidden_size=(20,),
    ridge_coeff=(1.0, np.nan, None),
    drift=(0.0,),
    use_payoff_as_input=(True, False),
)

table_spots_Dim_BS_MinPut = _DimensionTable(
    payoffs=("MinPut",), spots=(80.0, 100.0, 120.0), drift=(0.02,), use_payoff_as_input=(True, False)
)
table_spots_Dim_BS_MinPut_do = _DimensionTable(
    payoffs=("MinPut",), algos=("DOS",), spots=(80.0, 100.0, 120.0), drift=(0.02,), use_payoff_as_input=(True, False)
)
table_spots_Dim_BS_MinPut_bf = _SmallDimensionTable(
    payoffs=("MinPut",), spots=(80.0, 100.0, 120.0), drift=(0.02,), use_payoff_as_input=(True, False)
)
table_spots_Dim_BS_MinPut_gt = _DimensionTable(
    payoffs=("MinPut",),
    spots=(80.0, 100.0, 120.0),
    algos=("NLSM", "RLSM", "LSM", "DOS"),
    ridge_coeff=(1.0, np.nan, None),
    drift=(0.02,),
    use_payoff_as_input=(False,),
)
table_spots_Dim_BS_MinPut_gt1 = _DimensionTable(
    payoffs=("MinPut",),
    spots=(80.0, 100.0, 120.0),
    algos=("NLSM", "RLSM", "LSM", "DOS"),
    ridge_coeff=(1.0, np.nan, None),
    drift=(0.02,),
    use_payoff_as_input=(True, False),
)

table_Dim_BS_MaxCall_div = _DimensionTable(
    payoffs=("MaxCall",), algos=("NLSM", "RLSM", "DOS"), dividends=(0.1,), drift=(0.05,), use_payoff_as_input=(True, False)
)
table_Dim_BS_MaxCall_div_bf = _SmallDimensionTable(
    dividends=(0.1,), payoffs=("MaxCall",), drift=(0.05,), use_payoff_as_input=(True, False)
)
table_Dim_BS_MaxCall_div_gt1 = _DimensionTable(
    payoffs=("MaxCall",), algos=("NLSM", "RLSM", "LSM", "DOS"), dividends=(0.1,), drift=(0.05,), use_payoff_as_input=(True, False)
)

@dataclass
class _DefaultPlotNbPaths(_DefaultConfig):
    nb_runs: int = 20
    nb_stocks: Iterable[int] = (5,)
    maturities: Iterable[int] = (1,)
    representations: Iterable[str] = ("ConvergenceStudy",)


table_conv_study_BS_LND = _DefaultPlotNbPaths(
    nb_paths=list(200 * 2 ** np.array(range(8))),
    hidden_size=(10, 50, 100),
    algos=("RLSM",),
    stock_models=("BlackScholes",),
)


# Hidden-layer randomness
SensRand_greeks_table1 = _DimensionTable(
    nb_runs=10,
    nb_epochs=(10,),
    payoffs=("MaxCall",),
    volatilities=(0.2,),
    drift=(0.02,),
    strikes=(100.0,),
    spots=(100.0,),
    nb_dates=(10,),
    hidden_size=(20,),
    use_payoff_as_input=(False,),
    train_ITM_only=(True,),
    algos=("RLSMSoftplus", "RLSMSoftplusReinit"),
    nb_stocks=(1,),
    nb_paths=(100000,),
)
SensRand_greeks_table1_1 = _DimensionTable(
    nb_runs=10,
    nb_epochs=(10, 30, 50),
    payoffs=("MaxCall",),
    volatilities=(0.2,),
    drift=(0.02,),
    strikes=(100.0,),
    spots=(100.0,),
    nb_dates=(10,),
    hidden_size=(20,),
    use_payoff_as_input=(True,),
    train_ITM_only=(True,),
    algos=("NLSM",),
    nb_stocks=(1,),
    nb_paths=(100000,),
)


# Fractional Brownian motion / path-dependent DOS and regression variants
hurst = list(np.linspace(0, 1, 21))
hurst[0] = 0.01

table_RNN_DOS = _DefaultConfig(
    payoffs=("Identity",),
    nb_stocks=(1,),
    spots=(0,),
    nb_epochs=(30,),
    hurst=hurst,
    train_ITM_only=(False,),
    stock_models=("FractionalBrownianMotion",),
    hidden_size=(20,),
    maturities=(1,),
    nb_paths=(20000,),
    nb_dates=(100,),
    algos=("NLSM", "DOS", "RLSM"),
)
table_RNN_DOS_PD = _DefaultConfig(
    payoffs=("Identity",),
    nb_stocks=(1,),
    spots=(0,),
    nb_epochs=(30,),
    hurst=hurst,
    train_ITM_only=(False,),
    stock_models=("FractionalBrownianMotionPathDep",),
    hidden_size=(20,),
    maturities=(1,),
    nb_paths=(20000,),
    nb_dates=(100,),
    algos=("DOS",),
)
table_RNN_DOS_bf = _DefaultConfig(
    payoffs=("Identity",),
    nb_stocks=(1,),
    spots=(0,),
    nb_epochs=(30,),
    hurst=hurst,
    train_ITM_only=(False,),
    stock_models=("FractionalBrownianMotion",),
    hidden_size=(20,),
    maturities=(1,),
    nb_paths=(20000,),
    nb_dates=(100,),
    algos=("LSM",),
)
factors0 = ([0.0001, 0.3],)
table_RNN_DOS_randRNN = _DefaultConfig(
    payoffs=("Identity",),
    nb_stocks=(1,),
    spots=(0,),
    nb_epochs=(30,),
    hurst=hurst,
    train_ITM_only=(False,),
    factors=factors0,
    stock_models=("FractionalBrownianMotion",),
    hidden_size=(20,),
    maturities=(1,),
    nb_paths=(20000,),
    nb_dates=(100,),
    algos=("RRLSM",),
)

table_highdim_hurst0 = _DefaultConfig(
    payoffs=("Identity",),
    nb_stocks=(1,),
    spots=(0,),
    nb_epochs=(30,),
    hurst=(0.05,),
    train_ITM_only=(False,),
    stock_models=("FractionalBrownianMotion",),
    hidden_size=(20,),
    maturities=(1,),
    nb_paths=(20000,),
    nb_dates=(100,),
    algos=("DOS", "RLSM"),
)
table_highdim_hurst_PD0 = _DefaultConfig(
    payoffs=("Identity",),
    nb_stocks=(1,),
    spots=(0,),
    nb_epochs=(30,),
    hurst=(0.05,),
    train_ITM_only=(False,),
    use_path=(True,),
    stock_models=("FractionalBrownianMotion",),
    hidden_size=(20,),
    maturities=(1,),
    nb_paths=(20000,),
    nb_dates=(100,),
    algos=("DOS",),
)
factors = ([0.0008, 0.11],)
table_highdim_hurst_RNN0 = _DefaultConfig(
    payoffs=("Identity",),
    nb_stocks=(1,),
    spots=(0,),
    nb_epochs=(30,),
    hurst=(0.05,),
    train_ITM_only=(False,),
    factors=factors,
    stock_models=("FractionalBrownianMotion",),
    hidden_size=(20,),
    maturities=(1,),
    nb_paths=(20000,),
    nb_dates=(100,),
    algos=("RRLSM",),
)

table_highdim_hurst = _DefaultConfig(
    payoffs=("Max", "Mean"),
    nb_stocks=(5, 10),
    spots=(0,),
    nb_epochs=(30,),
    hurst=(0.05,),
    train_ITM_only=(False,),
    stock_models=("FractionalBrownianMotion",),
    hidden_size=(20,),
    maturities=(1,),
    nb_paths=(20000,),
    nb_dates=(100,),
    use_payoff_as_input=(True,),
    algos=("DOS", "RLSM"),
)
table_highdim_hurst_PD = _DefaultConfig(
    payoffs=("Max", "Mean"),
    nb_stocks=(5, 10),
    spots=(0,),
    nb_epochs=(30,),
    hurst=(0.05,),
    train_ITM_only=(False,),
    use_path=(True,),
    use_payoff_as_input=(True,),
    stock_models=("FractionalBrownianMotion",),
    hidden_size=(20,),
    maturities=(1,),
    nb_paths=(20000,),
    nb_dates=(100,),
    algos=("pathDOS",),
)
table_highdim_hurst_RNN = _DefaultConfig(
    payoffs=("Max", "Mean"),
    nb_stocks=(5, 10),
    spots=(0,),
    nb_epochs=(30,),
    hurst=(0.05,),
    train_ITM_only=(False,),
    factors=factors,
    stock_models=("FractionalBrownianMotion",),
    hidden_size=(20,),
    maturities=(1,),
    nb_paths=(20000,),
    nb_dates=(100,),
    use_payoff_as_input=(True,),
    algos=("RRLSM",),
)
factors1 = tuple([str((1.0, 1.0, 1.0))] + [str(x) for x in tuple(factors) + tuple(factors0)])
table_highdim_hurst_gt = _DefaultConfig(
    payoffs=("Identity", "Max", "Mean"),
    nb_stocks=(1, 5, 10),
    spots=(0,),
    nb_epochs=(30,),
    hurst=(0.05,),
    train_ITM_only=(False,),
    stock_models=("FractionalBrownianMotion",),
    hidden_size=(20,),
    maturities=(1,),
    nb_paths=(20000,),
    nb_dates=(100,),
    factors=factors1,
    use_payoff_as_input=(True, False),
    algos=("DOS", "pathDOS", "RLSM", "RRLSM"),
)


table_Ridge_MaxCall = _SmallDimensionTable(
    spots=(100.0,), algos=("LSMRidge", "RLSMRidge"), ridge_coeff=(1.0, 0.5, 2.0)
)
table_OtherBasis_MaxCall = _SmallDimensionTable(
    spots=(100.0,), algos=("LSMLaguerre", "LSM"), nb_stocks=(5, 10, 50), nb_runs=10
)


# Greeks
test_table_greeks_model_compare_1d_qmc = _DimensionTable(
    nb_runs=1,
    nb_epochs=(10,),
    payoffs=("MinPut",),
    volatilities=(0.2,),
    drift=(0.05,),
    strikes=(100.0,),
    spots=(100.0,),
    maturities=(1.0,),
    nb_dates=(50,),
    hidden_size=(10,),
    use_payoff_as_input=(True,),
    train_ITM_only=(True,),
    algos=("LSM", "NLSM", "RLSM"),
    path_samplers=QMC_PATH_SAMPLERS,
    nb_stocks=(1,),
    nb_paths=(50000,),
    ridge_coeff=(np.nan,),
)
table_greeks_model_compare_1d_qmc = _DimensionTable(
    nb_runs=10,
    nb_epochs=(10,),
    payoffs=("MinPut",),
    volatilities=(0.2,),
    drift=(0.05,),
    strikes=(100.0,),
    spots=(100.0,),
    maturities=(1.0,),
    nb_dates=(50,),
    hidden_size=(10,),
    use_payoff_as_input=(True,),
    train_ITM_only=(True,),
    algos=("LSM", "NLSM", "RLSM"),
    path_samplers=QMC_PATH_SAMPLERS,
    nb_stocks=(1,),
    nb_paths=(50000,),
    ridge_coeff=(np.nan,),
)
table_greeks_model_compare_1d_binomial = _DimensionTable(
    nb_runs=1,
    algos=("B",),
    payoffs=("Put1Dim",),
    volatilities=(0.2,),
    drift=(0.05,),
    strikes=(100.0,),
    spots=(100.0,),
    maturities=(1.0,),
    nb_dates=(1000, 2000, 4000),
    nb_stocks=(1,),
    ridge_coeff=(np.nan,),
)
test_table_greeks_1 = _DimensionTable(
    nb_runs=1,
    nb_epochs=(20,),
    payoffs=("MinPut",),
    volatilities=(0.2,),
    drift=(0.06,),
    strikes=(36.0,),
    spots=(40.0,),
    nb_dates=(10,),
    hidden_size=(100,),
    use_payoff_as_input=(True, False),
    train_ITM_only=(True,),
    algos=("RLSMSoftplus",),
    nb_stocks=(1,),
    nb_paths=(100000,),
)
test_table_greeks_1_qmc = _DimensionTable(
    nb_runs=1,
    nb_epochs=(20,),
    payoffs=("MinPut",),
    volatilities=(0.2,),
    drift=(0.06,),
    strikes=(36.0,),
    spots=(40.0,),
    nb_dates=(10,),
    hidden_size=(100,),
    use_payoff_as_input=(True, False),
    train_ITM_only=(True,),
    algos=("RLSMSoftplus",),
    path_samplers=QMC_PATH_SAMPLERS,
    nb_stocks=(1,),
    nb_paths=(100000,),
)
table_greeks_1 = _DimensionTable(
    nb_runs=10,
    nb_epochs=(10,),
    payoffs=("MinPut",),
    volatilities=(0.2,),
    drift=(0.06,),
    strikes=(36.0, 40.0, 44.0),
    spots=(40.0,),
    nb_dates=(10,),
    hidden_size=(10,),
    use_payoff_as_input=(True,),
    train_ITM_only=(True,),
    algos=("RLSMSoftplus", "RLSMElu", "RLSMSilu", "RLSMGelu", "RLSMTanh", "RLSM", "LSM", "NLSM", "DOS"),
    nb_stocks=(1,),
    nb_paths=(100000,),
)
table_greeks_1_qmc = _DimensionTable(
    nb_runs=10,
    nb_epochs=(10,),
    payoffs=("MinPut",),
    volatilities=(0.2,),
    drift=(0.06,),
    strikes=(36.0, 40.0, 44.0),
    spots=(40.0,),
    nb_dates=(10,),
    hidden_size=(10,),
    use_payoff_as_input=(True,),
    train_ITM_only=(True,),
    algos=("RLSMSoftplus", "RLSMElu", "RLSMSilu", "RLSMGelu", "RLSMTanh", "RLSM", "LSM", "NLSM", "DOS"),
    path_samplers=QMC_PATH_SAMPLERS,
    nb_stocks=(1,),
    nb_paths=(100000,),
)
table_greeks_1_2 = _DimensionTable(
    nb_runs=10,
    nb_epochs=(10,),
    payoffs=("MinPut",),
    volatilities=(0.2,),
    drift=(0.06,),
    strikes=(36.0, 40.0, 44.0),
    spots=(40.0,),
    nb_dates=(10,),
    hidden_size=(10,),
    use_payoff_as_input=(True,),
    train_ITM_only=(True,),
    algos=("NLSM", "DOS"),
    nb_stocks=(1,),
    nb_paths=(100000,),
)
table_greeks_1_2_qmc = _DimensionTable(
    nb_runs=10,
    nb_epochs=(10,),
    payoffs=("MinPut",),
    volatilities=(0.2,),
    drift=(0.06,),
    strikes=(36.0, 40.0, 44.0),
    spots=(40.0,),
    nb_dates=(10,),
    hidden_size=(10,),
    use_payoff_as_input=(True,),
    train_ITM_only=(True,),
    algos=("NLSM", "DOS"),
    path_samplers=QMC_PATH_SAMPLERS,
    nb_stocks=(1,),
    nb_paths=(100000,),
)
table_greeks_binomial = _DimensionTable(
    nb_runs=1,
    algos=("B",),
    payoffs=("Put1Dim",),
    volatilities=(0.2,),
    drift=(0.06,),
    strikes=(36.0, 40.0, 44.0),
    spots=(40.0,),
    nb_dates=(10000, 50000),
    nb_stocks=(1,),
)
spots = np.linspace(20, 60, 41)
table_greeks_plots = _DimensionTable(
    nb_runs=5,
    nb_epochs=(10,),
    payoffs=("MinPut",),
    volatilities=(0.1, 0.2, 0.3, 0.4),
    maturities=(1, 0.5, 2, 4, 8),
    drift=(0.06,),
    strikes=(40.0,),
    spots=tuple(spots.tolist()),
    nb_dates=(10,),
    hidden_size=(10,),
    use_payoff_as_input=(True,),
    train_ITM_only=(True,),
    algos=("RLSMSoftplus",),
    nb_stocks=(1,),
    nb_paths=(100000,),
)
table_greeks_plots_qmc = _DimensionTable(
    nb_runs=5,
    nb_epochs=(10,),
    payoffs=("MinPut",),
    volatilities=(0.1, 0.2, 0.3, 0.4),
    maturities=(1, 0.5, 2, 4, 8),
    drift=(0.06,),
    strikes=(40.0,),
    spots=tuple(spots.tolist()),
    nb_dates=(10,),
    hidden_size=(10,),
    use_payoff_as_input=(True,),
    train_ITM_only=(True,),
    algos=("RLSMSoftplus",),
    path_samplers=QMC_PATH_SAMPLERS,
    nb_stocks=(1,),
    nb_paths=(100000,),
)
table_greeks_plots_binomial = _DimensionTable(
    nb_runs=1,
    payoffs=("Put1Dim",),
    volatilities=(0.1, 0.2, 0.3, 0.4),
    maturities=(1, 0.5, 2, 4, 8),
    drift=(0.06,),
    strikes=(40.0,),
    spots=tuple(spots.tolist()),
    nb_dates=(10000,),
    hidden_size=(10,),
    algos=("B",),
    nb_stocks=(1,),
)


# Upper bound
table_price_lower_upper_1 = _DimensionTable(
    nb_runs=10,
    nb_epochs=(10,),
    payoffs=("MaxCall",),
    volatilities=(0.2,),
    drift=(0.05,),
    dividends=(0.1,),
    strikes=(100.0,),
    spots=(90.0, 100.0, 110.0),
    nb_dates=(9,),
    maturities=(3,),
    hidden_size=(100,),
    use_payoff_as_input=(True,),
    train_ITM_only=(False,),
    algos=("RLSMSoftplus",),
    nb_stocks=(2, 3, 5, 10, 20),
    nb_paths=(100000,),
)
table_price_lower_upper_1_1 = _DimensionTable(
    nb_runs=10,
    nb_epochs=(10,),
    payoffs=("MaxCall",),
    volatilities=(0.2,),
    drift=(0.05,),
    dividends=(0.1,),
    strikes=(100.0,),
    spots=(90.0, 100.0, 110.0),
    nb_dates=(9,),
    maturities=(3,),
    hidden_size=(-5,),
    use_payoff_as_input=(True,),
    train_ITM_only=(False,),
    algos=("RLSMSoftplus",),
    nb_stocks=(2, 3, 5, 10, 20, 30, 50, 100, 200, 500),
    nb_paths=(20000,),
)


single_test_maxcall_1stock = _SmallDimensionTable(
    algos=("RLSMSoftplus",),
    payoffs=("MaxCall",),
    nb_stocks=(1,),
    nb_paths=(1024,),
    nb_dates=(10,),
    spots=(100.0,),
    strikes=(100.0,),
    drift=(0.02,),
    volatilities=(0.2,),
    hidden_size=(20,),
    nb_runs=1,
    use_payoff_as_input=(True,),
)

test_table = _SmallDimensionTable(
    spots=(10.0,),
    strikes=(10.0,),
    algos=(
        "NLSM",
        "LSM",
        "DOS",
        "RLSM",
        "LSMLaguerre",
        "LSMRidge",
        "RLSMRidge",
        "RLSMTanh",
        "RRLSM",
        "RRLSMmix",
        "LSMDeg1",
    ),
    nb_stocks=(5,),
    nb_dates=(5,),
    nb_paths=(100,),
    use_payoff_as_input=(True, False),
    nb_runs=1,
    factors=((0.001, 0.001, 0.001),),
)

test_table2 = _SmallDimensionTable(
    spots=(10.0,),
    strikes=(10.0,),
    algos=("NLSM", "LSM", "DOS", "RLSM", "RRLSM", "RRLSMmix", "pathDOS"),
    stock_models=("FractionalBrownianMotion", "FractionalBrownianMotionPathDep"),
    hurst=(0.25,),
    nb_stocks=(5,),
    nb_dates=(5,),
    nb_paths=(100,),
    use_payoff_as_input=(True, False),
    nb_runs=1,
    factors=((0.001, 0.001, 0.001),),
)
