import yaml
from dataclasses import dataclass
from calliope import nempy_utils
import pandas as pd

@dataclass
class Config:
    """
    Class to hold the config to represent an ESS.
    """
    # id info
    station_name: str
    region: str

    # bess info
    capacity: float
    duration: float
    charge_efficiency: float
    discharge_efficiency: float
    min_energy: float
    max_energy: float
    cycle_limit: float
    max_daily_cycles: float
    init_storage: float
    final_storage: float
    raisefast_capacity: float
    raiseslow_capacity: float
    raisedelay_capacity: float
    raisereg_capacity: float
    lowerfast_capacity: float
    lowerslow_capacity: float
    lowerdelay_capacity: float
    lowerreg_capacity: float

    # bid extra info
    mlf_gen: float
    mlf_load: float

    # optional stuff
    rreg_util: float = 0.18
    lreg_util: float = 0.08

    # post init
    gen_duid: str = ""
    load_duid: str = ""

    def __post_init__(self):
        # format the generator and load duids
        self.gen_duid = f'{self.station_name}G1'
        self.load_duid = f'{self.station_name}L1'

    def get_unit_info(self):
        """
        Get unit information for `nempy.markets.SpotMarket` class.
        """
        unit_gen = nempy_utils.format_unit_info(
            unit=self.gen_duid, 
            dispatch_type='generator', 
            region=self.region,
            loss_factor=self.mlf_gen
        )
        unit_load = nempy_utils.format_unit_info(
            unit=self.load_duid, 
            dispatch_type='load', 
            region=self.region,
            loss_factor=self.mlf_load
        )
        df = pd.concat([unit_gen, unit_load]).reset_index(drop=True)
        return df


def load_config(path: str) -> Config:
    """
    Load a yaml file into a Config object.

    Parameters
    ----------
    path : str
        path to the yaml file

    Returns
    -------
    Config
        returns a Config object to represent an ESS
    """
    with open(path) as stream:
        cfg = yaml.safe_load(stream)

    # Create the Config object
    config_obj = Config(**cfg)

    return config_obj
