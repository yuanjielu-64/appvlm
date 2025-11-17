from gym.envs.registration import register

register(
    id="dwa_param-v0",
    entry_point="envs.DWA.parameter_tuning_envs:DWAParamContinuousLaser"
)

register(
    id="teb_param-v0",
    entry_point="envs.Teb.parameter_tuning_envs:TebParamContinuousLaser"
)

register(
    id="eband_param-v0",
    entry_point="envs.Eband.parameter_tuning_envs:EbandParamContinuousLaser"
)

register(
    id="mppi_param-v0",
    entry_point="envs.MPPI.mppi_envs:MPPIPlanning"
)

register(
    id="ddp_param-v0",
    entry_point="envs.DDP.ddp_envs:DDPPlanning"
)