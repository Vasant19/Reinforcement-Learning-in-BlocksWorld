from gymnasium.envs.registration import register

register(
    id="blocksworld_env/GridWorld-v0",
    entry_point="blocksworld_env.envs:GridWorldEnv",
)

register(
    id="blocksworld_env/BlocksWorld-v0",
    entry_point="blocksworld_env.envs:BlocksWorldEnv",
)

register(
    id="blocksworld_env/BlocksWorldEnvTarget-v0",
    entry_point="blocksworld_env.envs:BlocksWorldEnvTarget",
)