from gym.envs.registration import register

register(
      id='RadarMap-DoubleIntegrator-v0',
      entry_point='radar_maps.env:RadarMap_DoubleIntegrator',
      max_episode_steps=1000
  )