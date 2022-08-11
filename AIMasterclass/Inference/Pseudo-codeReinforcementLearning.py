# Reinforcement Learning pseudo código

obs = env.reset()
h = mdnrnn.initial_state
done = False
cumulative_reward = 0
while not done:
    z = cvae(obs)
    a = controller([z, h])
    obs, reward, done = env.step(a)
    cumulative_reward += reward
    h = mdnrnn([a, z, h])
