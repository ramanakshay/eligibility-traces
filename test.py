import gymnasium
import particle_envs

env = gymnasium.make('particle_envs/particle-v0')

#
for i in range(10):
    state, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        print("State: ", state, "Action: ", action, "Next State: ", next_state, "Reward: ", reward, "Done: ", done, "Info: ", info)
        state = next_state
        print("Episode: ", i)
        print("Final State: ", state)
        print("Final Reward: ", reward)
        print("Final Done: ", done)
        print("Final Info: ", info)
        print("\n\n\n")