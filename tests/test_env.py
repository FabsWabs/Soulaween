from soulaween.env.soulaween import Soulaween

env = Soulaween()
env.reset()
env.render()
while True:
    m = np.random.choice(env.legal_actions)
    env.step(m)
    env.render()
for m in moves:
    env.step(m)
    env.render()
print(env.legal_actions)
env.step(0)
env.render()
print(env.legal_actions)
print()