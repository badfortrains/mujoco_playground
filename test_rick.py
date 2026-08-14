import sys
import jax
from mujoco_playground._src import locomotion

try:
    env = locomotion.load("RickJoystickFlatTerrain")
    print("Environment loaded successfully!")
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    print("Environment reset successfully!")
    print(f"Action size: {env.action_size}")
except Exception as e:
    print(f"Error loading environment: {e}")
    sys.exit(1)
