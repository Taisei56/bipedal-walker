# Straight-Legged Bipedal Walker

Undergraduate research project — NTU Mechanical Engineering, URECA program.
Supervised by Associate Professor Xie Ming.

Most bipedal walking models solve the ground clearance problem the same way: bend the knee, drop the hip, or shorten the leg. All of these add mechanical complexity.

This project looks at whether lateral body sway alone is enough. When the body rocks sideways over the stance foot, the opposite hip rises. If the sway is large enough, a fully straight swing leg can pass forward without touching the ground — no knee required.

The model is two rigid legs connected at a hip joint. The stance leg rocks in the frontal plane (left-right). The swing leg swings in the sagittal plane (front-back). These two motions are coupled through the hip, and the coupling is what makes the gait work.

Still work in progress. 



## Running it

Requires MuJoCo 3.x and Python 3.10+.

```bash
pip install mujoco numpy

# passive simulation on a slope (no actuation)
python sim/walker.py --mode passive --duration 10 --render

# minimal actuation on flat ground
python sim/walker.py --mode actuated --duration 10 --render
```

## Notes

The floor in `bipedal.xml` has a small slope (`zaxis="0.04 0 1"`) for passive walking tests. Set `zaxis="0 0 1"` for flat ground and switch to actuated mode.

The `energy_injection_controller` in `walker.py` is a placeholder 