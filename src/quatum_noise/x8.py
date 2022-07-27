import json
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields import RemoteEngine
from strawberryfields.utils import random_interferometer


class X8:
    def __init__(self):
        self.eng = RemoteEngine("X8")

    def get_samples(self):
        prog = sf.Program(8, name="remote_job1")
        U = random_interferometer(4)

        with prog.context as q:
            # Initial squeezed states
            # Allowed values are r=1.0 or r=0.0
            ops.S2gate(1.0) | (q[0], q[4])
            ops.S2gate(1.0) | (q[1], q[5])
            ops.S2gate(1.0) | (q[3], q[7])

            # Interferometer on the signal modes (0-3)
            ops.Interferometer(U) | (q[0], q[1], q[2], q[3])
            ops.BSgate(0.543, 0.123) | (q[2], q[0])
            ops.Rgate(0.453) | q[1]
            ops.MZgate(0.65, -0.54) | (q[2], q[3])

            # *Same* interferometer on the idler modes (4-7)
            ops.Interferometer(U) | (q[4], q[5], q[6], q[7])
            ops.BSgate(0.543, 0.123) | (q[6], q[4])
            ops.Rgate(0.453) | q[5]
            ops.MZgate(0.65, -0.54) | (q[6], q[7])

            ops.MeasureFock() | q

        results = self.eng.run(prog, shots=100000)

        return results.samples


x8 = X8()

for i in range(10):
    samples = x8.get_samples()
    with open(f"quantum_noise/{i}.json", "w") as f:
        json.dump(samples.tolist(), f)
