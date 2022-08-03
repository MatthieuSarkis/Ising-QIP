# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis (https://github.com/MatthieuSarkis).
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from typing import Optional
from qiskit import Aer
from qiskit.providers.aer import AerError
from qiskit.utils import QuantumInstance

class QiskitKernel():

    def __init__(
        self,
        backend_type: Optional[str] = 'simulator',
        backend_name: Optional[str] = 'statevector_simulator',
        mitigate: bool = False,
        seed: Optional[int] = 42,
        shots: Optional[int] = None,
        hub: Optional[str] = 'ibm-q',
        group: Optional[str] = 'open',
        project: Optional[str] = 'main',
        job_name: Optional[str] = 'ising_qip',
    ) -> None:
        """ Initialize the Projected Kernel Ridge Regression from parent's init. """

        super(QiskitKernel, self).__init__()

        self._backend_type = backend_type
        self._backend_name = backend_name
        self.mitigate = mitigate
        self._seed = seed
        self._shots = shots
        self._hub = hub
        self._group = group
        self._project = project
        self._job_name = job_name

        self._backend = None
        self.qi: Optional[QuantumInstance] = None
        self.get_quantum_instance()

    def get_quantum_instance(self) -> None:
        """ Get the quantum instance for quamtum experiment. """

        if self._backend_type == "GPU":
            self._backend = Aer.get_backend(self._backend_name)
            try:
                self._backend.set_options(device='GPU')
            except AerError as e:
                print(e)

        elif self._backend_type == "IBMQ":
            from qiskit import IBMQ
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub=self._hub, group=self._group, project=self._project)
            self._backend = provider.get_backend(self._backend_name)

        else:
            self._backend = Aer.get_backend(self._backend_name)

        self.qi = QuantumInstance(self._backend, seed_transpiler=self._seed, seed_simulator=self._seed, shots=self._shots)