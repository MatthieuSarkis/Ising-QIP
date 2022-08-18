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

r"""
This class plays the role of a wrapper containing the information
about the quantum backend used, and the information related to the
job execution.
"""

from typing import Optional
from qiskit import Aer
from qiskit.providers.aer import AerError
from qiskit.providers.ibmq import least_busy
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
        r"""Initialize the Projected Kernel Ridge Regression from parent's init.
        Args:
            backend_type (Optional[str]): Whether to use a simulator or an IBMQ hardware.
            backend_name (Optional[str]): Name of the backend.
            mitigate (bool): Whether or not to use quantum error mitigation.
            seed (Optional[int]): Seed for the pseudo-random number generators.
            shots (Optional[int]): Number of shots for the quantum simulations.
            hub (Optional[str]): Name of the hub on IBMQ.
            group (Optional[str]): Name of the group on IBMQ.
            project (Optional[str]): Name of the project on IBMQ.
            job_name (Optional[str]): Name of the job on IBMQ.
        """

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
        self.__get_quantum_instance()

    def __get_quantum_instance(self) -> None:
        """ Get the quantum instance for quamtum experiment. """

        if self._backend_type == "IBMQ":
            from qiskit import IBMQ
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub=self._hub, group=self._group, project=self._project)
            filter = lambda x: x.configuration().n_qubits >= 7 and not x.configuration().simulator and x.status().operational == True
            self._backend = least_busy(provider.backends(filters=filter))

        else:
            self._backend = Aer.get_backend(self._backend_name)

            if self._backend_type == "GPU":
                self._backend = Aer.get_backend(self._backend_name)
                try:
                    self._backend.set_options(device='GPU')
                except AerError as e:
                    print(e)

        self.qi = QuantumInstance(self._backend, seed_transpiler=self._seed, shots=self._shots)
        #self.qi = QuantumInstance(self._backend, seed_transpiler=self._seed, seed_simulator=self._seed, shots=self._shots)