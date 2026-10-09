import numpy as np
from scipy.sparse import csr_matrix

from pulse.model.elements.elements_builder import build_structural_element
from pulse.model.elements.structural_element import DOF_PER_ELEMENT
from pulse.model.node import DOF_PER_NODE_STRUCTURAL


class KinematicCoupling:
    """
    Master-slave (MPC) coupling built from the coupling elements.

    Each coupling element ties its slave (last) node to its master (first) node
    through the rigid-body relation ``u_slave = T_JR @ u_master``. The slave
    dofs are eliminated from the global system and expressed as a function of
    the master dofs (MPC / kinematic transformation method).
    """

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.total_dof = DOF_PER_NODE_STRUCTURAL * len(preprocessor.nodes)
        self._build()

    def _build(self):

        couplings = []
        for element_attributes in self.preprocessor.elements_attributes.values():
            if element_attributes.structural_element_type != "coupling_element":
                continue

            element = build_structural_element(element_attributes)
            couplings.append(
                dict(
                    master=element.first_node,
                    slave=element.last_node,
                    T=element.translation_constraint_matrix(),
                    spring=element.rotation_spring_matrix(),
                    mask=element.eliminated_mask(),
                )
            )

        self.couplings = couplings

        if not couplings:
            self.active = False
            self.slave_dofs = np.array([], dtype=int)
            self.n_reduced = self.total_dof
            self.full_to_reduced = None
            self.T_mpc = None
            self.node_couplings = {}
            return

        self.active = True
        self.node_couplings = {c["slave"].index: c for c in couplings}

        slave_dofs = [int(dof) for c in couplings for dof in c["slave"].structural_global_dof[c["mask"]]]
        self.slave_dofs = np.array(slave_dofs, dtype=int)

        kept_dofs = np.setdiff1d(np.arange(self.total_dof), self.slave_dofs)
        self.n_reduced = len(kept_dofs)

        self.full_to_reduced = np.full(self.total_dof, -1, dtype=int)
        self.full_to_reduced[kept_dofs] = np.arange(self.n_reduced)

        rows = list()
        cols = list()
        data = list()

        for full in kept_dofs:
            rows.append(int(full))
            cols.append(int(self.full_to_reduced[full]))
            data.append(1.0)

        for c in couplings:
            master = c["master"].structural_global_dof
            slave = c["slave"].structural_global_dof
            for i, slave_dof in enumerate(slave):
                if not c["mask"][i]:
                    continue
                for j, master_dof in enumerate(master):
                    rows.append(int(slave_dof))
                    cols.append(int(self.full_to_reduced[master_dof]))
                    data.append(c["T"][i, j])

        self.T_mpc = csr_matrix((data, (rows, cols)), shape=(self.total_dof, self.n_reduced))

    def element_transform(self, element_attributes):
        """
        Returns ``(L, dofs_red)`` such that ``u_e = L @ u_e_red``, with the
        independent dofs ``u_e_red`` placed at ``dofs_red`` in the reduced
        global vector.

        Kept nodes map through the identity, while fully-eliminated slave
        nodes map through their rigid-body transformation.

        The element transformation matrix ``L`` is block-diagonal (12x12)::

            L = | B1   0  |
                | 0    B2 |

            B_i = I6      if node i is a regular node
            B_i = T_JR    if node i is a slave node

        T_JR (slave node expressed in terms of the master)::

            T_JR = [ 1  0  0    0   dz -dy ]
                   [ 0  1  0  -dz    0  dx ]
                   [ 0  0  1   dy  -dx   0 ]
                   [ 0  0  0    1    0   0 ]
                   [ 0  0  0    0    1   0 ]
                   [ 0  0  0    0    0   1 ]

        Expanded (second node is the slave)::

            L = | I6    0   |
                | 0    T_JR  |

        """
        L = np.eye(DOF_PER_ELEMENT, dtype=float)
        dofs_red = np.empty(DOF_PER_ELEMENT, dtype=int)

        for i, node in enumerate((element_attributes.first_node, element_attributes.last_node)):
            sl = slice(DOF_PER_NODE_STRUCTURAL * i, DOF_PER_NODE_STRUCTURAL * (i + 1))
            coupling = self.node_couplings.get(node.index)

            if coupling is None:
                dofs_red[sl] = self.full_to_reduced[node.structural_global_dof]
                continue

            if not coupling["mask"].all():
                raise NotImplementedError("Partial elimination (rotational springs) is not implemented yet.")

            L[sl, sl] = coupling["T"]
            dofs_red[sl] = self.full_to_reduced[coupling["master"].structural_global_dof]

        return L, dofs_red

    def element_spring(self, element_attributes):
        return

    def reduce_matrix(self, matrix):
        """Applies the congruence ``T^T A T`` (identity when inactive)."""
        if not self.active:
            return matrix
        return self.T_mpc.T @ matrix @ self.T_mpc

    def reduce_load(self, load):
        """Applies ``T^T f``, transferring slave loads onto the master."""
        if not self.active:
            return load
        return np.asarray(self.T_mpc.T @ load)

    def expand(self, reduced_solution):
        """Expands a reduced solution back to the full dof vector."""
        if not self.active:
            return reduced_solution
        return np.asarray(self.T_mpc @ reduced_solution)
