import numpy as np
from scipy.sparse import csr_matrix

from pulse.model.elements.elements_builder import build_structural_element
from pulse.model.elements.structural_element import DOF_PER_ELEMENT
from pulse.model.node import DOF_PER_NODE_STRUCTURAL


class KinematicCoupling:
    """
    Master-slave (MPC) coupling built from the coupling elements.

    Each coupling element ties its slave (last) node to its master (first) node.
    The kinematic relation is:

        u_t,s = T_JR_t * u_master                  (translations, always rigid)
        theta_s = theta_master + Q_free^T * alpha   (rotations)

    where ``Q_free`` holds the joint rotation axes whose spring stiffness is
    finite, and ``alpha`` are internal relative-rotation DOFs. The relative
    rotation about a rigid axis (spring stiffness equal to zero) is constrained
    to zero, i.e. ``(theta_s - theta_m) . e_rigid = 0``.

    This is an exact constraint in the joint frame, so it does not depend on the
    slave rotations being aligned with the global axes.
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

            # Rotational stiffness per joint axis, ordered as (e_t, e_ip, e_op).
            k_ip, k_op, k_t = element.rotation_spring_stiffness()
            stiffness_per_axis = np.array([k_t, k_ip, k_op], dtype=float)

            # A zero stiffness means a rigid axis: the relative rotation about
            # that axis is constrained to zero. Only the axes with a finite
            # stiffness remain as independent relative rotations (alpha).
            free_axes_mask = stiffness_per_axis > 0
            rotation_axes = element.joint_rotation_axes()
            free_axes = rotation_axes[free_axes_mask]  # (n_free x 3)

            couplings.append(
                dict(
                    master=element.first_node,
                    slave=element.last_node,
                    T=element.translation_constraint_matrix(),
                    free_axes=free_axes,
                    n_free=int(free_axes_mask.sum()),
                )
            )

        self.couplings = couplings

        if not couplings:
            self.active = False
            self.slave_dofs = np.array([], dtype=int)
            self.n_reduced = self.total_dof
            self.full_to_reduced = None
            self.T_global = None
            self.node_couplings = {}
            return

        self.active = True
        self.node_couplings = {c["slave"].index: c for c in couplings}

        # All slave DOFs are eliminated from the global nodal space. Translations
        # are expressed by the master through T_JR, while rotations become
        # theta_s = theta_m + Q_free^T * alpha, with alpha being new internal
        # (relative rotation) DOFs.
        slave_dofs = []
        for coupling in self.couplings:
            slave_dofs.extend(int(dof) for dof in coupling["slave"].structural_global_dof)
        self.slave_dofs = np.array(slave_dofs, dtype=int)

        kept_dofs = np.setdiff1d(np.arange(self.total_dof), self.slave_dofs)
        n_kept_global = len(kept_dofs)

        # Allocate reduced indices for the internal relative rotations (alpha).
        next_internal = n_kept_global
        for coupling in self.couplings:
            n_free = coupling["n_free"]
            coupling["alpha_reduced"] = np.arange(next_internal, next_internal + n_free, dtype=int)
            next_internal += n_free
        self.n_reduced = next_internal

        self.full_to_reduced = np.full(self.total_dof, -1, dtype=int)
        self.full_to_reduced[kept_dofs] = np.arange(n_kept_global)

        rows = list()
        cols = list()
        data = list()

        # Identity for kept DOFs: u_full = T_global * u_reduced
        for full in kept_dofs:
            rows.append(int(full))
            cols.append(int(self.full_to_reduced[full]))
            data.append(1.0)

        for coupling in self.couplings:
            master_global_dofs = coupling["master"].structural_global_dof
            slave_global_dofs = coupling["slave"].structural_global_dof
            master_reduced = self.full_to_reduced[master_global_dofs]

            if np.any(master_reduced < 0):
                raise ValueError(
                    "Master node is not part of the reduced system "
                    "(chained couplings are not supported)."
                )

            T_jr = coupling["T"]
            free_axes = coupling["free_axes"]
            alpha_reduced = coupling["alpha_reduced"]

            # Slave translations: u_t,s = T_JR * u_master
            for i in range(3):
                slave_dof = int(slave_global_dofs[i])
                for j in range(DOF_PER_NODE_STRUCTURAL):
                    rows.append(slave_dof)
                    cols.append(int(master_reduced[j]))
                    data.append(float(T_jr[i, j]))

            # Slave rotations: theta_s = theta_m + Q_free^T * alpha
            for i in range(3):
                slave_dof = int(slave_global_dofs[3 + i])

                # Rigid-body part: follows the master rotation.
                rows.append(slave_dof)
                cols.append(int(master_reduced[3 + i]))
                data.append(1.0)

                # Relative part: only the axes with a finite spring stiffness.
                for j in range(coupling["n_free"]):
                    rows.append(slave_dof)
                    cols.append(int(alpha_reduced[j]))
                    data.append(float(free_axes[j, i]))

        self.T_global = csr_matrix((data, (rows, cols)), shape=(self.total_dof, self.n_reduced))

    def element_transform(self, element_attributes):
        """
        Returns (L, red_cols) such that u_e = L @ u_e_red.

        L is (12 x n_cols) and maps the reduced element DOFs u_e_red back to the
        full element DOFs u_e. Multiplying by L changes the number of variables
        of the element (congruence transform to the reduced independent space).

        u_e = [u1,v1,w1,rx1,ry1,rz1, u2,v2,w2,rx2,ry2,rz2]
        node 1 = master (first), node 2 = slave (last)
        red_cols holds the reduced DOF index of each column of L. Those indices
        may be global nodal DOFs or internal relative-rotation DOFs (alpha).

        Regular nodes map through the identity.

        For a slave node:

            u_t,s   = T_JR[0:3, :] * u_master
            theta_s = theta_master + Q_free^T * alpha

        where ``Q_free`` holds the joint rotation axes whose spring stiffness is
        finite (n_free rows) and ``alpha`` are the internal relative-rotation
        DOFs. The axes with zero stiffness are rigid: their relative rotation is
        constrained to zero exactly (they do not appear in Q_free), so

            (theta_s - theta_master) . e_rigid = 0

        independently of the global axes.

        T_JR (rigid body relation for the slave translations):

            T_JR = [ 1  0  0    0   dz -dy ]
                   [ 0  1  0  -dz   0   dx ]
                   [ 0  0  1   dy  -dx   0  ]
                   [ 0  0  0    1    0   0  ]
                   [ 0  0  0    0    1   0  ]
                   [ 0  0  0    0    0   1  ]

        Examples:

        - Fully rigid joint (n_free = 0):
              theta_s = theta_master
              L is (12 x 6): rows 6-11 use only the 6 master columns.

        - Torsion rigid, bending flexible (n_free = 2):
              theta_s = theta_master + Q_free^T * alpha
              Q_free rows are the two bending axes. L has 6 master columns plus
              2 alpha columns. The rigid axis does not appear in Q_free, so the
              relative rotation about it is exactly zero.

        - All rotations flexible (n_free = 3):
              Q_free = Q (all joint axes). L has 6 master columns plus 3 alpha
              columns, and K_e = S^T K_s S couples theta_m and alpha through
              L^T K_e L.

        Usage:
        - u_e = L @ u_e_red maps reduced element DOFs back to full element DOFs.
        - Ke_red = L^T @ Ke @ L maps the element stiffness to the reduced space.
        - Assembly builds red_K/red_M by scattering Ke_red/Me_red with red_cols
          using n_cols = L.shape[1].
        - u_full = T_global @ u_red expands the global reduced solution back to full
          DOFs (consistent with L). Reactions use full_K with u_full.
        """
        red_cols = list()
        col_of = dict()

        def col(reduced_index):
            reduced_index = int(reduced_index)
            if reduced_index < 0:
                raise ValueError(
                    "Reduced dof is not part of the reduced system "
                    "(chained couplings are not supported)."
                )
            if reduced_index not in col_of:
                col_of[reduced_index] = len(red_cols)
                red_cols.append(reduced_index)
            return col_of[reduced_index]

        entries = list()

        # Process both nodes of the element (master first, slave second)
        for node_index, node in enumerate((element_attributes.first_node, element_attributes.last_node)):
            dof_base = DOF_PER_NODE_STRUCTURAL * node_index
            node_global_dofs = node.structural_global_dof
            coupling = self.node_couplings.get(node.index)

            if coupling is None:
                # Regular node: map each DOF directly to its reduced DOF
                for dof_local in range(DOF_PER_NODE_STRUCTURAL):
                    reduced_dof = self.full_to_reduced[node_global_dofs[dof_local]]
                    entries.append((dof_base + dof_local, col(reduced_dof), 1.0))
                continue

            master_global_dofs = coupling["master"].structural_global_dof
            master_reduced = self.full_to_reduced[master_global_dofs]
            T_jr = coupling["T"]
            free_axes = coupling["free_axes"]
            alpha_reduced = coupling["alpha_reduced"]

            # Translations: u_t,s = T_JR * u_master
            for dof_local in range(3):
                for dof_m in range(DOF_PER_NODE_STRUCTURAL):
                    entries.append(
                        (dof_base + dof_local, col(master_reduced[dof_m]), float(T_jr[dof_local, dof_m]))
                    )

            # Rotations: theta_s = theta_m + Q_free^T * alpha
            for dof_local in range(3):
                row = dof_base + 3 + dof_local

                # Rigid-body part: follows the master rotation.
                entries.append((row, col(master_reduced[3 + dof_local]), 1.0))

                # Relative part: only the axes with a finite spring stiffness.
                for j in range(coupling["n_free"]):
                    entries.append((row, col(alpha_reduced[j]), float(free_axes[j, dof_local])))

        L = np.zeros((DOF_PER_ELEMENT, len(red_cols)), dtype=float)
        for row, column, value in entries:
            L[row, column] += value

        return L, np.array(red_cols, dtype=int)

    def reduce_matrix(self, matrix):
        """Applies the congruence ``T^T A T`` (identity when inactive)."""
        if not self.active:
            return matrix
        return self.T_global.T @ matrix @ self.T_global

    def reduce_load(self, load):
        """Applies ``T^T f``, transferring slave loads onto the master."""
        if not self.active:
            return load
        return np.asarray(self.T_global.T @ load)

    def expand(self, reduced_solution):
        """Expands a reduced solution back to the full dof vector."""
        if not self.active:
            return reduced_solution
        return np.asarray(self.T_global @ reduced_solution)
