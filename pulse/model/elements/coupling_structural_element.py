
from typing import TYPE_CHECKING

import numpy as np

from pulse.model.elements.structural_element import StructuralElement

if TYPE_CHECKING:
    from pulse.model.elements.element_attributes import ElementAttributes


class CouplingStructuralElement(StructuralElement):
    """A structural element.
    This class creates a structural element from input data.

    Parameters
    ----------
    first_node : Node object
        Fist node of element.

    last_node : Node object
        Last node of element.

    index : int
        Element index.

    element_type : str, ['pipe_1', 'beam_1', 'expansion_joint', 'valve'], optional
        Element type
        Default is 'pipe_1'.

    material : Material object, optional
        Element structural material.
        Default is 'None'.

    fluid : Fluid object, optional
        Element acoustic fluid.
        Default is 'None'.

    cross_section : CrossSection object, optional
        Element cross section.
        Default is 'None'.

    loaded_forces : array, optional
        Structural forces and moments on the nodes.
        Default is zeros(12).
    """
    def __init__(self, element_attributes: "ElementAttributes", **kwargs):
        super().__init__(element_attributes, **kwargs)


    def matrices_gcs(self):
        """
        This method returns the element stiffness and mass matrices of
        the coupling element.

        Returns
        -------
        stiffness : array
            Element stiffness matrix in the global coordinate system.

        mass : array
            Element mass matrix in the global coordinate system.

        """

        stiffness = self.stiffness_matrix_coupling_element()
        mass = self.mass_matrix_coupling_element()

        return stiffness, mass


    def translation_constraint_matrix(self) -> np.ndarray:
        dx = self.delta_x
        dy = self.delta_y
        dz = self.delta_z

        self.T_JR = np.array([
            [ 1, 0, 0,   0,  dz, -dy ],
            [ 0, 1, 0, -dz,   0,  dx ],
            [ 0, 0, 1,  dy, -dx,   0 ],
            [ 0, 0, 0,   1,   0,   0 ],
            [ 0, 0, 0,   0,   1,   0 ],
            [ 0, 0, 0,   0,   0,   1 ]
            ], dtype = float)

        return self.T_JR


    @property
    def master_dofs(self) -> np.ndarray:
        """Structural dofs of the master (first) node."""
        return self.first_node.structural_global_dof

    @property
    def slave_dofs(self) -> np.ndarray:
        """Structural dofs of the slave (last) node."""
        return self.last_node.structural_global_dof


    def rotation_spring_stiffness(self) -> tuple[float, float, float]:
        """
        Rotational stiffness of the joint springs (out-of-plane bending,
        in-plane bending and torsion).

        A zero value keeps the joint rotationally rigid. A finite value adds
        the joint flexibility (shell/ovalization effects), so the relative
        rotation between J and S grows with the applied moment.
        """
        material = self.material
        cross_section = self.cross_section

        E = material.elasticity_modulus if material else 0
        Iyy = cross_section.second_moment_area_y if cross_section else 0
        Izz = cross_section.second_moment_area_z if cross_section else 0
        Iyz = cross_section.second_moment_area_yz if cross_section else 0

        k_ip = 0
        k_op = 0
        k_t = 0

        return k_ip, k_op, k_t


    def rotation_spring_matrix(self) -> np.ndarray:
        """
        Rotational-spring stiffness of the joint, in the slave node coordinate
        system: ``diag([0, 0, 0, k_t, k_ip, k_op])``. The translational entries
        are zero (always rigid); a finite rotational entry adds flexibility.
        """
        k_ip, k_op, k_t = self.rotation_spring_stiffness()
        return np.diag([0.0, 0.0, 0.0, k_t, k_ip, k_op])

    def eliminated_mask(self) -> np.ndarray:
        """
        Which slave dofs are rigidly eliminated (True) and which stay as
        unknowns (False). Translations are always rigid; rotations become
        flexible once their spring stiffness is finite.
        """
        k_ip, k_op, k_t = self.rotation_spring_stiffness()
        return np.array([True, True, True, k_t == 0, k_ip == 0, k_op == 0], dtype=bool)


    def get_reduced_mass_matrix_element(self):
        return 0.
