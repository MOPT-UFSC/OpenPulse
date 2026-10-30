
from typing import TYPE_CHECKING

import numpy as np

from pulse.model.elements.structural_element import DOF_PER_ELEMENT, DOF_PER_NODE_STRUCTURAL, StructuralElement

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
    def coupling_info(self) -> dict | None:
        """Header and branch data of the coupling (cross-section and material)."""
        return self.element_attributes.coupling_info


    @property
    def header_direction(self) -> np.ndarray | None:
        """
        Unit vector of the header (main pipe) axis at the junction, in global
        coordinates. It is stored in ``coupling_info["header"]["direction"]``
        when the T-junction is created. Returns ``None`` when unavailable.
        """
        if self.coupling_info is None:
            return None

        header = self.coupling_info.get("header", {})
        direction = header.get("direction")
        if direction is None:
            return None

        direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None

        return direction / norm


    def joint_rotation_axes(self) -> np.ndarray:
        """
        Orthonormal triad of the joint rotational axes, in global coordinates.

        Returns a 3x3 matrix ``Q`` whose rows are ``(e_t, e_ip, e_op)``:

            e_t  = branch axis (element axis, master -> subordinate) -> torsion
            e_ip = header axis orthogonalized to the branch     -> in-plane
            e_op = e_t x e_ip (normal to the header/branch plane) -> out-of-plane

        When the header direction is unavailable (or the header is parallel to
        the branch), it falls back to the element local frame, which reproduces
        the previous behavior.
        """
        branch_vector = self.last_node.coordinates - self.first_node.coordinates
        branch_norm = np.linalg.norm(branch_vector)

        local_frame = self.element_rotation_matrix[3:6, 3:6].copy()
        if branch_norm == 0:
            return local_frame

        e_t = branch_vector / branch_norm

        header_direction = self.header_direction
        if header_direction is None:
            return local_frame

        e_ip = header_direction - np.dot(header_direction, e_t) * e_t
        e_ip_norm = np.linalg.norm(e_ip)
        if e_ip_norm <= 1e-8:
            return local_frame

        e_ip = e_ip / e_ip_norm
        e_op = np.cross(e_t, e_ip)

        return np.vstack((e_t, e_ip, e_op))


    def rotation_spring_stiffness(self, rigid=False, standard="ASME") -> tuple[float, float, float]:
        """
        Rotational stiffness of the joint springs, returned as
        ``(k_ip, k_op, k_t)``.

        The axes are defined by ``joint_rotation_axes``:

            k_t  -> torsion, about the branch axis
            k_ip -> in-plane axis, along the header direction
            k_op -> out-of-plane axis, normal to the (header, branch) plane

        A zero value keeps the joint rotationally rigid (the subordinate rotation is
        eliminated). A finite value adds the joint flexibility (shell and
        ovalization effects), so the relative rotation between master and subordinate
        grows with the applied moment.

        ``standard`` selects the flexibility model used to evaluate the
        stiffness. Only ``"ASME"`` is implemented for now.
        """
        if rigid or self.coupling_info is None:
            return 0.0, 0.0, 0.0

        standard = (standard or "ASME").upper()
        if standard == "ASME":
            return self._asme_rotation_spring_stiffness()

        raise ValueError(f"Unsupported joint flexibility standard: {standard!r}")


    def _asme_rotation_spring_stiffness(self) -> tuple[float, float, float]:
        """
        ASME-based rotational spring stiffness. The in-plane and out-of-plane
        stiffness are kept equal for now (only the header geometry is used);
        the torsion is kept at 1.0 as the reference value.
        """
        header = self.coupling_info.get("header", {})
        cross_section_info = header.get("cross_section_info") or {}
        section_parameters = cross_section_info.get("section_parameters") or []

        if len(section_parameters) < 2:
            return 0.0, 0.0, 0.0

        header_diameter = section_parameters[0]
        header_thickness = section_parameters[1]

        avg_header_radius = (header_diameter - header_thickness) / 2
        if avg_header_radius <= 0 or header_thickness <= 0:
            return 0.0, 0.0, 0.0

        h = header_thickness / avg_header_radius  # flexibility characteristic

        k_ip = 0.9 / h ** (2 / 3)
        k_op = k_ip
        k_t = 0 # rigid, as per ASME recommendation

        return k_ip, k_op, k_t


    def rotation_spring_matrix(self, standard="ASME") -> np.ndarray:
        """
        Rotational-spring stiffness of the joint (6x6), in the global
        coordinate system.

        The axes are the rows of ``Q``: ``(e_t, e_ip, e_op)``, where ``e_t`` is
        the branch axis, ``e_ip`` is the header direction and ``e_op`` is the
        normal to the (header, branch) plane.

        Each stiffness is associated with its own axis:

            k_t  -> e_t  (torsion, about the branch)
            k_ip -> e_ip (axis along the header)
            k_op -> e_op (normal to the header/branch plane)

        The local stiffness is rotated to global with ``K = Q^T diag Q``. The
        translational entries are zero (always rigid); a finite rotational entry
        adds flexibility.
        """
        k_ip, k_op, k_t = self.rotation_spring_stiffness(standard=standard)

        K = np.zeros((DOF_PER_NODE_STRUCTURAL, DOF_PER_NODE_STRUCTURAL), dtype=float)
        if k_t == 0 and k_ip == 0 and k_op == 0:
            return K

        Q = self.joint_rotation_axes()
        K[3:6, 3:6] = Q.T @ np.diag([k_t, k_ip, k_op]) @ Q
        return K


    def stiffness_matrix_coupling_element(self) -> np.ndarray:
        """
        Element stiffness matrix (12x12) of the rotational joint spring, in the
        global coordinate system. It penalizes the relative rotation between the
        subordinate (last) and the master (first) node::

            0.5 * (theta_s - theta_m)^T K_s (theta_s - theta_m)
        """
        K = self.rotation_spring_matrix()

        selector = np.zeros((DOF_PER_NODE_STRUCTURAL, DOF_PER_ELEMENT), dtype=float)
        selector[3:6, 3:6] = -np.eye(3)
        selector[3:6, 9:12] = np.eye(3)

        return selector.T @ K @ selector


    def mass_matrix_coupling_element(self) -> np.ndarray:
        """The coupling element carries no distributed mass."""
        return np.zeros((DOF_PER_ELEMENT, DOF_PER_ELEMENT), dtype=float)
