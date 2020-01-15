import torch
import numpy as np


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    assert q.is_cuda == v.is_cuda

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


# note(yanzhey): Corrected the expmap implementation in Quaternet
# ref: https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm
def expmap_to_quaternion(e):
    # Convert axis-angle rotations (aka exponential maps) to quaternions.
    # Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    # Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    # Returns a tensor of shape (*, 4).
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)
    
    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    e = e / (theta + 1e-5)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = np.sin(0.5 * theta) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


class ForwardKinematics:
    def __init__(self, offsets, parents):
        self.offsets = torch.from_numpy(offsets.astype(np.float32))
        self.parents = parents
        self.num_joint = len(parents)
        self.has_children = np.zeros(len(parents)).astype(bool)
        for i, parent in enumerate(parents):
            if parent != -1:
                self.has_children[parent] = True

    def run_local(self, rotations):
        """
        Return position in local reference frame
        :param rotations: [N, L, J, 4]
        :return: 
        """
        positions_world = []
        rotations_world = []
        N = rotations.shape[0]
        L = rotations.shape[1]
        J = rotations.shape[2]

        identity_qs = torch.cat((torch.ones((N, L, 1)), torch.zeros((N, L, 3))), dim=-1)
        identity_ts = torch.zeros((N, L, 3))
        if rotations.is_cuda:
            identity_qs = identity_qs.cuda()
            identity_ts = identity_ts.cuda()
            expanded_offsets = self.offsets.cuda().expand(N, L, J, 3)
        else:
            expanded_offsets = self.offsets.expand(N, L, J, 3)
        for i in range(J):
            if self.parents[i] == -1:
                positions_world.append(identity_ts)
                rotations_world.append(identity_qs)
            else:
                pos = qrot(rotations_world[self.parents[i]],
                           expanded_offsets[:, :, i]) + positions_world[self.parents[i]]
                positions_world.append(pos)
                if self.has_children[i]:
                    rotations_world.append(qmul(rotations_world[self.parents[i]], rotations[:, :, i]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        # position_word (J, [N, L, 3])
        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

    def run(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4

        positions_world = []
        rotations_world = []

        expanded_offsets = self.offsets.expand(rotations.shape[0],  rotations.shape[1],
                                               self.offsets.shape[0], self.offsets.shape[1])

        # Parallelize along the batch and time dimensions
        for i in range(self.offsets.shape[0]):
            if self.parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(qrot(rotations_world[self.parents[i]], expanded_offsets[:, :, i]) + positions_world[self.parents[i]])
                if self.has_children[i]:
                    rotations_world.append(qmul(rotations_world[self.parents[i]], rotations[:, :, i]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

    def run_mat(self, rot_local):
        """
        :param rot_local: [N, L, J, 3, 3]
        :return: [N, L, J, 3]
        """
        N = rot_local.shape[0]
        L = rot_local.shape[1]
        J = rot_local.shape[2]
        rot_local = rot_local.view(-1, J, 3, 3)
        rot_global = list()         # (J, [N x L, 3, 3])
        pos_global = list()         # (J, [N x L, 3, 1])

        if rot_local.is_cuda:
            identity_pos = torch.zeros((N * L, 3, 1)).cuda()
            identity_rot = torch.from_numpy(np.tile(np.eye(3)[None, :, :], (N * L, 1, 1)).astype(np.float32)).cuda()
            expanded_offsets = self.offsets.cuda().expand(N * L, J, 3)[..., None]
        else:
            identity_pos = torch.zeros((N * L, 3, 1))
            identity_rot = torch.from_numpy(np.tile(np.eye(3)[None, :, :], (N * L, 1, 1)).astype(np.float32))
            expanded_offsets = self.offsets.expand(N * L, J, 3)[..., None]

        for i in range(J):
            if self.parents[i] == -1:
                pos_global.append(identity_pos)
                # rot_global.append(identity_rot)
                rot_global.append(rot_local[:, i])
            else:
                pos = torch.bmm(rot_global[self.parents[i]], expanded_offsets[:, i]) + pos_global[self.parents[i]]
                pos_global.append(pos)
                if self.has_children[i]:
                    rot_global.append(torch.bmm(rot_global[self.parents[i]], rot_local[:, i]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rot_global.append(None)
        pos = torch.stack(pos_global, dim=1)[..., 0]    # [NxL, J, 3]
        pos = pos.view(N, L, J, 3)
        return pos
