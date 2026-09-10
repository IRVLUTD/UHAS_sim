#!/usr/bin/env python
"""trimesh.transformations with ROS / tf xyzw quaternion convention.

``trimesh.transformations`` uses ``[w, x, y, z]``. ROS ``tf.transformations``
(and this package's ``sphere_cik.json`` / Isaac consumers) use ``[x, y, z, w]``.
Import this module as ``tf`` in process_urdf so every quaternion in the pipeline
is xyzw.
"""
import numpy as np
import trimesh.transformations as _tf


def _wxyz_to_xyzw(q):
    q = np.asarray(q)
    return np.concatenate((q[..., 1:4], q[..., 0:1]), axis=-1)


def _xyzw_to_wxyz(q):
    q = np.asarray(q)
    return np.concatenate((q[..., 3:4], q[..., 0:3]), axis=-1)


def quaternion_from_matrix(matrix, isprecise=False):
    return _wxyz_to_xyzw(_tf.quaternion_from_matrix(matrix, isprecise=isprecise))


def quaternion_matrix(quaternion):
    return _tf.quaternion_matrix(_xyzw_to_wxyz(quaternion))


def quaternion_from_euler(ai, aj, ak, axes="sxyz"):
    return _wxyz_to_xyzw(_tf.quaternion_from_euler(ai, aj, ak, axes=axes))


def euler_from_quaternion(quaternion, axes="sxyz"):
    return _tf.euler_from_quaternion(_xyzw_to_wxyz(quaternion), axes=axes)


def quaternion_about_axis(angle, axis):
    return _wxyz_to_xyzw(_tf.quaternion_about_axis(angle, axis))


def random_quaternion(rand=None, num=1):
    return _wxyz_to_xyzw(_tf.random_quaternion(rand=rand, num=num))


def quaternion_conjugate(quaternion):
    return _wxyz_to_xyzw(_tf.quaternion_conjugate(_xyzw_to_wxyz(quaternion)))


def quaternion_inverse(quaternion):
    return _wxyz_to_xyzw(_tf.quaternion_inverse(_xyzw_to_wxyz(quaternion)))


def quaternion_multiply(quaternion1, quaternion0):
    return _wxyz_to_xyzw(
        _tf.quaternion_multiply(_xyzw_to_wxyz(quaternion1), _xyzw_to_wxyz(quaternion0))
    )


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    return _wxyz_to_xyzw(
        _tf.quaternion_slerp(
            _xyzw_to_wxyz(quat0),
            _xyzw_to_wxyz(quat1),
            fraction,
            spin=spin,
            shortestpath=shortestpath,
        )
    )


def quaternion_real(quaternion):
    return _tf.quaternion_real(_xyzw_to_wxyz(quaternion))


def quaternion_imag(quaternion):
    return _tf.quaternion_imag(_xyzw_to_wxyz(quaternion))


def __getattr__(name):
    return getattr(_tf, name)
