import numpy as np

def project(joints, K, R=None, t=None, distCoef=None):
    """ Perform Projection.
        joints: N * 3
    """ 
    x = joints.T
    if R is not None:
        x = np.dot(R, x)
    if t is not None:
        x = x + t.reshape(3, 1)

    xp = x[:2, :] / x[2, :]

    if distCoef is not None:
        X2 = xp[0, :] * xp[0, :]
        Y2 = xp[1, :] * xp[1, :]
        XY = X2 * Y2
        R2 = X2 + Y2
        R4 = R2 * R2
        R6 = R4 * R2

        dc = distCoef
        radial = 1.0 + dc[0] * R2 + dc[1] * R4 + dc[4] * R6
        tan_x = 2.0 * dc[2] * XY + dc[3] * (R2 + 2.0 * X2)
        tan_y = 2.0 * dc[3] * XY + dc[2] * (R2 + 2.0 * Y2)

        xp[0, :] = radial * xp[0, :] + tan_x
        xp[1, :] = radial * xp[1, :] + tan_y

    pt = np.dot(K[:2, :2], xp) + K[:2, 2].reshape((2, 1))

    return pt.T, x.T

