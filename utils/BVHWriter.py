import os
import json
import numpy as np
import numpy.linalg as nl

class BVHData(object):
    def __init__(self, name, offset=(0.0, 0.0, 0.0), scale=1.0):
        self.name = name
        self.offset = offset
        self.scale = scale
        self.children = []
        self.euler = []


class BVHWriter(object):
    def __init__(self, hand_model='./utils/hand2_l_all_uv.json'):
        with open(hand_model) as f:
            model = json.load(f)
        self.MTa = np.array(model['MTa'])
        self.update_inds = np.array(model['update_inds'])
        self.parents = np.array(model['parents'])


    def ParsePath(self, path, outname='output.bvh'):
        # parse every file in the path
        assert os.path.isdir(path)
        filelist = [_ for _ in os.listdir(path) if _.endswith('.json')]
        filelist.sort()

        first = True

        self.trans = []
        for filename in filelist:
            filename = os.path.join(path, filename)
            with open(filename) as f:
                data = json.load(f)

            if first:
                first = False
                firstCoeff = np.array(data['coeff'])
                self.get_hierarchy(firstCoeff)

            pose_array = np.array(data['pose'])
            self.get_dynamic(pose_array)
            self.trans.append(np.array(data['trans']))

        self.WriteBVH(path, outname, len(filelist))


    def get_hierarchy(self, coeff):
        # maintain a dict containing pointer to data
        dict_data = {}

        i = 0
        idj = self.update_inds[i]
        offset = self.MTa[4*idj:4*idj+3, 3]
        scale = coeff[idj, 0]

        root = BVHData('Wrist', (offset[0], offset[1], offset[2]), scale)
        dict_data[idj] = root

        bone_order = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']

        for i in range(1, 21):
            idj = self.update_inds[i]
            idp = self.parents[idj] # parent id
            offset = self.MTa[4*idj:4*idj+3, 3] * dict_data[idp].scale
            scale = dict_data[idp].scale * coeff[idj, 0] # multiplication of scales
            name = bone_order[(i-1)%len(bone_order)] + str(int((i-1)/5))

            this_data = BVHData(name, (offset[0], offset[1], offset[2]), scale)
            dict_data[idj] = this_data
            dict_data[idp].children.append(this_data)

        self.root = root
        self.dict_data = dict_data


    def get_dynamic(self, pose):
        # Traverse the tree
        i = 0
        idj = self.update_inds[i]
        poseR = self.AngleAxisToRotationMatrix(pose[idj, :3])
        det = nl.det(self.MTa[4*idj:4*idj+3, :3]) ** (1./3)
        toparentR = self.MTa[4*idj:4*idj+3, :3] / det
        R = np.dot(toparentR, poseR)
        self.root.euler.append(self.RotationMatrixToEulerAngle(R))

        for i in range(1, 21):
            idj = self.update_inds[i]
            poseR = self.EulerAngleToRotationMatrix(pose[idj, :3])
            toparentR = self.MTa[4*idj:4*idj+3, :3]
            R = np.dot(toparentR, poseR)
            self.dict_data[idj].euler.append(self.RotationMatrixToEulerAngle(R))


    def WriteBVH(self, path, filename, num_frame):
        fullname = os.path.join(path, filename)
        self.dynamic_str = ['' for _ in range(num_frame)]
        f = open(fullname, 'w')
        f.write('HIERARCHY\n')
        # write hierarchy data
        self.WriteData(self.root, f, 0) # call this function recursively
        # write dynamic data
        f.write('MOTION\n')
        f.write('Frames: {}\n'.format(num_frame))
        f.write('Frame Time: 0.03333\n')

        for d in self.dynamic_str:
            f.write(d + '\n')

        f.close()


    def WriteData(self, data, f, depth):
        """ This function is to be called recursively. 'depth' is used to compute the indent
        """
        if depth == 0:
            f.write('ROOT ')
            f.write(data.name + '\n')
        elif len(data.children) > 0:
            f.write(depth * '\t' + 'JOINT ')
            f.write(data.name + '\n')
        else:
            f.write(depth * '\t' + 'End Site\n')

        f.write(depth * '\t' + '{\n')

        f.write((depth+1) * '\t' + 'OFFSET ')
        f.write('{:.5f} {:.5f} {:.5f}\n'.format(data.offset[0], data.offset[1], data.offset[2]))

        if len(data.children) > 0:
            f.write((depth+1) * '\t' + 'CHANNELS {} '.format(6 if depth == 0 else 3))
            if depth == 0:
                f.write('Xposition Yposition Zposition ')
                for iframe, trans in enumerate(self.trans):
                    self.dynamic_str[iframe] += '{:.5f} {:.5f} {:.5f}'.format(trans[0], trans[1], trans[2])
            f.write('Zrotation Yrotation Xrotation\n')
            for iframe, euler_angle in enumerate(data.euler):
                self.dynamic_str[iframe] += ' {:.5f} {:.5f} {:.5f}'.format(euler_angle[2], euler_angle[1], euler_angle[0])

            for child in data.children:
                self.WriteData(child, f, depth+1) # call this function recursively

        f.write(depth * '\t' + '}\n')


    @classmethod
    def AngleAxisToRotationMatrix(cls, angle_axis):
        """ angle_axis is a 3d vector whose direction points to the rotation axis and whose norm is the angle (in radians)
            Refer to ceres/rotation.h for more details
        """
        assert angle_axis.shape == (3,)
        theta = nl.norm(angle_axis)
        if theta > 0.0:
            x, y, z = angle_axis / theta
            cos = np.cos(theta)
            sin = np.sin(theta)

            R = np.zeros((3,3), dtype=np.float64) # ceres uses column major
            R[0, 0] = cos + x * x * (1. - cos)
            R[1, 0] = sin * z + x * y * (1. - cos)
            R[2, 0] = -sin * y + x * z * (1. - cos)
            R[0, 1] = x * y * (1. - cos) - z * sin
            R[1, 1] = cos + y * y * (1. - cos)
            R[2, 1] = x * sin + y * z * (1 - cos)
            R[0, 2] = y * sin + x * z * (1 - cos) 
            R[1, 2] = -x * sin + y * z * (1 - cos)
            R[2, 2] = cos + z * z * (1 - cos)

        else:
            R = np.identity(3, dtype=np.float64)
            R[1, 0] = -angle_axis[2]
            R[2, 0] = angle_axis[1]
            R[0, 1] = angle_axis[2]
            R[1, 2] = -angle_axis[0]
            R[2, 0] = -angle_axis[1]
            R[2, 1] = angle_axis[0]

        return R


    @classmethod
    def EulerAngleToRotationMatrix(cls, euler_angle):
        """ This function computes the rotation matrix corresponding to Euler Angle (x, y, z) R_z * R_y * R_x (consistent with Ceres).
            (x, y, z) in degrees.
        """
        # z -> 1, y -> 2, x -> 1
        assert euler_angle.shape == (3,)
        deg = euler_angle * np.pi / 180
        c3, c2, c1 = np.cos(deg)
        s3, s2, s1 = np.sin(deg)
        R = np.zeros((3,3), dtype=np.float64)
        R[0, 0] = c1 * c2
        R[0, 1] = -s1 * c3 + c1 * s2 * s3
        R[0, 2] = s1 * s3 + c1 * s2 * c3
        R[1, 0] = s1 * c2
        R[1, 1] = c1 * c3 + s1 * s2 * s3
        R[1, 2] = -c1 * s3 + s1 * s2 * c3
        R[2, 0] = -s2
        R[2, 1] = c2 * s3
        R[2, 2] = c2 * c3
        return R


    @classmethod
    def RotationMatrixToEulerAngle(cls, R):
        """ A rotation matrix converted to Euler angles (x, y, z) in degrees
        """
        assert R.shape == (3, 3)
        if R[2, 0] < 1.0:
            if R[2, 0] > -1.0:
                euler_angle = np.array([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(R[2, 0]), np.arctan2(R[1, 0], R[0, 0])])
            else: # solution not unique
                euler_angle = np.array([0.0, np.pi/2, -np.arctan2(R[1, 2], R[1, 1])])
        else:
            euler_angle = np.array([0.0, -np.pi/2, -np.arctan2(R[1, 2], R[1, 1])])
        return euler_angle * 180 / np.pi


if __name__ == '__main__':
    w = BVHWriter()    
    w.ParsePath('/home/donglaix/Documents/Experiments/output_3d_direct1/')

    # R = w.AngleAxisToRotationMatrix(np.array([0., np.pi/2, 0]))
    # euler_angle = np.array([1.0, -0.5, 0.8])
    # R = w.EulerAngleToRotationMatrix(euler_angle)
    # euler_angle_ = w.RotationMatrixToEulerAngle(R)