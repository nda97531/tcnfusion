import numpy as np


def cos_angle_3d(vectorA, vectorB):
    """
    calculate cosine value of two 3D vectors
    """

    dot_prod = np.dot(vectorA, vectorB)
    leng_a = np.sqrt((vectorA ** 2).sum())
    leng_b = np.sqrt((vectorB ** 2).sum())
    cos = dot_prod / (leng_a * leng_b)

    if np.isnan(cos):
        print('cos nan')
        return None

    return cos


def remove_joints(window, deljoints=np.array([2, 4, 8, 7, 11, 1, 12, 16, 15, 19])):
    '''
    remove specified joints from skeleton data
    :param window: shape (timesteps, number of joints, 3)
    :return: shape (timesteps, number of joints, 3)
    '''

    window = np.delete(window, deljoints, -2)

    return window


Oy_vector = np.array([0, 1, 0])


def process_skeleton_sequence(ske_window, select_representative_joints=True, add_angle_features=True):
    '''
    Select representative joints and Add angle features
    :param ske_window: (timesteps, 60)
    :return: (timesteps, new number of features)
    '''

    if (not select_representative_joints) and (not add_angle_features):
        return ske_window

    ske_window = ske_window.reshape((ske_window.shape[0], -1, 3))

    all_frame_angles = []
    for j in range(ske_window.shape[0]):  # each frame

        head_joint = ske_window[j, 4 - 1]
        hip_center_joint = ske_window[j, 1 - 1]

        shoulder_left = ske_window[j, 5 - 1]
        shoulder_right = ske_window[j, 9 - 1]

        spine_vector = head_joint - hip_center_joint
        shoudler_vector = shoulder_right - shoulder_left

        # cos of angle between spine and Oy axis
        cos_spine = cos_angle_3d(Oy_vector, spine_vector)
        if (cos_spine is None):
            cos_spine = 0.

        # cos of angle between shoulder and Oy axis
        cos_shoudler = cos_angle_3d(Oy_vector, shoudler_vector)
        if (cos_shoudler is None):
            cos_shoudler = 0.

        all_frame_angles.append(np.array([cos_spine, cos_shoudler]))
    all_frame_angles = np.array(all_frame_angles)

    # remove joints
    if select_representative_joints:
        ske_window = remove_joints(ske_window)

    ske_window = ske_window.reshape((ske_window.shape[0], -1))

    # add angles
    if add_angle_features:
        ske_window = np.concatenate([ske_window, all_frame_angles], -1)

    return ske_window
