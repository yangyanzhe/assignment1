""" Import FBX

Load fbx files and return

Author: Yanzhe Yang
Time: 2019 July 19
"""
from __future__ import print_function

import fbx
import numpy as np
from Quaternions import Quaternions
from Animation import Animation


def get_scene_nodes_recursive(node, parent_id, nodes, parents, joint_names):
    nodes.append(node)
    parents.append(parent_id)
    joint_names.append(node.GetName())
    node_id = len(nodes) - 1
    for i in range(node.GetChildCount()):
        get_scene_nodes_recursive(node.GetChild(i),
                                  node_id,
                                  nodes,
                                  parents,
                                  joint_names)


def get_nodes(root_node):
    nodes = []
    parents = []
    joint_names = []

    nodes.append(root_node)
    parents.append(-1)
    joint_names.append(root_node.GetName())
    for i in range(root_node.GetChildCount()):
        get_scene_nodes_recursive(root_node.GetChild(i),
                                  0,
                                  nodes,
                                  parents,
                                  joint_names)

    return nodes, parents, joint_names


def get_node(node, joint_name, transform):
    if node.GetName() == joint_name:
        return node, transform

    # copy matrix considering python passes pointer
    transform = fbx.FbxAMatrix(node.EvaluateLocalTransform() * transform)
    for i in range(node.GetChildCount()):
        parent = node.GetChild(i)
        found_node, transform = get_node(parent, joint_name, transform)
        if found_node:
            return found_node, transform
    return None, transform


def get_node_dict(scene, root_name):
    pre_transform = fbx.FbxAMatrix()
    pre_transform.SetIdentity()
    pelvis, pre_transform = get_node(scene.GetRootNode(), root_name, pre_transform)
    nodes, parents, joint_names = get_nodes(pelvis)
    rs = dict()
    for i in range(len(joint_names)):
        rs[joint_names[i]] = nodes[i]
    return rs


def euler_angle_to_quaternions(x, y, z):
    transform = fbx.FbxAMatrix()
    transform.SetIdentity()
    transform.SetR(fbx.FbxVector4(x, y, z, 0.0))        # euler angle is 'eEulerXYZ' here, i.e. apply Z first
    q = transform.GetQ()
    return q


def fbxQuaternionToArray(q):
    """
    :param q: (F, J, fbx.Quaternion)
    :return: (F, J, 4)
    """
    rs = list()
    nframe = len(q)
    njoint = len(q[0])

    for i in range(nframe):
        rs_node = list()
        for j in range(njoint):
            # in fbx sdk, q is stored as [x, y, z, w]
            # in utility, q is stored at [w, x, y, z]
            rs_node.append([q[i, j][3], q[i, j][0], q[i, j][1], q[i, j][2]])
        rs.append(rs_node)
    return np.array(rs)


def get_anim(layer, nodes):
    qs = list()             # (J, F, 4)
    ts = list()             # (J, F, 3)
    offsets = list()        # (J, 3)

    for j in range(len(nodes)):
        t = nodes[j].LclTranslation.Get()
        offsets.append([t[0], t[1], t[2]])

    root_rx = nodes[0].LclRotation.GetCurve(layer, "X", True)
    nkeys = root_rx.KeyGetCount()

    for j in range(len(nodes)):
        node = nodes[j]
        tx = node.LclTranslation.GetCurve(layer, "X", True)
        ty = node.LclTranslation.GetCurve(layer, "Y", True)
        tz = node.LclTranslation.GetCurve(layer, "Z", True)
        rx = node.LclRotation.GetCurve(layer, "X", True)
        ry = node.LclRotation.GetCurve(layer, "Y", True)
        rz = node.LclRotation.GetCurve(layer, "Z", True)

        node_t = list()
        node_q = list()

        if j == 0:
            if tx.KeyGetCount() != nkeys:
                raise Exception("tx.KeyGetCount() != nkeys: %d != %d" % (tx.KeyGetCount(), nkeys))
            if ty.KeyGetCount() != nkeys:
                raise Exception("ty.KeyGetCount() != nkeys: %d != %d" % (ty.KeyGetCount(), nkeys))
            if tz.KeyGetCount() != nkeys:
                raise Exception("tz.KeyGetCount() != nkeys: %d != %d" % (tz.KeyGetCount(), nkeys))
            if rx.KeyGetCount() != nkeys:
                raise Exception("rx.KeyGetCount() != nkeys: %d != %d" % (rx.KeyGetCount(), nkeys))
            if ry.KeyGetCount() != nkeys:
                raise Exception("ry.KeyGetCount() != nkeys: %d != %d" % (ry.KeyGetCount(), nkeys))
            if rz.KeyGetCount() != nkeys:
                raise Exception("rz.KeyGetCount() != nkeys: %d != %d" % (rz.KeyGetCount(), nkeys))

            for i in range(nkeys):
                node_t.append([tx.KeyGetValue(i), ty.KeyGetValue(i), tz.KeyGetValue(i)])
                node_q.append(euler_angle_to_quaternions(rx.KeyGetValue(i), ry.KeyGetValue(i), rz.KeyGetValue(i)))
        else:
            for i in range(nkeys):
                if rx.KeyGetCount() == 0:
                    x = 0
                elif rx.KeyGetCount() == nkeys:
                    x = rx.KeyGetValue(i)
                else:
                    x, _ = rx.Evaluate(root_rx.KeyGetTime(i))

                if ry.KeyGetCount() == 0:
                    y = 0
                elif ry.KeyGetCount() == nkeys:
                    y = ry.KeyGetValue(i)
                else:
                    y, _ = ry.Evaluate(root_rx.KeyGetTime(i))

                if rz.KeyGetCount() == 0:
                    z = 0
                elif rz.KeyGetCount() == nkeys:
                    z = rz.KeyGetValue(i)
                else:
                    z, _ = rz.Evaluate(root_rx.KeyGetTime(i))

                node_t.append(offsets[j])
                node_q.append(euler_angle_to_quaternions(x, y, z))

        qs.append(node_q)
        ts.append(node_t)

    qs = np.array(qs)
    ts = np.array(ts)
    qs = np.swapaxes(qs, 0, 1)  # (F, J, fbxQuaternion)
    ts = np.swapaxes(ts, 0, 1)  # (F, J, 3)
    return np.array(offsets), np.array(qs), np.array(ts)


def fbxVec4ToList(vec):
    rs = np.zeros(3)
    rs[0] = vec[0]
    rs[1] = vec[1]
    rs[2] = vec[2]
    return rs


def get_global_transform(layer, nodes):
    qs = list()  # (J, F, fbxQuaternion)
    translations = list()  # (J, F, 3)
    nkeys = nodes[0].LclRotation.GetCurve(layer, "X", True).KeyGetCount()
    fps = 30.0

    for j in range(len(nodes)):
        qs_node = list()
        translations_node = list()
        for i in range(nkeys):
            time = fbx.FbxTime()
            time.SetSecondDouble(i / fps)
            transform = nodes[j].EvaluateGlobalTransform(time)
            translations_node.append(fbxVec4ToList(transform.GetT()))
            qs_node.append(transform.GetQ())
        qs.append(qs_node)
        translations.append(translations_node)

    qs = np.array(qs)
    translations = np.array(translations)
    qs = np.swapaxes(qs, 0, 1)
    translations = np.swapaxes(translations, 0, 1)

    return qs, translations


def get_order(nodes):
    order = nodes[0].GetRotationOrder(fbx.FbxNode.eDestinationPivot)
    for node in nodes[1:]:
        order_new = node.GetRotationOrder(fbx.FbxNode.eDestinationPivot)
        if order_new != order:
            raise Exception("Inconsistent euler order")

    if order == fbx.eEulerZYX:
        return 'zyx'
    else:
        print(order)
        raise Exception('Unsupported euler order')


def import_fbx(filename, pelvis_name=None):
    """
    :param filename: fbx file name
    :param pelvis_name: the name of skeleton root
    :return:
        joint_names:    (J, )
        parents:        (J, )
        offsets:        (J, 3)
        qs:             (F, J, 4) quaternions
        translations:   (F, J, 3)
        pre_transform:  fbx.FbxAMatrix, the global transformation applied to the whole skeleton, e.g. rotX = 180
    """
    # Initialize scene
    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)
    importer = fbx.FbxImporter.Create(manager, "")
    status = importer.Initialize(filename, -1, manager.GetIOSettings())
    if not status:
        status = importer.GetStatus()
        print("Call to FbxImporter::Initialize() failed")
        print("Error returned: %s" % status.GetErrorString())
        return

    scene = fbx.FbxScene.Create(manager, "scene")
    importer.Import(scene)
    importer.Destroy()

    # Load Skeleton
    pre_transform = fbx.FbxAMatrix()
    pre_transform.SetIdentity()
    if pelvis_name is not None:
        pelvis, pre_transform = get_node(scene.GetRootNode(), pelvis_name, pre_transform)
    else:
        root = scene.GetRootNode()
        pelvis = root
        for i in range(root.GetChildCount()):
            node = root.GetChild(i)
            if isinstance(node.GetNodeAttribute(), fbx.FbxSkeleton):
                pelvis = node
                break
    nodes, parents, joint_names = get_nodes(pelvis)

    # Load anim
    # nobjects = scene.GetSrcObjectCount()
    stack = scene.GetCurrentAnimationStack()
    layer = stack.GetMember(0)
    offsets, qs, ts = get_anim(layer, nodes)
    offsets[0] = np.zeros(3)
    qs_global, ts_global = get_global_transform(layer, nodes)

    # consider pre transform
    preQ = pre_transform.GetQ()                 # fbx.FbxQuaternion
    qs[:, 0] = preQ * qs[:, 0]
    preT = fbxVec4ToList(pre_transform.GetT())  # (3, )
    ts[:, 0] += preT

    # fbx.FbxQuaternion to np array
    qs = fbxQuaternionToArray(qs)
    qs_global = fbxQuaternionToArray(qs_global)

    return joint_names, parents, offsets, qs, ts, qs_global, ts_global


def import_fbx_as_anim(filename, pelvis_name=None):
    joint_names, parents, offsets, qs, ts, qs_global, ts_global = import_fbx(filename, pelvis_name)
    orients = Quaternions.id(len(joint_names))
    # anim = Animation(rotations=Quaternions.id((qs.shape[0], qs.shape[1])),
    #                  positions=ts, orients=orients, offsets=offsets, parents=parents)
    anim = Animation(rotations=Quaternions(qs), positions=ts, orients=orients, offsets=offsets, parents=parents)

    return joint_names, parents, offsets, anim, ts_global


def import_scene(filename):
    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)
    importer = fbx.FbxImporter.Create(manager, "")
    status = importer.Initialize(filename, -1, manager.GetIOSettings())
    if not status:
        status = importer.GetStatus()
        print("Call to FbxImporter::Initialize() failed")
        print("Error returned: %s" % status.GetErrorString())
        return

    scene = fbx.FbxScene.Create(manager, "scene")
    importer.Import(scene)
    importer.Destroy()
    return scene


def transform_format(fbx_filename, bvh_filename, remove_namespace=True):
    from BVH import save
    joint_names, _, _, anim, _ = import_fbx_as_anim(fbx_filename)
    if remove_namespace:
        for i in range(len(joint_names)):
            joint_names[i] = joint_names[i].split(':')[-1]
    save(bvh_filename, anim=anim, names=joint_names, frametime=1/30.0)

