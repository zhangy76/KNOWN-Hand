import torch
import numpy as np

def batch_euler2matzxy(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = ymat @ xmat @ zmat
    return rotMat

def coordtrans(rotmat, local2global, transtype):

    batchsize = rotmat.shape[0]
    local2global = local2global.clone()
    if transtype == 1:
        rotmat[:,1:] = torch.einsum('...ij,...jk->...ik', rotmat[:,1:], local2global[:batchsize])
        rotmat[:,1:] = torch.einsum('...ij,...jk->...ik', local2global[:batchsize].transpose(2,3), rotmat[:,1:])
    else:
        rotmat[:,1:] = torch.einsum('...ij,...jk->...ik', rotmat[:,1:], local2global[:batchsize].transpose(2,3))
        rotmat[:,1:] = torch.einsum('...ij,...jk->...ik', local2global[:batchsize], rotmat[:,1:])

    return rotmat


def crop2expandsquare_zeros(img, bbox, expand_ratio):

    H = img.shape[0]
    W = img.shape[1]
    try:
        C = img.shape[2]
    except:
        C = 0
    if C == 0:
        img = np.repeat(img[:,:,None], 3, axis=2)
    elif C == 1:
        img = np.repeat(img, 3, axis=2)
    C = 3

    scale = 2*(np.random.rand(4)-0.5) * expand_ratio + 1
    # update bbox using length to center and scale
    x1_new = max(int(bbox[0] - bbox[2]*scale[0]), 0)
    y1_new = max(int(bbox[1] - bbox[3]*scale[1]), 0)
    x2_new = min(int(bbox[0] + bbox[2]*scale[2]), W)
    y2_new = min(int(bbox[1] + bbox[3]*scale[3]), H)

    Len_max = np.amax([y2_new-y1_new, x2_new-x1_new])

    x1_new_square = int((Len_max - x2_new + x1_new)/2)
    y1_new_square = int((Len_max - y2_new + y1_new)/2)
    x2_new_square = int((Len_max + x2_new - x1_new)/2)
    y2_new_square = int((Len_max + y2_new - y1_new)/2)

    # square image
    img_cropped_square = np.zeros([Len_max,Len_max,C], dtype = np.uint8)

    img_cropped_square[y1_new_square:y2_new_square,x1_new_square:x2_new_square,:] = img[y1_new:y2_new,x1_new:x2_new]

    offset_Pts = np.array([[x1_new-x1_new_square,y1_new-y1_new_square]])
    offset = [x1_new_square,y1_new_square]
    return img_cropped_square, offset_Pts, offset