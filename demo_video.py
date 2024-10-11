#!/usr/bin/env python3

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torchvision.transforms as transforms

import cv2
import argparse
import numpy as np

import constants

from mano_torch import MANO
from hmr_model import hmr
from renderer import Renderer
from utils import batch_euler2matzxy, coordtrans, crop2expandsquare_zeros

def video_inference(video_path, device, mano, model, renderer):

    # read hand images
    video_name = video_path.split('/')[-1].split('.')[0]
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # define joint rotation transformation
    local2global = torch.from_numpy(constants.euler_coordtrans_RIGHT).to(device).type(torch.float32)
    local2global = batch_euler2matzxy(local2global.view(-1,3)).view(-1,15,3,3).expand(1,-1,-1,-1)

    # define image data processing
    trans = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Normalize(constants.IMG_NORM_MEAN, constants.IMG_NORM_STD)
                        ])

    # save output including the original video with the rendered 3D reconstruction 
    H, W, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'./output/{video_name}.avi', fourcc, 20, (int(constants.IMG_RES*2),constants.IMG_RES))

    for image in frames:

        # process each frame
        X_orig = image
        bbox = [W/2,H/2,W/2,H/2]

        # cut and resize
        X_cropped, offset_Pts, _ = crop2expandsquare_zeros(X_orig, bbox, 0)
        X_croppedresize = cv2.resize(X_cropped, (constants.IMG_RES, constants.IMG_RES), interpolation = cv2.INTER_CUBIC)
        X = trans(X_croppedresize.copy()).float().unsqueeze(0).to(device)

        with torch.no_grad():
            # get prediction for the input image
            pred_all = model(X)
            pred_pose_mean, _, pred_kp_sigma, pred_beta_mean, _, pred_cam_mean, _ = pred_all

            # obtain 3D hand vertex positions
            pred_pose_rotmat_mean = batch_euler2matzxy(pred_pose_mean.reshape(-1,3)).view(-1,16,3,3)
            pred_pose_rotmat_mean = coordtrans(pred_pose_rotmat_mean, local2global, 1)
            pred_output_mean = mano.forward(betas=pred_beta_mean, thetas_rotmat=pred_pose_rotmat_mean)
            pred_verts = pred_output_mean.vertices

            # render the 3D hand 
            pred_cam_t = torch.stack([pred_cam_mean[:,1],
                              pred_cam_mean[:,2],
                              constants.FOCAL_LENGTH/(pred_cam_mean[:,0] +1e-9)],dim=-1)
            rend_img = renderer.demo(pred_verts[0].detach().cpu().numpy().copy(), pred_cam_t[0].detach().cpu().numpy().copy(), X_croppedresize[:,:,::-1]/255.)

            # save the original image frame and the rendered 3D hand 
            output_frame = np.hstack([X_croppedresize[:,:,::-1]/255., rend_img])
            cv2.imshow('Video Demo', output_frame.copy())

            out.write((output_frame[:,:,::-1]*255).astype(np.uint8))
            if cv2.waitKey(1) & 0xff == 27: # exit if pressed `ESC`
                break

    out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='plot estimation or not')
    parser.add_argument('--checkpoint', type=str, default='./assets/KNOWN_HAND.pt', help='Path to network checkpoint')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to be used')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    video_path = args.video_path

    device = torch.device('cuda')

    # Load mano hand model
    mano = MANO('RIGHT', device)

    # Load trained 3D hand reconstruction model
    model = hmr()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=mano.faces)

    video_inference(video_path, device, mano, model, renderer)

if __name__ == '__main__':
    main()