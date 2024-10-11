import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
from torchvision.utils import make_grid

import numpy as np
import trimesh
import pyrender

class Renderer:
    """
    Renderer used for visualizing the MANO model
    """
    def __init__(self, focal_length, img_res, faces):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces

    def demo(self, vertices, camera_translation, image, alpha=0):

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.15,
            alphaMode='OPAQUE',
            baseColorFactor=(.7, .7, .7, 1.))
        
        vertices += np.expand_dims(camera_translation, axis=0)
        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]

        output_img = (1 - valid_mask) * image + color[:, :, :3] * valid_mask * (1-alpha) + image * valid_mask *alpha
        return output_img
