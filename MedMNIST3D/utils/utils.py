import torch.nn as nn
import numpy as np
from .batchnorm import SynchronizedBatchNorm3d, SynchronizedBatchNorm2d
import torchvision.transforms.functional as TF

class Transform3D:

    def __init__(self, mul=None, rotation=None, scale=None, translate=None):
        self.mul = mul
        self.rotation = rotation
        self.scale = scale
        self.translate = translate

    def __call__(self, voxel):
   
        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()
        
        if self.rotation is not None:
            # rotate by given angle for the last 3 dimensions
            # convert to 3D using reshape
            voxel_3d = voxel.reshape(28,28,28)
            voxel = self._rotate(voxel_3d, self.rotation)
            voxel = voxel.reshape(1,28,28,28)
            
        # if self.scale is not None:
        #     # scale by given factor
        #     voxel = self._scale(voxel)

        # if self.translate is not None:
        #     # translate by given factor
        #     voxel = self._translate(voxel)
        
        return voxel.astype(np.float32)
    
    def _rotate(self, voxel, angle):
        """
        the voxel is a 28x28x28 numpy array
        loop over the third dimension and rotate each slice
        and stack them back together
        TF.to_pil_image
        TF.rotate
        TF.to_tensor
        """
        pil_slices = []
        for i in range(voxel.shape[2]):
            slice = voxel[:, :, i]
            slice = (slice * 255).astype(np.uint8)  # convert to uint8 data type
            pil_slice = TF.to_pil_image(slice)
            rotated_slice = TF.rotate(pil_slice, angle)
            pil_slices.append(rotated_slice)
        stacked_slices = np.stack([TF.to_tensor(slice) for slice in pil_slices], axis=2)
        return stacked_slices
    
    def _scale(self, voxel):
        return TF.affine(voxel, angle=0, translate=[0, 0], scale=0.8, shear=0)
    
    def _translate(self, voxel):
        return TF.affine(voxel, angle=0, translate=[0.1, 0.1], scale=1, shear=0)



def model_to_syncbn(model):
    preserve_state_dict = model.state_dict()
    _convert_module_from_bn_to_syncbn(model)
    model.load_state_dict(preserve_state_dict)
    return model


def _convert_module_from_bn_to_syncbn(module):
    for child_name, child in module.named_children(): 
        if hasattr(nn, child.__class__.__name__) and \
            'batchnorm' in child.__class__.__name__.lower():
            TargetClass = globals()['Synchronized'+child.__class__.__name__]
            arguments = TargetClass.__init__.__code__.co_varnames[1:]
            kwargs = {k: getattr(child, k) for k in arguments}
            setattr(module, child_name, TargetClass(**kwargs))
        else:
            _convert_module_from_bn_to_syncbn(child)
