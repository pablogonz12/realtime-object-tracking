
# RTMDet Configuration (Detection only)
_base_ = ['rtmdet_l_8xb32-300e_coco.py']

# Override specific configurations
model = dict(
    bbox_head=dict(num_classes=80)
)