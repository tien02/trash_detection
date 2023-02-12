import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_fasterrcnn(out_features:int):
    # Get Pretrained Faster RCNN #
    '''
    Input: 
        out_feature (int): Number of output channels
    Output:
        Faster RCNN
    '''
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, out_features)
    return model

# Get DETR