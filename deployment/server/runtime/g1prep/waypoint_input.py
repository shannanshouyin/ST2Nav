"""Camera input ordering and periodic peak extraction; no learned correction."""
import numpy as np
import torch


def ordered_views(observations):
    """Bind each RGB/depth pair to its named 30-degree counterclockwise view.

    The upstream waypoint branch uses dictionary insertion order, not key angles.
    Auxiliary masks must not be mistaken for additional depth cameras.
    """
    result={}
    for slot in range(12):
        suffix='' if slot==0 else '_'+str(slot*30)
        for kind,shape,dtype in [('rgb',(224,224,3),np.uint8),('depth',(256,256,1),np.float32)]:
            key=kind+suffix
            if key not in observations:raise ValueError('Missing model view: '+key)
            value=observations[key]
            if not isinstance(value,np.ndarray) or value.shape[1:]!=shape or value.dtype!=dtype or value.shape[0]<=0:
                raise ValueError('Unexpected model view shape or dtype: '+key)
            if kind=='depth' and (not np.isfinite(value).all() or value.min()<0 or value.max()>1):
                raise ValueError('Expected normalized finite axial depth in [0, 1]: '+key)
            result[key]=value
    if len({v.shape[0] for v in result.values()})!=1:raise ValueError('Inconsistent camera batch size')
    return result


def circular_waypoint_nms(pred,max_predictions=5,sigma=(7.,5.),gaussian=False):
    """Select original heatmap maxima, periodic in angle and linear in range.

    Input/output follow upstream's one-row wrapped heatmap API. Padding is never
    an action and cannot consume the five-action budget. Integer row indices
    avoid a fractional suppression center from true division. No preferred
    direction, quotas, candidate increase, text, or RGB enter this operation.
    """
    if gaussian or pred.ndim!=4 or pred.shape[1:]!=(1,122,12):
        raise ValueError('Expected original 120-angle/12-range waypoint heatmap with one padding row')
    if max_predictions<0:raise ValueError('Invalid candidate limit')
    scores=pred[:,0,1:-1,:]
    remaining=scores.clone();selected=torch.zeros_like(scores)
    angle=torch.arange(120,device=pred.device)[None,:,None]
    distance=torch.arange(12,device=pred.device)[None,None,:]
    batch=torch.arange(pred.shape[0],device=pred.device)
    for _ in range(max_predictions):
        value,index=remaining.flatten(1).max(1)
        row=torch.div(index,12,rounding_mode='floor');col=index%12
        selected[batch,row,col]=torch.where(value>0,scores[batch,row,col],selected[batch,row,col])
        delta=(angle-row[:,None,None]).abs()
        delta=torch.minimum(delta,120-delta)
        suppress=(delta<=sigma[1])&((distance-col[:,None,None]).abs()<=sigma[0])
        remaining.masked_fill_(suppress,0)
    result=torch.zeros_like(pred);result[:,0,1:-1,:]=selected
    return result
