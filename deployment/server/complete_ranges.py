"""Continuous, untrained range estimates, with no numeric missing-value sentinel.

Measured values are immutable. Estimated values keep a separate mask and must
not be used as measured obstacle clearance. No image, instruction, or preferred
heading enters this operation.
"""
import cv2
import numpy as np


def complete_ranges(ranges,measured,valid=None,*,max_gap_radius_px=26,periodic=True,inpaint_radius_px=6):
    # `valid` describes RGB coverage for callers. Distance completion now covers
    # the entire array, including positions without an original RGB projection.
    values=np.asarray(ranges)
    measured=np.asarray(measured,dtype=bool)
    valid=np.ones_like(measured)if valid is None else np.asarray(valid,dtype=bool)
    if values.ndim!=2 or values.shape!=measured.shape or valid.shape!=measured.shape:
        raise ValueError('Range image and masks must be the same 2-D shape')
    if not np.issubdtype(values.dtype,np.floating) or max_gap_radius_px<0 or inpaint_radius_px<=0:
        raise ValueError('Floating-point ranges and nonnegative gap size are required')
    if np.any(measured&(~np.isfinite(values)|(values<=0))):
        raise ValueError('A measured range must be positive and finite')
    result=values.copy();estimated=np.zeros_like(measured)
    support=measured
    if not support.any():
        raise ValueError('Cannot estimate ranges without any positive measured support')
    extended_count=0;nearest_count=0
    if support.any():
        missing=~support
        # A plane in a perspective range image has affine inverse depth. Fill
        # narrow seams by interpolating reciprocal ranges between measured
        # boundaries, instead of blurring metre values across floor gradients.
        work=np.zeros_like(values,dtype=np.float32)
        work[support]=1/values[support]
        # Both horizontal sides see the same angular neighbourhood; the top
        # and bottom never wrap into one another.
        pad=int(np.ceil(max_gap_radius_px+2*inpaint_radius_px+3))if periodic else 0
        if pad:
            work=np.pad(work,((0,0),(pad,pad)),mode='wrap')
            missing=np.pad(missing,((0,0),(pad,pad)),mode='wrap')
        mask=missing.astype(np.uint8)
        distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        prediction=cv2.inpaint(work,mask,inpaint_radius_px,cv2.INPAINT_NS)
        weighted=np.zeros_like(work);weights=np.zeros_like(work)
        known=~missing
        for axis in [0,1]:
            shape=[1,1];shape[axis]=work.shape[axis]
            coordinate=np.arange(work.shape[axis]).reshape(shape)
            lower=np.maximum.accumulate(np.where(known,coordinate,-1),axis=axis)
            upper=np.minimum.accumulate(np.flip(np.where(known,coordinate,work.shape[axis]),axis),axis=axis)
            upper=np.flip(upper,axis)
            bilateral=(lower>=0)&(upper<work.shape[axis])&(upper>lower)
            before=np.take_along_axis(work,np.clip(lower,0,work.shape[axis]-1),axis=axis)
            after=np.take_along_axis(work,np.clip(upper,0,work.shape[axis]-1),axis=axis)
            span=np.maximum(upper-lower,1)
            weight=np.where(bilateral,1/span.astype(np.float32)**2,0)
            linear=(before*(upper-coordinate)+after*(coordinate-lower))/span
            weighted+=linear*weight;weights+=weight
        use=weights>0
        prediction[use]=weighted[use]/weights[use]
        prediction=np.clip(prediction,float(work[known].min()),float(work[known].max()))
        prediction=1/prediction
        if pad:
            distance=distance[:,pad:-pad];prediction=prediction[:,pad:-pad]
            use=use[:,pad:-pad]
        # Narrow gaps retain the existing interpolation. Wider gaps use their
        # farther measured boundaries; where only one side exists, extend the
        # nearest actual measurement. Neither source consults missing values.
        far=(~measured)&(distance>max_gap_radius_px)
        nearest_needed=far&~use
        if nearest_needed.any():
            field=np.tile(values,(1,3)) if periodic else values
            known=np.tile(support,(1,3)) if periodic else support
            _,labels=cv2.distanceTransformWithLabels((~known).astype(np.uint8),
                cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
            nearby=field[known][labels-1]
            if periodic:nearby=nearby[:,values.shape[1]:2*values.shape[1]]
            prediction[nearest_needed]=nearby[nearest_needed]
        estimated=~measured
        extended_count=int(far.sum());nearest_count=int(nearest_needed.sum())
        result[estimated]=prediction[estimated]
    assert np.array_equal(result[measured],values[measured])
    if not np.all(np.isfinite(result)&(result>0)):
        raise ValueError('Range completion did not produce positive finite values everywhere')
    unknown=~measured&~estimated
    return result,estimated,{
        'method':'inverse-range boundary interpolation; wider search and nearest measured extension for remaining gaps',
        'max_gap_radius_px':max_gap_radius_px,'inpaint_radius_px':inpaint_radius_px,
        'horizontal_wrap':periodic,'estimated_pixels':int(estimated.sum()),
        'extended_search_pixels':extended_count,'nearest_measured_extension_pixels':nearest_count,
        'missing_value_sentinel':None,'all_pixels_completed':True,
        'unresolved_pixels':int(unknown.sum()),'measured_values_changed':0,
        'estimates_are_measured_clearance':False}
