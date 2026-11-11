"""Offline shared panorama: bounded image registration, coherent seams, RGB fallback."""
import argparse
import os
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy import sparse
from scipy.ndimage import distance_transform_edt
from scipy.sparse.linalg import lsqr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime'))
from g1prep.memory import RGBDFrame, camera_transform
from PIL import Image, ImageDraw, ImageFont
from complete_ranges import complete_ranges

WIDTH, HEIGHT, GRID = 2048, 640, 4
SPAN = HEIGHT / WIDTH * 360
PIPELINE_REVISION = 'dual-shared-panorama-20261023-full-range-v4'


def contact(paths,destination,columns,width,height):
    canvas=Image.new('RGB',(columns*width,math.ceil(len(paths)/columns)*(height+28)),'#202020')
    draw=ImageDraw.Draw(canvas)
    for i,(path,label) in enumerate(paths):
        image=Image.open(path).convert('RGB');image.thumbnail((width,height),Image.Resampling.LANCZOS)
        x,y=(i%columns)*width,(i//columns)*(height+28)
        draw.text((x+6,y+7),label,fill='white')
        canvas.paste(image,(x+(width-image.width)//2,y+28+(height-image.height)//2))
    canvas.save(destination)


def validate_record(record):
    if (record.get('kind')!='dual_camera_measured_circle' or not record.get('complete')
            or len(record.get('views',[]))!=24 or record.get('synthetic')
            or record.get('stop_error') or record.get('final_measurement_error')
            or 'return_pose' not in record
            or not record.get('final_displacement',{}).get('stationary_by_sensors')):
        raise ValueError('A complete, stopped, real 24-frame circle is required')
    if (not 0<float(record['elapsed_s'])<=480 or not 0<=float(record['max_base_drift_m'])<=.5
            or abs(record['final_displacement']['heading_change_deg'])>8):
        raise ValueError('Capture exceeded its tested time, drift or heading envelope')
    calibration=record['calibration']
    checked=Path(os.environ.get('NAV_CALIBRATION', str(ROOT/'dual-camera-static-calibration.json'))).expanduser()
    if (calibration.get('status')!='STATIC_LOCAL_CHECKS_PASSED' or not checked.exists()
            or hashlib.sha256(checked.read_bytes()).hexdigest()!=record.get('calibration_sha256')
            or json.loads(checked.read_text())!=calibration):
        raise ValueError('Capture does not match the installed dual-camera calibration')
    serials={c['serial'] for c in calibration['cameras']}
    expected={(i,s) for i in range(12) for s in serials}
    if len(serials)!=2 or {(v['index'],v['serial']) for v in record['views']}!=expected:
        raise ValueError('Missing or duplicated heading/camera pair')
    for v in record['views']:
        if (v['file']!='view_%03d_%s.npz'%(v['index']*30,v['serial'])
                or v['target_relative_yaw_deg']!=v['index']*30
                or abs(v['pose']['relative_yaw_deg']-v['index']*30)>5
                or not 0<=v['rgb_depth_skew_s']<=.1 or not 0<=v['pose_time_error_s']<=.1):
            raise ValueError('Invalid capture path, heading or synchronization')


def load_frames(directory):
    record = json.loads((directory/'scan.json').read_text())
    validate_record(record)
    cameras = {c['serial']: c for c in record['calibration']['cameras']}
    frames = []
    record['views']=sorted(record['views'],key=lambda v:(v['index'],v['serial']))
    for entry in record['views']:
        data = np.load(directory/entry['file'], allow_pickle=False)
        pose, stamp = entry['pose'], entry['capture_monotonic_s']
        if data['rgb'].shape!=(720,1280,3) or data['depth_m'].shape!=(720,1280):
            raise ValueError('Saved RGBD profile differs from installation calibration')
        f = RGBDFrame(data['rgb'], data['depth_m'], tuple(cameras[entry['serial']]['intrinsics']),
            np.array(entry['world_camera']), (*pose['position_m'][:2], pose['rpy_rad'][2]), stamp,stamp,pose['pose_monotonic_s'])
        f.validate(max_depth=np.inf, min_valid_depth_fraction=0)
        frames.append(f)
    return frames, record


def grid_weights(pixels, shape):
    h,w=shape
    xy=np.clip(pixels/np.array([w-1,h-1])*(GRID-1),0,GRID-1-1e-6)
    x,y=np.floor(xy).astype(int).T
    dx,dy=(xy-np.floor(xy)).T
    indices=np.column_stack((y*GRID+x,y*GRID+x+1,(y+1)*GRID+x,(y+1)*GRID+x+1))
    weights=np.column_stack(((1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy))
    return indices,weights


def correction(pixels, shape, controls):
    ids,weights=grid_weights(pixels.reshape(-1,2),shape)
    return np.sum(controls[ids]*weights[...,None],axis=1).reshape((*pixels.shape[:-1],2))


def completed_frame_depth(frame):
    """Cache estimated raw depth separately; never relabel it as measured."""
    if not hasattr(frame,'_completed_depth_m'):
        actual=np.isfinite(frame.depth_m)&(frame.depth_m>.1)
        frame._completed_depth_m,_,_=complete_ranges(frame.depth_m,actual,periodic=False)
    return frame._completed_depth_m


def angles(frame,pixels,origin,yaw):
    x,y=np.rint(pixels).astype(int).T
    z=completed_frame_depth(frame)[y,x]
    fx,fy,cx,cy=frame.intrinsics
    p=np.column_stack(((pixels[:,0]-cx)/fx*z,(pixels[:,1]-cy)/fy*z,z))
    q=p@frame.world_camera[:3,:3].T+frame.world_camera[:3,3]-origin
    return np.column_stack((np.arctan2(q[:,1],q[:,0])-yaw,
                            np.arctan2(q[:,2],np.linalg.norm(q[:,:2],axis=1))))*180/np.pi


def feature_edges(frames):
    features=[]
    sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.015)
    for f in frames:
        keys,desc=sift.detectAndCompute(cv2.cvtColor(f.rgb,cv2.COLOR_RGB2GRAY),None)
        features.append((np.array([k.pt for k in keys]),desc))
    edges=[]
    for i in range(len(frames)):
        for j in range(i+1,len(frames)):
            if frames[i].world_camera[:3,2]@frames[j].world_camera[:3,2]<math.cos(math.radians(75)):
                continue
            ua,da=features[i]; ub,db=features[j]
            if da is None or db is None or min(len(da),len(db))<2:
                continue
            matcher=cv2.BFMatcher()
            forward={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(da,db,k=2) if m.distance<.75*n.distance}
            backward={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(db,da,k=2) if m.distance<.75*n.distance}
            pairs=np.array([(a,b) for a,b in forward.items() if backward.get(b)==a])
            if len(pairs)<10:
                continue
            a,b=ua[pairs[:,0]],ub[pairs[:,1]]
            _,mask=cv2.findHomography(a,b,cv2.RANSAC,3,maxIters=3000)
            if mask is not None and mask.sum()>=8:
                edges.append(dict(i=i,j=j,a=a[mask[:,0]>0].tolist(),b=b[mask[:,0]>0].tolist()))
    return edges


def register(frames,edges,origin,yaw):
    """Small visual correction only; never replace the robot or mounting calibration."""
    n=len(frames)*GRID*GRID
    rows,cols,values,targets,held,pairs=[],[],[],[],[],[]
    groups=[{i} for i in range(len(frames))]
    def add(coefficients,target,hold=False,pair=None):
        row=len(targets)
        for c,v in coefficients:
            rows.append(row);cols.append(c);values.append(v)
        targets.append(target);held.append(hold);pairs.append(pair)
    for edge in edges:
        i,j=edge['i'],edge['j']; a,b=np.array(edge['a']),np.array(edge['b'])
        delta=angles(frames[i],a,origin,yaw)-angles(frames[j],b,origin,yaw)
        delta[:,0]=(delta[:,0]+180)%360-180
        ia,wa=grid_weights(a,frames[i].depth_m.shape);ib,wb=grid_weights(b,frames[j].depth_m.shape)
        good=np.linalg.norm(delta,axis=1)<4
        if good.sum()<6:
            continue
        for k in np.flatnonzero(good):
            coeff=list(zip(ia[k]+i*GRID*GRID,wa[k]))+list(zip(ib[k]+j*GRID*GRID,-wb[k]))
            add(coeff,-delta[k],k%5==0,(i,j))
        g1=next(g for g in groups if i in g);g2=next(g for g in groups if j in g)
        if g1 is not g2:g1.update(g2);groups.remove(g2)
    data_rows=len(targets)
    if not data_rows:
        return np.zeros((len(frames),GRID*GRID,2)),{'accepted':False,'reason':'no shared details'}
    # Keep unconstrained regions near measured projection and penalize curvature.
    for i in range(len(frames)):
        offset=i*GRID*GRID
        for k in range(GRID*GRID):add([(offset+k,.06)],[0,0])
        for y in range(GRID):
            for x in range(GRID):
                k=offset+y*GRID+x
                if x<GRID-1:add([(k,.08),(k+1,-.08)],[0,0])
                if y<GRID-1:add([(k,.08),(k+GRID,-.08)],[0,0])
                if 0<x<GRID-1:add([(k-1,.6),(k,-1.2),(k+1,.6)],[0,0])
                if 0<y<GRID-1:add([(k-GRID,.6),(k,-1.2),(k+GRID,.6)],[0,0])
    for group in groups:
        anchor=min(group)
        for k in range(GRID*GRID):add([(anchor*GRID*GRID+k,.6)],[0,0])
    A=sparse.csr_matrix((values,(rows,cols)),shape=(len(targets),n)); b=np.asarray(targets)
    train=~np.asarray(held);weights=np.ones(len(b));weights[~train]=0
    controls=np.zeros((n,2))
    for _ in range(6):
        Aw=A.multiply(np.sqrt(weights)[:,None]).tocsr()
        for axis in range(2):controls[:,axis]=lsqr(Aw,b[:,axis]*np.sqrt(weights),atol=1e-8,btol=1e-8,iter_lim=800)[0]
        residual=np.linalg.norm(A@controls-b,axis=1)
        weights[:data_rows]=np.minimum(1,.25/np.maximum(residual[:data_rows],1e-6))
        weights[~train]=0
    controls=np.clip(controls,-3,3)
    before=np.linalg.norm(b[:data_rows],axis=1);after=np.linalg.norm((A@controls-b)[:data_rows],axis=1)
    h=np.asarray(held[:data_rows]); report=[]
    for pair in sorted(set(pairs[:data_rows])):
        selected=np.array([p==pair for p in pairs[:data_rows]])&h
        if selected.any():report.append({'frames':pair,'samples':int(selected.sum()),
            'before_deg':float(np.median(before[selected])),'after_deg':float(np.median(after[selected]))})
    accepted=(np.median(after[h])<np.median(before[h])*.8 and np.percentile(after[h],90)<np.percentile(before[h],90))
    audit={'accepted':bool(accepted),'matched_pairs':len(edges),'held_out_points':int(h.sum()),
        'before_median_deg':float(np.median(before[h])),'after_median_deg':float(np.median(after[h])),
        'before_p90_deg':float(np.percentile(before[h],90)),'after_p90_deg':float(np.percentile(after[h],90)),
        'max_control_deg':float(np.max(np.abs(controls))), 'pairs':report,
        'meaning':'visual seam registration only; held-out matches share feature selection; not metric calibration',
        'groups':[sorted(g) for g in groups]}
    return (controls.reshape(len(frames),GRID*GRID,2) if accepted else np.zeros((len(frames),GRID*GRID,2))),audit


def layer(frame,controls,origin,yaw):
    """Project original RGBD; never move a measured foreground pixel to infinity."""
    h,w=frame.depth_m.shape; yy,xx=np.indices((h,w),dtype=np.float32)
    raw=frame.depth_m; measured=np.isfinite(raw)&(raw>.1)
    z=completed_frame_depth(frame)
    fx,fy,cx,cy=frame.intrinsics
    points=np.stack(((xx-cx)/fx*z,(yy-cy)/fy*z,z),-1)
    delta=points@frame.world_camera[:3,:3].T+frame.world_camera[:3,3]-origin
    radius=np.linalg.norm(delta,axis=-1)
    theta=np.degrees(np.arctan2(delta[...,1],delta[...,0])-yaw)
    elev=np.degrees(np.arctan2(delta[...,2],np.linalg.norm(delta[...,:2],axis=-1)))
    adjust=correction(np.stack((xx,yy),-1),raw.shape,controls)
    theta+=adjust[...,0];elev+=adjust[...,1]
    u=(theta%360)/360*WIDTH;v=(SPAN/2-elev)/SPAN*HEIGHT
    ix=np.floor(u).astype(int)%WIDTH;iy=np.floor(v).astype(int)
    good=(iy>=0)&(iy<HEIGHT)&(radius>.1)&np.isfinite(radius)
    pixels=(iy[good]*WIDTH+ix[good]);ranges=radius[good]
    # A projected estimate must not replace an actual measurement in the same
    # output pixel. Within each evidence type, retain the nearer surface.
    order=np.lexsort((ranges,~measured[good],pixels));_,first=np.unique(pixels[order],return_index=True);choose=order[first]
    ids=pixels[choose]; native=np.flatnonzero(good)[choose]
    image=np.zeros((HEIGHT*WIDTH,3),np.uint8);depth=np.zeros(HEIGHT*WIDTH,np.float32)
    real=np.zeros(HEIGHT*WIDTH,bool)
    image[ids]=frame.rgb.reshape(-1,3)[native];depth[ids]=ranges[choose];real[ids]=measured.ravel()[native]
    image=image.reshape(HEIGHT,WIDTH,3);depth=depth.reshape(HEIGHT,WIDTH);real=real.reshape(HEIGHT,WIDTH)
    mask=(depth>0).astype(np.uint8)*255
    return image,depth,real,mask


def exposure(images,depths,masks):
    equations=[];values=[];n=len(images)
    for i in range(n):
        for j in range(i+1,n):
            a=images[i][::4,::4].astype(float);b=images[j][::4,::4].astype(float)
            da,db=depths[i][::4,::4],depths[j][::4,::4]
            good=((masks[i][::4,::4]>0)&(masks[j][::4,::4]>0)&
                (np.abs(da-db)<.15+.04*np.minimum(da,db))&
                np.all((a>25)&(a<235)&(b>25)&(b<235),axis=-1))
            if good.sum()<80:continue
            difference=np.log(a[good]/b[good]);shift=np.median(difference,axis=0)
            if np.max(np.median(np.abs(difference-shift),axis=0))>.15:continue
            row=np.zeros(n);row[i],row[j]=-1,1
            equations.append(row);values.append(shift)
    if not equations:return images,np.ones((n,3)).tolist()
    A=np.vstack((equations,np.ones((1,n))*10,np.eye(n)*.02));b=np.vstack((values,np.zeros((n+1,3))))
    gains=np.exp(np.linalg.lstsq(A,b,rcond=None)[0]).clip(.55,1.8)
    return [np.clip(im.astype(float)*g,0,255).astype(np.uint8) for im,g in zip(images,gains)],gains.tolist()


def compose(frames,controls,origin,yaw,cache=None):
    layers=[]
    key=hashlib.sha256(PIPELINE_REVISION.encode()+controls.tobytes()+origin.tobytes()+str((yaw,WIDTH,HEIGHT)).encode()+
        b''.join(f.world_camera.tobytes()+f.rgb.tobytes()+f.depth_m.tobytes() for f in frames)).hexdigest()[:16]
    if cache:cache.mkdir(parents=True,exist_ok=True)
    for index,(f,c) in enumerate(zip(frames,controls)):
        path=cache/(key+'_%02d.npz'%index) if cache else None
        if path and path.exists():
            with np.load(path) as data:entry=tuple(data[k] for k in ('rgb','depth','real','mask'))
        else:
            entry=layer(f,c,origin,yaw)
            if path:np.savez_compressed(path,rgb=entry[0],depth=entry[1],real=entry[2],mask=entry[3])
        layers.append(entry)
        print(json.dumps({'projected_photo':index+1}),flush=True)
    images,depths,reals,masks=map(list,zip(*layers))
    images,gains=exposure(images,depths,masks)
    # Keep complete people and nearby objects in the lower camera. Both cameras
    # can contribute in a narrow band above the horizon, away from their bodies.
    elevation=SPAN/2-(np.arange(HEIGHT)+.5)/HEIGHT*SPAN
    for i in range(len(masks)):
        eligible=elevation>=6 if i%2==0 else elevation<=10
        masks[i]=masks[i]*eligible[:,None]
    support_masks=[m.copy() for m in masks]
    yy,xx=np.indices((HEIGHT,WIDTH),dtype=np.float32)
    theta=(xx+.5)/WIDTH*2*np.pi+yaw;el=np.radians(SPAN/2-(yy+.5)/HEIGHT*SPAN)
    directions=np.stack((np.cos(el)*np.cos(theta),np.cos(el)*np.sin(theta),np.sin(el)),-1)
    quality=[]
    for f,mask in zip(frames,masks):
        q=directions@f.world_camera[:3,:3];fx,fy,cx,cy=f.intrinsics;h,w=f.depth_m.shape
        u=q[...,0]/np.maximum(q[...,2],1e-6)*fx+cx;v=q[...,1]/np.maximum(q[...,2],1e-6)*fy+cy
        quality.append(np.where(mask>0,((u-cx)/(w/2))**2+((v-cy)/(h/2))**2,np.inf))
    best_quality=np.min(quality,axis=0)
    closest=np.min([np.where(real,cv2.medianBlur(depth,3),np.inf) for real,depth in zip(reals,depths)],axis=0)
    for i in range(len(masks)):
        visible=(reals[i]&(depths[i]<=closest+.15+.03*depths[i]))|(~reals[i]&~np.isfinite(closest))
        masks[i]=masks[i]*visible*(quality[i]<=best_quality+.35)
    padding=64
    def pad(a):return np.concatenate((a[:,-padding:],a,a[:,:padding]),axis=1)
    # Select contiguous source regions; narrow-band color blending follows below.
    seam_images,seam_masks,corners,owners=[],[],[],[]
    for index,(image,mask) in enumerate(zip(images,masks)):
        image,mask=pad(image),pad(mask)
        count,components,stats,_=cv2.connectedComponentsWithStats(mask)
        for component in range(1,count):
            x,y,w,h,area=stats[component]
            if area<100:
                continue
            patch=image[y:y+h,x:x+w].copy()
            valid=components[y:y+h,x:x+w]==component
            # Reflect-like nearest-photo extension is only filter padding;
            # the output seam mask still requires a real source pixel.
            nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
            patch[~valid]=patch[tuple(nearest[:,~valid])]
            seam_images.append(patch)
            seam_masks.append(cv2.UMat(((components[y:y+h,x:x+w]==component)*255).astype(np.uint8)))
            corners.append((int(x),int(y)));owners.append(index)
    finder=cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
    print(json.dumps({'finding_seams_between_patches':len(seam_masks)}),flush=True)
    finder.find([im.astype(np.float32) for im in seam_images],corners,seam_masks)
    extended=np.full((HEIGHT,WIDTH+padding*2),-1,np.int16)
    blender=cv2.detail_MultiBandBlender(0,3)
    blender.prepare((0,0,WIDTH+padding*2,HEIGHT))
    for index,(m,(x,y)) in enumerate(zip(seam_masks,corners)):
        use=m.get()>0;h,w=use.shape
        extended[y:y+h,x:x+w][use]=owners[index]
        blender.feed(seam_images[index].astype(np.int16),m,(x,y))
    labels=extended[:,padding:-padding].copy()
    blended,blended_mask=blender.blend(None,None)
    color=np.clip(blended[:,padding:-padding],0,255).astype(np.uint8)
    # If graph-cut leaves a shared boundary pixel unlabelled, choose a real
    # covering photo, preferring its optical center. Never invent a color.
    uncovered=labels<0
    if uncovered.any():
        y,x=np.indices(labels.shape);theta=(x+.5)/WIDTH*2*np.pi+yaw;elev=np.radians(SPAN/2-(y+.5)/HEIGHT*SPAN)
        directions=np.stack((np.cos(elev)*np.cos(theta),np.cos(elev)*np.sin(theta),np.sin(elev)),-1)
        best=np.full(labels.shape,np.inf)
        for i,f in enumerate(frames):
            q=directions@f.world_camera[:3,:3];score=-q[...,2]
            use=uncovered&(support_masks[i]>0)&(score<best)
            labels[use]=i;best[use]=score[use]
    rgb=color.copy();radial=np.zeros((HEIGHT,WIDTH),np.float32);measured=np.zeros((HEIGHT,WIDTH),bool)
    for i in range(len(frames)):
        use=labels==i;radial[use]=depths[i][use];measured[use]=reals[i][use]
        fallback=use&uncovered;rgb[fallback]=images[i][fallback]
        # This fallback only repairs the COLOR seam. The covering source can
        # already contain a measured range: retain it, rather than converting
        # an RGB seam-finder hole into an artificial 100 m opening in depth.
    # A translated viewpoint can expose background that no original camera saw.
    # Borrow nearby projected background pixels, rather than copying the occluding
    # foreground along an infinite-distance ray. Keep its range explicitly assumed.
    missing=(labels<0)&(np.abs(elevation[:,None])<=46)
    borrowed_count=int(missing.sum())
    count,components,stats,_=cv2.connectedComponentsWithStats(missing.astype(np.uint8))
    for component in range(1,count):
        region=components==component
        boundary=(cv2.dilate(region.astype(np.uint8),np.ones((9,9),np.uint8))>0)&(labels>=0)
        threshold=float(np.percentile(radial[boundary],65)) if boundary.any() else 0
        candidate=(labels>=0)&(radial>=threshold)
        if not candidate.any():candidate=labels>=0
        if not candidate.any():raise RuntimeError('No real photograph covers the requested panorama')
        nearest=distance_transform_edt(~candidate,return_distances=False,return_indices=True)
        take=tuple(nearest[:,region])
        rgb[region]=rgb[take];labels[region]=labels[take]
        radial[region]=radial[take];measured[region]=False
    return rgb,radial,measured,labels,{'exposure_gains':gains,
        'rgb_seam_fallback_changes_measured_depth':False,
        'source_selection':'shared graph-cut; measured depth follows dominant photo',
        'color_composition':'global exposure correction; three-band narrow boundary blending',
        'camera_seam_band_elevation_deg':[6,10],
        'borrowed_background_pixels':borrowed_count,
        'disocclusion_fill':'color borrows nearby projected background; range completion keeps explicit estimate masks'}


def sample_view(atlas,depth,measured,labels,slot,yaw,size=256):
    y,x=np.indices((size,size),dtype=np.float32);rays=np.stack(((x-(size-1)/2)/(size/2),(y-(size-1)/2)/(size/2),np.ones_like(x)),-1)
    T=camera_transform(0,0,0,yaw+slot*np.pi/6);world=rays@T[:3,:3].T
    theta=(np.arctan2(world[...,1],world[...,0])-yaw)%(2*np.pi)
    elevation=np.arctan2(world[...,2],np.linalg.norm(world[...,:2],axis=-1))
    u=(theta/(2*np.pi)*WIDTH-.5).astype(np.float32);v=((SPAN/2-np.degrees(elevation))/SPAN*HEIGHT-.5).astype(np.float32)
    # Horizontal wrap is implemented explicitly; vertical image boundaries never wrap.
    def take(a,method):
        padded=np.concatenate((a[:,-1:],a,a[:,:1]),axis=1)
        return cv2.remap(padded,u+1,v,method,borderMode=cv2.BORDER_REPLICATE)
    rgb=take(atlas,cv2.INTER_LINEAR);radial=take(depth,cv2.INTER_NEAREST)
    real=take(measured.astype(np.uint8),cv2.INTER_NEAREST)>0;source=take(labels,cv2.INTER_NEAREST)
    return rgb,radial/np.linalg.norm(rays,axis=-1),real,source


def comparisons(before,output):
    saved=output/'before';saved.mkdir(exist_ok=True)
    pairs=[]
    for slot in range(12):
        name='%03d.png'%(slot*30)
        Image.open(before/'views90'/name).save(saved/name)
        pairs.extend([(saved/name,'%03d deg BEFORE'%(slot*30)),
                      (output/'views90'/name,'%03d deg AFTER'%(slot*30))])
    contact(pairs,output/'before-after-all.jpg',4,384,384)
    font=ImageFont.load_default()
    examples=[(0,(0,20,256,190),'上下相机重叠：花盆与墙面'),
              (180,(0,0,256,230),'相邻方向接缝：走廊'),
              (270,(0,0,256,235),'上下相机与相邻方向：蓝色吊架')]
    heights=[round((b[3]-b[1])/(b[2]-b[0])*512) for _,b,_ in examples]
    canvas=Image.new('RGB',(1056,64+sum(h+48 for h in heights)),'#202020');draw=ImageDraw.Draw(canvas)
    draw.text((20,14),'改善前',font=font,fill='white');draw.text((548,14),'改善后',font=font,fill='white')
    y=64
    for (degrees,box,title),height in zip(examples,heights):
        draw.text((14,y),str(degrees)+'°  '+title,font=font,fill='white');y+=40
        for x,folder in [(8,saved),(536,output/'views90')]:
            image=Image.open(folder/('%03d.png'%degrees)).crop(box).resize((512,height),Image.Resampling.NEAREST)
            canvas.paste(image,(x,y))
        y+=height+8
    canvas.save(output/'comparison-highlights.png')
    cards=[]
    for slot in range(12):
        name='%03d.png'%(slot*30)
        cards.append('<section><h2>'+str(slot*30)+'°</h2><div class="pair"><figure><figcaption>改善前</figcaption><a href="before/'+name+'"><img src="before/'+name+'"></a></figure><figure><figcaption>改善后</figcaption><a href="views90/'+name+'"><img src="views90/'+name+'"></a></figure></div></section>')
    html='<!doctype html><meta charset="utf-8"><title>拼接改善前后</title><style>body{font:16px sans-serif;margin:24px;background:#eee} .pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0;background:white;padding:8px}img{width:100%;image-rendering:auto}figcaption{padding:8px}section{max-width:1100px;margin:auto}p,h1{max-width:1100px;margin:20px auto}</style><h1>拼接改善前后</h1><p>每组左边为改善前，右边为改善后。仅使用已保存照片，未连接机器人。12 张生成图已没有留白；借用邻近背景补齐的位置仍标记为估计距离，并非新增实测。局部复杂边缘仍可能有接缝。点击图片查看原尺寸。</p>'+''.join(cards)
    (output/'gallery.html').write_text(html,encoding='utf-8')


def reconstruct(directory,output,*,reference='start',before=None,cache=None):
    output.mkdir(parents=True,exist_ok=True)
    frames,record=load_frames(directory)
    pose=[*record['start_pose']['position_m'][:2],record['start_pose']['rpy_rad'][2]]
    if reference=='end':
        final=record['final_displacement']
        pose=[*final['final_position_xy_m'],pose[2]+math.radians(final['heading_change_deg'])]
    elif reference!='start':raise ValueError('Invalid reference pose')
    if not np.isfinite(pose).all():raise ValueError('Finite reference pose required')
    origin=np.array([*pose[:2],1.25]);yaw=pose[2]
    edges=feature_edges(frames);controls,registration=register(frames,edges,origin,yaw)
    print(json.dumps({'registration':{k:registration.get(k) for k in
        ('accepted','held_out_points','before_median_deg','after_median_deg')}}),flush=True)
    atlas,radial,real,labels,composition=compose(frames,controls,origin,yaw,cache)
    radial,estimated,completion=complete_ranges(radial,real,labels>=0)
    Image.fromarray(atlas).save(output/'panorama.png')
    np.savez_compressed(output/'panorama-data.npz',radial_m=radial,measured=real,estimated=estimated,source=labels,warp_controls_deg=controls)
    outputs={};meters={};paths=[];coverage=[]
    square=output/'views90';square.mkdir(exist_ok=True)
    for slot in range(12):
        rgb,depth,measured,source=sample_view(atlas,radial,real,labels,slot,yaw)
        inferred=sample_view(atlas,radial,estimated,labels,slot,yaw)[2]
        valid=source>=0;unknown=~measured&~inferred
        if unknown.any() or not np.all(np.isfinite(depth)&(depth>0)):
            raise RuntimeError('Every output distance must be measured or estimated')
        suffix='' if not slot else '_'+str(slot*30)
        outputs['rgb'+suffix]=cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA)[None]
        outputs['depth'+suffix]=(np.minimum(depth,10)/10)[None,...,None]
        outputs['valid'+suffix]=valid[None,...,None]
        outputs['depth_measured'+suffix]=measured[None,...,None]
        outputs['depth_assumed'+suffix]=(~measured&valid)[None,...,None]
        outputs['depth_estimated'+suffix]=inferred[None,...,None]
        outputs['depth_unknown'+suffix]=unknown[None,...,None]
        meters['depth_m'+suffix]=depth[None,...,None]
        name='%03d.png'%(slot*30);Image.fromarray(rgb).save(square/name)
        paths.append((square/name,'%03d deg'%(slot*30)));coverage.append(float(valid.mean()))
    if not all(c==1. for c in coverage):raise RuntimeError('Incomplete stitched model input; no route executed')
    np.savez_compressed(output/'model-input.npz',**outputs);np.savez_compressed(output/'depth-meters.npz',**meters)
    contact(paths,output/'12-views90.png',4,320,320)
    if before is not None:comparisons(before,output)
    audit={'source_frames':24,'model_views':12,'rgb_coverage':coverage,'registration':registration,**composition,
        'pipeline_revision':PIPELINE_REVISION,'range_completion':completion,'reference_pose_xy_yaw':pose,'reference':reference,
        'virtual_camera_height_m':1.25,'horizontal_fov_deg':90,'vertical_fov_deg':90,
        'calibration_sha256':record['calibration_sha256'],'motion_dispatch':False,
        'output_directory':str(output),
        'missing_value_sentinel':None,'all_output_ranges_measured_or_estimated':True,
        'color_fallback':'projected original RGB; disocclusions borrow nearby background; measured and estimated ranges stay distinct',
        'measured_ranges_over_10m_retained':True,'shared_panorama':True,'visual_warp_is_metric_calibration':False,
        'metric_navigation_validated':False,'robot_connected':False,'generative_fill':False}
    (output/'reconstruction.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({'coverage':coverage}),flush=True)
    observations={k:outputs[k] for i in range(12) for k in
                  ('rgb'+('' if i==0 else '_'+str(i*30)),'depth'+('' if i==0 else '_'+str(i*30)))}
    stamp=max(v['capture_monotonic_s'] for v in record['views'])
    return observations,pose,stamp,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scan',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--before',type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--reference',choices=('start','end'),default='start')
    args=parser.parse_args()
    reconstruct(args.scan,args.output,reference=args.reference,before=args.before,cache=args.cache)


if __name__=='__main__':main()
