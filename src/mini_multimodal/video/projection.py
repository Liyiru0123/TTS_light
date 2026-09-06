"""Dynamic programming projection that preserves space and time semantics."""
from __future__ import annotations
from itertools import product
from math import isclose
from mini_multimodal.world.schema import Scene
from mini_multimodal.vision.schema import ImageRenderConfig

def _sign(v):return 0 if isclose(v,0.,abs_tol=1e-9) else (-1 if v<0 else 1)

def _project_axis(scene:Scene,axis:str,cfg:ImageRenderConfig)->list[tuple[int,...]]:
    size=(scene.canvas_width if axis=="x" else scene.canvas_height)*cfg.output_scale;low=cfg.safe_margin;high=size-1-cfg.safe_margin
    values=[[getattr(obj.position_at(f),axis) for obj in scene.objects] for f in range(scene.frame_count)]
    ideals=[[int(low+v*(high-low)+.5) for v in row] for row in values]
    for radius in (1,2):
        states=[]
        for f in range(scene.frame_count):
            current=[]
            ranges=[range(max(low,p-radius),min(high,p+radius)+1) for p in ideals[f]]
            for state in product(*ranges):
                if all(_sign(values[f][i]-values[f][j])==_sign(state[i]-state[j]) for i in range(len(state)) for j in range(i+1,len(state))):current.append(state)
            states.append(current)
        dp={state:(sum(abs(state[i]-ideals[0][i]) for i in range(len(state))),[state]) for state in states[0]}
        for f in range(1,scene.frame_count):
            nxt={}
            for state in states[f]:
                for previous,(cost,path) in dp.items():
                    if all((_sign(values[f][i]-values[f-1][i])==0 and state[i]==previous[i]) or (_sign(values[f][i]-values[f-1][i])>0 and state[i]>=previous[i]) or (_sign(values[f][i]-values[f-1][i])<0 and state[i]<=previous[i]) for i in range(len(state))):
                        candidate=(cost+sum(abs(state[i]-ideals[f][i]) for i in range(len(state))),path+[state])
                        if state not in nxt or (candidate[0],candidate[1])<(nxt[state][0],nxt[state][1]):nxt[state]=candidate
            dp=nxt
        if dp:return min(dp.values(),key=lambda item:(item[0],item[1]))[1]
    raise ValueError(f"Unable to produce time-consistent {axis}-axis projection for {scene.scene_id}")

def project_video_centers(scene:Scene,cfg:ImageRenderConfig|None=None)->tuple[dict[str,tuple[int,int]],...]:
    config=cfg or ImageRenderConfig();xs=_project_axis(scene,"x",config);ys=_project_axis(scene,"y",config)
    return tuple({obj.object_id:(xs[f][i],ys[f][i]) for i,obj in enumerate(scene.objects)} for f in range(scene.frame_count))
