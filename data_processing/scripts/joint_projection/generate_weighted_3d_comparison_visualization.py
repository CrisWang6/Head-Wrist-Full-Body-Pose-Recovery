#!/usr/bin/env python3
"""Generate the inline 3D comparison fragment from a compact pose JSON."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.input.read_text(encoding="utf-8"))
    compact = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    template = r'''<div id="weighted-headgt-3d-viz">
  <div class="viz-controls" aria-label="三维骨架播放与视角控制">
    <button class="btn btn-primary" type="button" data-action="play" aria-pressed="false">播放</button>
    <button class="btn" type="button" data-view="iso" aria-pressed="true">等轴</button>
    <button class="btn" type="button" data-view="front" aria-pressed="false">正面</button>
    <button class="btn" type="button" data-view="side" aria-pressed="false">侧面</button>
    <button class="btn" type="button" data-view="top" aria-pressed="false">俯视</button>
    <label class="form-check"><input class="form-check-input" type="checkbox" data-layer="external" checked><span class="form-check-label">外部三角化</span></label>
    <label class="form-check"><input class="form-check-input" type="checkbox" data-layer="optimized" checked><span class="form-check-label">头部 GT 优化</span></label>
    <label class="form-check"><input class="form-check-input" type="checkbox" data-layer="headgt" checked><span class="form-check-label">稀疏头部 3D GT</span></label>
  </div>
  <div class="pose-stage" role="img" aria-label="外部双目三角化骨架与头部双目 GT 加权优化骨架的可旋转三维对比。拖拽旋转，滚轮缩放。"></div>
  <div class="timeline-row">
    <span class="text-small pose-time">seq 250 · 5.00 s</span>
    <input class="form-range" type="range" min="0" max="499" step="1" value="0" aria-label="验证段帧位置">
  </div>
  <div class="pose-legend text-small">
    <span><i class="swatch external"></i>外部相机三角化 3D GT</span>
    <span><i class="swatch optimized"></i>头部 2D/3D GT 加权优化</span>
    <span><i class="swatch headgt"></i>人工头部双目反投影 3D GT（仅标注帧）</span>
  </div>
</div>
<style>
  #weighted-headgt-3d-viz { width:100%; color:var(--foreground); }
  #weighted-headgt-3d-viz .viz-controls { margin-bottom:10px; }
  #weighted-headgt-3d-viz .pose-stage { position:relative; width:100%; height:560px; min-height:380px; overflow:hidden; background:color-mix(in srgb,var(--card) 72%,transparent); border-top:1px solid var(--border); border-bottom:1px solid var(--border); cursor:grab; }
  #weighted-headgt-3d-viz .pose-stage:active { cursor:grabbing; }
  #weighted-headgt-3d-viz .pose-stage canvas { display:block; width:100%; height:100%; }
  #weighted-headgt-3d-viz .timeline-row { display:grid; grid-template-columns:130px minmax(0,1fr); align-items:center; gap:12px; padding-top:10px; }
  #weighted-headgt-3d-viz .pose-time { color:var(--muted-foreground); font-variant-numeric:tabular-nums; }
  #weighted-headgt-3d-viz .pose-legend { display:flex; flex-wrap:wrap; gap:8px 18px; padding-top:8px; }
  #weighted-headgt-3d-viz .pose-legend span { display:inline-flex; align-items:center; gap:6px; }
  #weighted-headgt-3d-viz .swatch { width:11px; height:11px; border-radius:50%; display:inline-block; }
  #weighted-headgt-3d-viz .swatch.external { background:var(--viz-series-1); }
  #weighted-headgt-3d-viz .swatch.optimized { background:var(--viz-series-2); }
  #weighted-headgt-3d-viz .swatch.headgt { background:var(--viz-series-3); }
  @media (max-width:520px) { #weighted-headgt-3d-viz .pose-stage { height:420px; } #weighted-headgt-3d-viz .timeline-row { grid-template-columns:1fr; gap:4px; } }
</style>
<script type="application/json" id="weighted-headgt-3d-data">__DATA__</script>
<script type="module">
  import * as THREE from "https://esm.sh/three@0.180.0";
  import { OrbitControls } from "https://esm.sh/three@0.180.0/examples/jsm/controls/OrbitControls.js";
  const root=document.getElementById("weighted-headgt-3d-viz"),stage=root.querySelector(".pose-stage");
  const data=JSON.parse(document.getElementById("weighted-headgt-3d-data").textContent);
  const frames=data.frames,externalNames=data.joint_names_external,optimizedNames=data.joint_names_optimized;
  const baseEdges=[["left_shoulder","right_shoulder"],["left_shoulder","left_elbow"],["left_elbow","left_wrist"],["right_shoulder","right_elbow"],["right_elbow","right_wrist"],["left_shoulder","left_hip"],["right_shoulder","right_hip"],["left_hip","right_hip"],["left_hip","left_knee"],["left_knee","left_ankle"],["right_hip","right_knee"],["right_knee","right_ankle"]];
  const optimizedEdges=baseEdges.concat([["left_ankle","left_toe"],["right_ankle","right_toe"]]);
  function cssColor(name,fallback){const p=document.createElement("i");p.style.color=`var(${name})`;p.style.display="none";root.appendChild(p);const value=getComputedStyle(p).color||fallback;p.remove();const c=new THREE.Color();try{c.setStyle(value)}catch(_){c.set(fallback)}return c;}
  const colors={external:cssColor("--viz-series-1","#4c78a8"),optimized:cssColor("--viz-series-2","#f2cf5b"),headgt:cssColor("--viz-series-3","#54a24b"),neutral:cssColor("--muted-foreground","#7d8795")};
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(38,1,.01,20),renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0,0);renderer.outputColorSpace=THREE.SRGBColorSpace;stage.appendChild(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xffffff,0x777777,1.7));const key=new THREE.DirectionalLight(0xffffff,2);key.position.set(2,2,3);scene.add(key);
  const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.08;controls.minDistance=.8;controls.maxDistance=5;
  const medianNose=(()=>{const a=frames.map(f=>f.optimized[optimizedNames.indexOf("nose")]);return new THREE.Vector3(...[0,1,2].map(d=>a.map(v=>v[d]).sort((x,y)=>x-y)[Math.floor(a.length/2)]));})();
  const origin=medianNose.clone();
  const grid=new THREE.GridHelper(2.4,12,colors.neutral,colors.neutral);grid.rotation.x=Math.PI/2;grid.position.copy(origin).add(new THREE.Vector3(0,0,-1.55));grid.material.transparent=true;grid.material.opacity=.16;scene.add(grid);
  const axes=new THREE.AxesHelper(.28);axes.position.copy(origin);scene.add(axes);
  function makeLayer(color){const g=new THREE.Group(),jointMaterial=new THREE.MeshStandardMaterial({color,roughness:.55,metalness:.08}),lineMaterial=new THREE.LineBasicMaterial({color});return{group:g,jointMaterial,lineMaterial,spheres:[],lines:[]};}
  const layers={external:makeLayer(colors.external),optimized:makeLayer(colors.optimized),headgt:makeLayer(colors.headgt)};Object.values(layers).forEach(l=>scene.add(l.group));
  function ensure(layer,count,edgeCount,radius){while(layer.spheres.length<count){const s=new THREE.Mesh(new THREE.SphereGeometry(radius,14,10),layer.jointMaterial);layer.group.add(s);layer.spheres.push(s)}while(layer.lines.length<edgeCount){const geom=new THREE.BufferGeometry();geom.setAttribute("position",new THREE.BufferAttribute(new Float32Array(6),3));const line=new THREE.Line(geom,layer.lineMaterial);layer.group.add(line);layer.lines.push(line)}}
  ensure(layers.external,externalNames.length,baseEdges.length,.018);ensure(layers.optimized,optimizedNames.length,optimizedEdges.length,.022);ensure(layers.headgt,optimizedNames.length,optimizedEdges.length,.027);
  function updateLayer(layer,names,values,edges){const points={};layer.spheres.forEach((s,i)=>{const value=values&&values[i];s.visible=Boolean(value);if(value){s.position.set(...value);points[names[i]]=value}});layer.lines.forEach((line,i)=>{const e=edges[i],a=points[e[0]],b=points[e[1]];line.visible=Boolean(a&&b);if(a&&b){const arr=line.geometry.attributes.position.array;arr.set(a,0);arr.set(b,3);line.geometry.attributes.position.needsUpdate=true;line.geometry.computeBoundingSphere()}})}
  let index=0,playing=false,last=0;
  const slider=root.querySelector(".form-range"),timeLabel=root.querySelector(".pose-time"),play=root.querySelector('[data-action="play"]');slider.max=String(frames.length-1);
  function update(){const f=frames[index];updateLayer(layers.external,externalNames,f.external,baseEdges);updateLayer(layers.optimized,optimizedNames,f.optimized,optimizedEdges);const gtValues=optimizedNames.map(n=>f.head_gt[n]||null);updateLayer(layers.headgt,optimizedNames,gtValues,optimizedEdges);slider.value=String(index);timeLabel.textContent=`seq ${f.sequence} · ${(f.sequence/50).toFixed(2)} s${Object.keys(f.head_gt).length?" · GT 标注帧":""}`;}
  slider.addEventListener("input",()=>{index=Number(slider.value);playing=false;play.textContent="播放";play.setAttribute("aria-pressed","false");update()});
  play.addEventListener("click",()=>{playing=!playing;play.textContent=playing?"暂停":"播放";play.setAttribute("aria-pressed",String(playing));last=performance.now()});
  root.querySelectorAll('[data-layer]').forEach(c=>c.addEventListener("change",()=>{layers[c.dataset.layer].group.visible=c.checked}));
  const presets={iso:[1.6,1.8,1.25],front:[0,-2.3,0],side:[2.3,0,0],top:[0,0,2.5]};let tween=null;
  function setView(name){const offset=new THREE.Vector3(...presets[name]);tween={start:performance.now(),from:camera.position.clone(),to:origin.clone().add(offset)};controls.target.copy(origin);root.querySelectorAll('[data-view]').forEach(b=>{const on=b.dataset.view===name;b.classList.toggle("btn-primary",on);b.setAttribute("aria-pressed",String(on))})}
  root.querySelectorAll('[data-view]').forEach(b=>b.addEventListener("click",()=>setView(b.dataset.view)));
  function resize(){const w=stage.clientWidth,h=stage.clientHeight;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h,false)}new ResizeObserver(resize).observe(stage);resize();camera.position.copy(origin).add(new THREE.Vector3(...presets.iso));controls.target.copy(origin);update();
  function animate(now){requestAnimationFrame(animate);if(playing&&now-last>=20){const steps=Math.max(1,Math.floor((now-last)/20));index=(index+steps)%frames.length;last+=steps*20;update()}if(tween){const q=Math.min(1,(now-tween.start)/420),k=1-Math.pow(1-q,3);camera.position.lerpVectors(tween.from,tween.to,k);if(q===1)tween=null}controls.update();renderer.render(scene,camera)}requestAnimationFrame(animate);
</script>'''
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(template.replace("__DATA__", compact), encoding="utf-8")
    print(a.output)


if __name__ == "__main__":
    main()
