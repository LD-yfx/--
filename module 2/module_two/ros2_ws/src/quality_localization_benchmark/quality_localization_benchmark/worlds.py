"""Independent map layouts and Gazebo geometry; no generated quality scores."""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


LAYOUTS={
    'dev_asymmetric':dict(split='development', bounds=[-6,6,-4,4],
        start=[-4.6,0.0,0.0], goal=[4.6,0.0],
        calibration_waypoints=[[0.,-2.5],[4.6,0.0],[0.,-2.5],[-4.6,0.0]],
        interior=[[-1.5,1.2,-1.2,1.5],[-3.6,-3.1,2.4,2.9],[2.7,3.2,2.3,2.8]],
        degradation=[-3.9,3.9,-3.6,-1.4]),
    'dev_rooms':dict(split='development', bounds=[-7,7,-5,5],
        start=[-5.4,-2.7,0.0], goal=[5.4,2.7],
        calibration_waypoints=[[0.,4.25],[5.4,2.7],[0.,4.25],[-5.4,-2.7]],
        interior=[[-0.2,0.2,-3.7,3.7],[-4.7,-1.7,-0.15,0.15],
                  [1.8,4.7,-0.15,0.15],[-4.4,-3.8,2.0,2.6],[3.7,4.3,-2.9,-2.3]],
        degradation=[-6.5,6.5,3.8,4.7]),
    'test_ring':dict(split='test', bounds=[-8,8,-6,6],
        start=[-6.3,-2.6,0.0], goal=[6.3,2.6],
        interior=[[-2.5,2.5,-2.0,2.0],[-5.0,-4.5,0.0,3.4],
                  [4.4,4.9,-3.5,0.0],[-1.3,-0.7,4.1,4.7],[0.8,1.4,-4.7,-4.1]],
        degradation=[-3.5,3.5,-5.5,-2.2]),
    'test_corridors':dict(split='test', bounds=[-8,8,-5,5],
        start=[-6.3,0.0,0.0], goal=[6.3,0.0],
        interior=[[-4.6,4.6,-0.55,0.55],[-4.6,2.0,2.55,2.8],
                  [-1.8,4.6,-2.8,-2.55],[-5.5,-5.0,3.4,3.9],[5.0,5.5,-3.9,-3.4]],
        degradation=[-4.7,4.7,-2.4,-0.7]),
}
CONDITIONS=('normal','visual','laser','combined')


def rectangle_model(name, rect, color='0.65 0.65 0.65 1', height=1.8):
    x0,x1,y0,y1=rect
    return f'''<model name="{name}"><static>true</static>
      <pose>{(x0+x1)/2} {(y0+y1)/2} {height/2} 0 0 0</pose><link name="body">
      <collision name="collision"><geometry><box><size>{x1-x0} {y1-y0} {height}</size></box></geometry></collision>
      <visual name="visual"><geometry><box><size>{x1-x0} {y1-y0} {height}</size></box></geometry>
      <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>
      </link></model>'''


def all_rectangles(layout):
    x0,x1,y0,y1=layout['bounds']; thickness=.2
    return [[x0,x0+thickness,y0,y1],[x1-thickness,x1,y0,y1],
            [x0,x1,y0,y0+thickness],[x0,x1,y1-thickness,y1]]+layout['interior']


def occupancy(layout, resolution=.1):
    x0,x1,y0,y1=layout['bounds']
    width,height=round((x1-x0)/resolution),round((y1-y0)/resolution)
    xx,yy=np.meshgrid(x0+(np.arange(width)+.5)*resolution,
                     y0+(np.arange(height)+.5)*resolution)
    blocked=np.zeros((height,width),dtype=bool)
    for left,right,bottom,top in all_rectangles(layout):
        blocked|=(xx>=left)&(xx<=right)&(yy>=bottom)&(yy<=top)
    return blocked


def reference_route_length(layout, waypoints, resolution=.1, robot_radius=.32):
    """Conservative four-neighbour geometric reference, not a continuous optimum.

    Every selected cell is clear for a radius-robot disk: its centre must be
    farther than robot_radius plus half the cell diagonal from every SDF box.
    Straight endpoint-to-cell-centre connectors are included in the length.
    This does not predict Nav2 turning, localization or execution performance.
    """
    x0,x1,y0,y1=layout['bounds']
    width,height=round((x1-x0)/resolution),round((y1-y0)/resolution)
    xx,yy=np.meshgrid(x0+(np.arange(width)+.5)*resolution,
                     y0+(np.arange(height)+.5)*resolution)
    distance=np.full((height,width),np.inf)
    for left,right,bottom,top in all_rectangles(layout):
        dx=np.maximum(np.maximum(left-xx,xx-right),0)
        dy=np.maximum(np.maximum(bottom-yy,yy-top),0)
        distance=np.minimum(distance,np.hypot(dx,dy))
    safe=distance>robot_radius+resolution/math.sqrt(2)

    def cell(point):
        x,y=map(float,point[:2])
        if not x0<=x<x1 or not y0<=y<y1:
            raise ValueError('reference endpoint outside map bounds')
        index=(math.floor((y-y0)/resolution),math.floor((x-x0)/resolution))
        if not safe[index]:
            raise ValueError('reference endpoint lacks conservative robot clearance')
        cy,cx=index
        connector=math.hypot(x-(x0+(cx+.5)*resolution),y-(y0+(cy+.5)*resolution))
        return index,connector

    current=layout['start']
    total=0.
    for goal in waypoints:
        start_index,start_connector=cell(current)
        goal_index,goal_connector=cell(goal)
        queue=deque([(start_index,0)]);visited={start_index};steps=None
        while queue:
            index,count=queue.popleft()
            if index==goal_index:
                steps=count;break
            y,x=index
            for nxt in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                ny,nx=nxt
                if 0<=ny<height and 0<=nx<width and safe[nxt] and nxt not in visited:
                    visited.add(nxt);queue.append((nxt,count+1))
        if steps is None:
            raise ValueError('reference endpoints are disconnected for the robot footprint')
        total+=steps*resolution+start_connector+goal_connector
        current=goal
    return total


def gazebo_world(map_id, layout):
    x0,x1,y0,y1=layout['bounds']
    models=[rectangle_model('wall_'+str(i),rect,
              '0.65 0.60 0.5 1' if i%3 else '0.3 0.5 0.7 1')
            for i,rect in enumerate(all_rectangles(layout))]
    # Visual-only checker panels on the outer wall: actual rendered image texture.
    # They neither change laser collisions nor publish/encode localization scores.
    panels=[]
    for i in range(16):
        for j in range(3):
            x=x0+.5+(x1-x0-1)*i/16
            color='0.12 0.16 0.22 1' if (i+j)%2 else '0.85 0.85 0.75 1'
            panels.append(f'''<model name="panel_{i}_{j}"><static>true</static>
              <pose>{x} {y1-.205} {0.45+j*.35} 0 0 0</pose><link name="visual_only">
              <visual name="panel"><geometry><box><size>.25 .01 .25</size></box></geometry>
              <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>
              </link></model>''')
    return f'''<?xml version="1.0"?><sdf version="1.6"><world name="{map_id}">
      <physics name="ode" type="ode"><max_step_size>0.002</max_step_size><real_time_update_rate>500</real_time_update_rate></physics>
      <scene><ambient>0.5 0.5 0.5 1</ambient><background>0.2 0.2 0.2 1</background><shadows>false</shadows></scene>
      <light name="sun" type="directional"><pose>0 0 10 0 0 0</pose><diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular><direction>0.1 0.2 -1</direction><cast_shadows>false</cast_shadows></light>
      <model name="ground"><static>true</static><pose>0 0 -.05 0 0 0</pose><link name="ground">
      <collision name="ground"><geometry><box><size>{x1-x0+4} {y1-y0+4} .1</size></box></geometry></collision>
      <visual name="ground"><geometry><box><size>{x1-x0+4} {y1-y0+4} .1</size></box></geometry>
      <material><ambient>0.55 0.55 0.55 1</ambient><diffuse>0.55 0.55 0.55 1</diffuse></material></visual></link></model>
      {''.join(models)}{''.join(panels)}
      <plugin name="state" filename="libgazebo_ros_state.so"><update_rate>30</update_rate></plugin>
    </world></sdf>'''


def write_worlds(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    for map_id,layout in LAYOUTS.items():
        target=directory/map_id; target.mkdir(parents=True,exist_ok=True)
        blocked=occupancy(layout)
        # PGM raster is top row first; world geometry is bottom row first.
        pixels=np.where(blocked[::-1],0,254).astype(np.uint8)
        (target/'map.pgm').write_bytes(f'P5\n{pixels.shape[1]} {pixels.shape[0]}\n255\n'.encode()+pixels.tobytes())
        x0,x1,y0,y1=layout['bounds']
        (target/'map.yaml').write_text(
            f'image: map.pgm\nmode: trinary\nresolution: 0.1\norigin: [{x0}, {y0}, 0.0]\n'
            'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n',encoding='utf-8')
        (target/'world.sdf').write_text(gazebo_world(map_id,layout),encoding='utf-8')
        mission_waypoints=[layout['goal'],layout['start'][:2]]
        reference_mission=reference_route_length(layout,mission_waypoints)
        metadata=dict(map_id=map_id,**layout,rectangles=all_rectangles(layout),
                      resolution=.1,width=pixels.shape[1],height=pixels.shape[0],
                      quality_resolution=.2,quality_width=pixels.shape[1]//2,quality_height=pixels.shape[0]//2,
                      origin=[x0,y0],world_to_map=[0.,0.,0.],robot_collision_radius=.31,
                      mission_waypoints=mission_waypoints,
                      reference_mission_length_m=reference_mission,
                      mission_time_limit_sec=max(90.,4*reference_mission/.4+30.),
                      reference_path_parameters=dict(resolution=.1,robot_radius=.32,connectivity=4,
                          cell_clearance_margin_m=.1/math.sqrt(2),endpoint_connectors_included=True,
                          continuous_shortest_path=False,reference_speed_mps=.4),
                      quality_values_preassigned=False)
        if 'calibration_waypoints' in layout:
            reference_calibration=reference_route_length(layout,layout['calibration_waypoints'])
            metadata.update(reference_calibration_length_m=reference_calibration,
                            calibration_time_limit_sec=max(90.,4*reference_calibration/.4+30.))
        (target/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
        for condition in CONDITIONS:
            zone=dict(name='measurement_degradation',bounds=layout['degradation'],
                      laser_sigma=.08 if condition in ('laser','combined') else 0.,
                      dropout=.45 if condition in ('laser','combined') else 0.,
                      visual_blur=9 if condition in ('visual','combined') else 0,
                      exposure=.35 if condition in ('visual','combined') else 1.)
            config=dict(version=1,seed=42,zones=[] if condition=='normal' else [zone],events=[])
            (target/(condition+'.json')).write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    hashes={str(path.relative_to(directory)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob('*')) if path.is_file() and path.name!='manifest.json'}
    (directory/'manifest.json').write_text(json.dumps(dict(layouts=list(LAYOUTS),sha256=hashes),indent=2)+'\n',encoding='utf-8')


def make_robot(reference, destination):
    """Reuse sensor intrinsics; replace truth odometry and stabilize actual dynamics."""
    tree=ET.parse(reference); root=tree.getroot(); root.set('name','research_robot')
    root.find("joint[@name='base_footprint_joint']/origin").set('xyz','0 0 0.10')
    macro=root.find('{http://www.ros.org/wiki/xacro}macro')
    macro.find('joint/origin').set('rpy','0 0 0')
    macro.find('joint/axis').set('xyz','0 1 0')
    for element in macro.findall('link/collision')+macro.findall('link/visual'):
        ET.SubElement(element,'origin',{'rpy':'1.57079632679 0 0'})
    drive=root.find("gazebo/plugin[@name='diff_drive']")
    ET.SubElement(drive,'odometry_source').text='0'
    # Joint-driven encoder integration from Gazebo, not WorldPose.
    drive.find('max_wheel_acceleration').text='1.0'
    for sensor_parent in root.findall('gazebo'):
        for sensor in list(sensor_parent.findall('sensor')):
            if sensor.get('type')=='depth': sensor_parent.remove(sensor)
    for name,x in [('rear_caster',-.19),('front_caster',.19)]:
        link=ET.SubElement(root,'link',{'name':name})
        inertial=ET.SubElement(link,'inertial'); ET.SubElement(inertial,'mass',{'value':'.05'})
        ET.SubElement(inertial,'inertia',dict(ixx='.00002',iyy='.00002',izz='.00002',ixy='0',ixz='0',iyz='0'))
        for kind in ('collision','visual'):
            elem=ET.SubElement(link,kind); geo=ET.SubElement(elem,'geometry');ET.SubElement(geo,'sphere',{'radius':'.03'})
        joint=ET.SubElement(root,'joint',{'name':name+'_joint','type':'fixed'})
        ET.SubElement(joint,'parent',{'link':'base_link'});ET.SubElement(joint,'child',{'link':name})
        ET.SubElement(joint,'origin',{'xyz':f'{x} 0 -.07'})
        gaz=ET.SubElement(root,'gazebo',{'reference':name})
        ET.SubElement(gaz,'mu1').text='.01';ET.SubElement(gaz,'mu2').text='.01'
    ET.register_namespace('xacro','http://www.ros.org/wiki/xacro')
    target=Path(destination);target.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(tree,space='  ');tree.write(target,encoding='utf-8',xml_declaration=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--robot-reference');parser.add_argument('--robot-output')
    args=parser.parse_args();write_worlds(args.output)
    if args.robot_reference and args.robot_output:make_robot(args.robot_reference,args.robot_output)
    print(json.dumps({'generated_layouts':list(LAYOUTS),'output':str(Path(args.output).resolve())}))

if __name__=='__main__':main()
