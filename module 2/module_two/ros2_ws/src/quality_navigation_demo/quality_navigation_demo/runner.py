"""Bounded closed-loop acceptance run using real Nav2 action servers."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rcl_interfaces.srv import GetParameters
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMessage
from nav2_msgs.msg import Costmap
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from lifecycle_msgs.srv import GetState
from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticArray
from std_srvs.srv import SetBool, Trigger
from .simulator import GOAL, STATIC_RECTS, TOP_BLOCK, ALL_BLOCK
from .reporting import completion_status


class Runner(Node):
    def __init__(self):
        super().__init__('quality_navigation_acceptance')
        self.state=None
        self.map=None
        self.samples=[]
        self.paths=[]
        self.record=False
        self.runtime={}
        self.create_subscription(String,'/simulation/state',self.on_state,30)
        self.create_subscription(PathMessage,'/plan',self.on_path,20)
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Costmap,'/global_costmap/costmap_raw',self.on_map,qos)
        self.create_subscription(String,'/simulation/runtime',self.on_runtime,qos)
        self.create_subscription(DiagnosticArray,'/quality_navigation/diagnostics',self.on_diagnostics,10)
        self.action=ActionClient(self,NavigateToPose,'/navigate_to_pose')

    def on_state(self,message):
        self.state=json.loads(message.data)
        if self.record:self.samples.append(dict(self.state))

    def on_path(self,message):
        if self.record and message.poses:
            points=[[p.pose.position.x,p.pose.position.y] for p in message.poses]
            middle=[y for x,y in points if abs(x)<0.6]
            self.paths.append({'wall_time':time.time(),'points':points,
                               'middle_y':sum(middle)/len(middle) if middle else None})

    def on_map(self,message):self.map=message

    def on_runtime(self,message):self.runtime['simulator']=json.loads(message.data)

    def on_diagnostics(self,message):
        for status in message.status:
            if status.name=='quality_navigation/cost_mapping':
                values={entry.key:entry.value for entry in status.values}
                self.runtime['cost_node']={key:values[key] for key in ('runtime_module_path','runtime_module_sha256') if key in values}

    def provenance(self):
        self.runtime['runner']={'runtime_module_path':str(Path(__file__).resolve()),
                                'runtime_module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        report_gate=Path(completion_status.__code__.co_filename).resolve()
        self.runtime['report_gate']={'runtime_module_path':str(report_gate),
                                     'runtime_module_sha256':hashlib.sha256(report_gate.read_bytes()).hexdigest()}
        workspace=next((p for p in Path(__file__).resolve().parents if p.name=='ros2_ws'),None)
        relatives={'runner':'quality_navigation_demo/quality_navigation_demo/runner.py',
                   'report_gate':'quality_navigation_demo/quality_navigation_demo/reporting.py',
                   'simulator':'quality_navigation_demo/quality_navigation_demo/simulator.py',
                   'cost_node':'quality_aware_navigation/quality_aware_navigation/ros_node.py'}
        result={}
        for key,relative in relatives.items():
            info=dict(self.runtime.get(key,{}))
            source=workspace/'src'/relative if workspace is not None else None
            info['source_path']=str(source) if source else None
            info['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest() if source and source.exists() else None
            runtime_path=info.get('runtime_module_path','')
            info['runtime_is_installed']=bool(workspace and runtime_path.startswith(str(workspace/'install')+'/'))
            info['source_matches_runtime']=bool(info.get('runtime_module_sha256') and info['source_sha256']==info['runtime_module_sha256'])
            result[key]=info
        return result

    def spin(self,seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:rclpy.spin_once(self,timeout_sec=min(0.05,end-time.monotonic()))

    def future(self,future,timeout=20.0):
        end=time.monotonic()+timeout
        while not future.done() and time.monotonic()<end:rclpy.spin_once(self,timeout_sec=0.05)
        if not future.done():raise TimeoutError('ROS operation exceeded %.1fs'%timeout)
        result=future.result()
        if result is None:raise RuntimeError('ROS operation returned no result')
        return result

    def service(self,typ,name,request,timeout=10.0):
        client=self.create_client(typ,name)
        try:
            if not client.wait_for_service(timeout_sec=timeout):raise TimeoutError('service unavailable: '+name)
            result=self.future(client.call_async(request),timeout)
            if hasattr(result,'success') and not result.success:raise RuntimeError(name+': '+result.message)
            return result
        finally:self.destroy_client(client)

    def boolean(self,name,value):
        req=SetBool.Request();req.data=value;return self.service(SetBool,name,req)

    def sample_cost(self,x,y):
        if self.map is None:return None
        m=self.map.metadata
        ix=math.floor((x-m.origin.position.x)/m.resolution)
        iy=math.floor((y-m.origin.position.y)/m.resolution)
        if not(0<=ix<m.size_x and 0<=iy<m.size_y):return None
        return self.map.data[iy*m.size_x+ix]

    def wait_ready(self,timeout=70.0):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            self.spin(0.3)
            if self.state is not None and self.map is not None and self.action.server_is_ready():
                try:
                    state=self.service(GetState,'/bt_navigator/get_state',GetState.Request(),2.0)
                    if state.current_state.id==3:return
                except TimeoutError:pass
        raise TimeoutError('Nav2 did not reach active state with live simulation and global costmap')

    def reset(self,quality):
        self.record=False
        self.service(Trigger,'/simulation/reset',Trigger.Request())
        self.boolean('/simulation/quality_enabled',quality)
        self.spin(0.6)
        for name in ['/global_costmap/clear_entirely_global_costmap','/local_costmap/clear_entirely_local_costmap']:
            self.service(ClearEntireCostmap,name,ClearEntireCostmap.Request())
        self.spin(2.0)
        self.samples=[];self.paths=[]

    def navigate(self,name,quality=True,event=None,timeout=75.0,reset=True):
        if reset:self.reset(quality)
        self.record=True
        start=time.monotonic()
        goal=NavigateToPose.Goal();goal.pose=PoseStamped()
        goal.pose.header.frame_id='map';goal.pose.header.stamp=self.get_clock().now().to_msg()
        goal.pose.pose.position.x=GOAL[0];goal.pose.pose.position.y=GOAL[1]
        goal.pose.pose.orientation.w=1.0
        handle=self.future(self.action.send_goal_async(goal))
        if not handle.accepted:raise RuntimeError('Nav2 rejected goal for '+name)
        result_future=handle.get_result_async()
        event_done=False;event_stamp=None;timed_out=False
        while not result_future.done() and time.monotonic()-start<timeout:
            self.spin(0.05)
            if event and not event_done and time.monotonic()-start>=2.0:
                self.boolean(event,True);event_done=True;event_stamp=time.time()
        if not result_future.done():
            timed_out=True
            self.future(handle.cancel_goal_async(),10.0)
        result=self.future(result_future,10.0)
        self.spin(1.0)
        self.record=False
        duration=time.monotonic()-start
        final=dict(self.state)
        middle=[p['middle_y'] for p in self.paths if p['middle_y'] is not None]
        result={
            'case':name,'status':int(result.status),'succeeded':result.status==4,
            'timed_out':timed_out,'duration_seconds':duration,'event_applied':event_done,
            'event_wall_time':event_stamp,'final_state':final,
            'goal_distance':math.hypot(final['x']-GOAL[0],final['y']-GOAL[1]),
            'collision_count':final['collisions'],'blocked_motion_steps':final['blocked_commands'],
            'path_publications':len(self.paths),'first_path_middle_y':middle[0] if middle else None,
            'min_path_middle_y':min(middle) if middle else None,
            'max_path_middle_y':max(middle) if middle else None,
            'trajectory':list(self.samples),'plans':list(self.paths)}
        print(json.dumps({k:v for k,v in result.items() if k not in ('trajectory','plans')},ensure_ascii=False),flush=True)
        return result

    def plugin_evidence(self):
        request=GetParameters.Request()
        request.names=['plugins','quality_layer.plugin']
        values=self.service(GetParameters,'/global_costmap/global_costmap/get_parameters',request)
        plugins=list(values.values[0].string_array_value)
        plugin=values.values[1].string_value
        self.boolean('/simulation/quality_enabled',False);self.spin(2.0)
        low_penalty=self.sample_cost(-3.0,-2.0)
        self.boolean('/simulation/quality_enabled',True);self.spin(2.0)
        high_penalty=self.sample_cost(-3.0,-2.0)
        high_quality=self.sample_cost(-3.0,2.0)
        result={'configured_plugins':plugins,'plugin_type':plugin,
                'same_free_cell_quality_off_cost':low_penalty,'same_free_cell_quality_on_cost':high_penalty,
                'high_quality_free_cell_cost':high_quality,
                'verified':plugin=='quality_nav2_layer::QualityCostLayer' and
                    low_penalty is not None and high_penalty is not None and
                    low_penalty<30 and high_penalty>150 and high_quality<30}
        return result


def save_results(output,cases,evidence,checks,error=None,mode='full',provenance=None,
                 finalized=False,require_installed=False):
    output.mkdir(parents=True,exist_ok=True)
    report={'evidence_scope':'Synthetic 2-D kinematics, ideal odometry, artificial quality field; real Nav2 planner/controller/BT. No SLAM/localization accuracy claim.',
            'mode':mode,'provenance':provenance or {},
            'ros_domain_id':os.environ.get('ROS_DOMAIN_ID','0'),
            'plugin_evidence':evidence,'acceptance':checks,
            'error':error,'cases':cases}
    report.update(completion_status(mode,cases,checks,finalized,require_installed,error))
    (output/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    for case in cases:
        rows=case['trajectory']
        if rows:
            with (output/(case['case']+'_trajectory.csv')).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if cases:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig,axes=plt.subplots(1,len(cases),figsize=(5*len(cases),4.5),squeeze=False)
        for ax,case in zip(axes[0],cases):
            for left,right,bottom,top in STATIC_RECTS:
                ax.add_patch(Rectangle((left,bottom),right-left,top-bottom,color='#444444'))
            block=ALL_BLOCK if case['case']=='fully_blocked' else TOP_BLOCK if case['case']=='dynamic_top_block' else None
            if block:
                left,right,bottom,top=block
                ax.add_patch(Rectangle((left,bottom),right-left,top-bottom,color='#d55e00',alpha=.6))
            if case['trajectory']:
                ax.plot([p['x'] for p in case['trajectory']],[p['y'] for p in case['trajectory']],color='#0072b2',lw=2)
            if case['plans']:
                first=case['plans'][0]['points'];ax.plot([p[0] for p in first],[p[1] for p in first],'--',color='#009e73',alpha=.8)
            ax.plot(*GOAL,'*',color='#cc79a7',ms=12)
            ax.set(xlim=(-6,6),ylim=(-4,4),aspect='equal',title=case['case'],xlabel='x (m)',ylabel='y (m)')
            ax.grid(alpha=.15)
        fig.suptitle('Synthetic mechanism test: real Nav2, ideal odometry; solid=executed, dashed=initial plan')
        fig.tight_layout();fig.savefig(output/'trajectories.png',dpi=160);plt.close(fig)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--existing-stack',action='store_true')
    parser.add_argument('--static-only',action='store_true')
    parser.add_argument('--require-installed',action='store_true',help='Fail unless runtime hashes match source and all modules run from install/')
    args=parser.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('ROS_DOMAIN_ID','74')
    os.environ.setdefault('RCUTILS_COLORIZED_OUTPUT','0')
    process=None;log=None;node=None;cases=[];evidence={};checks={};error=None
    mode='static_only' if args.static_only else 'full'
    try:
        if not args.existing_stack:
            log=(output/'nav2_launch.log').open('w',encoding='utf-8')
            source_launch=Path(__file__).resolve().parents[1]/'launch'/'headless.launch.py'
            launch_command=['ros2','launch',str(source_launch)] if source_launch.exists() else ['ros2','launch','quality_navigation_demo','headless.launch.py']
            process=subprocess.Popen(launch_command,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rclpy.init();node=Runner();node.wait_ready()
        evidence=node.plugin_evidence();checks['plugin_loaded_and_changes_master_cost']=evidence['verified']
        baseline=node.navigate('geometric_baseline',quality=False);cases.append(baseline)
        quality=node.navigate('quality_aware',quality=True);cases.append(quality)
        checks['static_goal_reached']=baseline['succeeded'] and quality['succeeded'] and baseline['goal_distance']<=.30 and quality['goal_distance']<=.30
        checks['quality_changes_route']=baseline['first_path_middle_y'] is not None and quality['first_path_middle_y'] is not None and baseline['first_path_middle_y']<-.5 and quality['first_path_middle_y']>.5
        checks['static_no_collisions']=baseline['collision_count']==quality['collision_count']==0
        save_results(output,cases,evidence,checks,mode=mode,provenance=node.provenance(),
                     require_installed=args.require_installed)
        if not args.static_only:
            dynamic=node.navigate('dynamic_top_block',quality=True,event='/simulation/top_blocked');cases.append(dynamic)
            checks['dynamic_replanned_and_arrived']=dynamic['succeeded'] and dynamic['goal_distance']<=.30 and dynamic['event_applied'] and dynamic['max_path_middle_y'] is not None and dynamic['max_path_middle_y']>.5 and dynamic['min_path_middle_y']<-.5
            checks['dynamic_no_collisions']=dynamic['collision_count']==0 and dynamic['blocked_motion_steps']==0
            # Observe clearing before reset/clear services, so removal evidence
            # cannot be produced by manually deleting the obstacle layer.
            before_removal_cost=node.sample_cost(0.25,3.0)
            node.boolean('/simulation/top_blocked',False);node.spin(3.0)
            removal_cost=node.sample_cost(0.25,3.0)
            evidence['top_obstacle_cost_before_removal']=before_removal_cost
            evidence['removed_top_obstacle_sample_cost']=removal_cost
            checks['removed_obstacle_cleared']=before_removal_cost is not None and before_removal_cost>=253 and removal_cost is not None and removal_cost<253
            reopened=node.navigate('after_obstacle_removal',quality=True);cases.append(reopened)
            checks['reopened_route_recovers']=reopened['succeeded'] and reopened['goal_distance']<=.30 and reopened['first_path_middle_y'] is not None and reopened['first_path_middle_y']>.5 and reopened['collision_count']==0
            node.reset(True)
            node.boolean('/simulation/all_blocked',True);node.spin(2.0)
            blocked=node.navigate('fully_blocked',quality=True,timeout=35.0,reset=False);cases.append(blocked)
            checks['fully_blocked_stops_with_bounded_failure']=not blocked['succeeded'] and not blocked['timed_out'] and abs(blocked['final_state']['v'])<.01 and abs(blocked['final_state']['w'])<.01
            checks['fully_blocked_no_collisions']=blocked['collision_count']==0 and blocked['blocked_motion_steps']==0
        if args.require_installed or mode=='full':
            checks['installed_code_matches_source']=all(p['runtime_is_installed'] and p['source_matches_runtime'] for p in node.provenance().values())
    except Exception as exc:
        import traceback
        error=type(exc).__name__+': '+str(exc);traceback.print_exc()
    finally:
        report=save_results(output,cases,evidence,checks,error,mode,node.provenance() if node else {},
                            finalized=True,require_installed=args.require_installed)
        if node is not None:node.destroy_node()
        rclpy.try_shutdown()
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
        if log is not None:log.close()
        print(json.dumps({'output':str(output),'completed':report['completed'],'all_passed':report['all_passed'],'acceptance':checks,'error':error},ensure_ascii=False),flush=True)
    raise SystemExit(0 if report['all_passed'] else 1)

if __name__=='__main__':main()
