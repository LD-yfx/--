"""Fixed dynamic experiments: Gazebo entities or actual sensor perturbations.

The node subscribes to no truth, estimated pose, quality, or navigation topic.
It never writes a costmap or chooses an event from a robot's observed state.
Only a once-triggered simulation epoch drives the preregistered schedule.

Node entry: benchmark_events = quality_localization_benchmark.dynamics:main.
Parameters: scenario, metadata (metadata.json path), config (optional JSON made
by make_dynamic_config), service_timeout_sec=5, max_event_lateness_sec=.5.
Trigger /benchmark/events/start is idempotent. For sensor_recovery it first
starts the bridge and adopts the exact epoch returned by its Trigger response.
/benchmark/events is reliable/transient-local String JSON containing the full
schedule/history/current state, so a late subscriber need not recover old deltas.

Gazebo service success brackets, rather than exactly timestamps, physical model
insertion/removal: consumers should preserve request and response times. Sensor
boundary records describe the schedule, not proof that a sensor frame was
processed; raw frames and bridge counters provide that evidence.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time


SCENARIOS=('partial_block','full_block','sensor_recovery')
ACTIVATION_OFFSET_SEC=8.0
RELEASE_OFFSET_SEC=28.0
RECTANGLES={
    'partial_block':[[-.15,.15,-5.8,-2.0]],
    'full_block':[[-.15,.15,-5.8,5.8]],
    'sensor_recovery':[],
}


def make_dynamic_config(scenario,metadata,seed=42):
    """Pure configuration generator; no sampled robot state or quality values."""
    if scenario not in SCENARIOS:
        raise ValueError('unknown dynamic scenario')
    if metadata.get('map_id')!='test_ring' or list(metadata.get('bounds',[]))!=[-8,8,-6,6]:
        raise ValueError('dynamic protocol is frozen for the test_ring map')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**63:
        raise ValueError('seed must be a nonnegative signed 64-bit integer')
    start=metadata.get('start',[])
    if len(start)<2 or not all(math.isfinite(float(v)) for v in start[:2]):
        raise ValueError('finite nominal start required')
    radius=.32;speed=.4
    rectangles=copy.deepcopy(RECTANGLES[scenario])
    margin=None
    if rectangles:
        # This is a priori reachability from the frozen initial condition. It
        # assumes no pre-epoch motion and the common .4 m/s controller limit.
        distance=min(math.hypot(max(a-start[0],start[0]-b,0),
                                max(c-start[1],start[1]-d,0)) for a,b,c,d in rectangles)
        margin=distance-radius-speed*ACTIVATION_OFFSET_SEC
        if margin<=0:
            raise ValueError('scheduled obstacle intersects the nominal reachable robot envelope')
    environment=dict(version=1,seed=seed,zones=[],events=[])
    if scenario=='sensor_recovery':
        environment['zones']=[dict(name='dynamic_measurement_environment',bounds=list(metadata['bounds']),
                                   laser_sigma=0.,dropout=0.,visual_blur=0,exposure=1.)]
        environment['events']=[dict(zone='dynamic_measurement_environment',start=ACTIVATION_OFFSET_SEC,
                                    end=RELEASE_OFFSET_SEC,laser_sigma=.08,dropout=.45,
                                    visual_blur=9,exposure=.35)]
    return dict(version=1,scenario=scenario,map_id='test_ring',seed=seed,
                activation_offset_sec=ACTIVATION_OFFSET_SEC,release_offset_sec=RELEASE_OFFSET_SEC,
                model_name='benchmark_'+scenario,rectangles=rectangles,
                require_event_start=scenario=='sensor_recovery',sensor_environment=environment,
                spawn_safety=dict(method='fixed_start_reachability_bound_no_runtime_pose',
                    nominal_start=list(start[:2]),assumed_no_pre_epoch_motion=True,
                    speed_bound_mps=speed,robot_radius_m=radius,clearance_margin_m=margin))


def obstacle_sdf(config):
    """One static Gazebo model whose collision boxes are the registered bounds."""
    if not config['rectangles']:
        raise ValueError('sensor scenario has no obstacle model')
    links=[]
    for index,(left,right,bottom,top) in enumerate(config['rectangles']):
        size=f'{right-left} {top-bottom} 1.8'
        links.append(f'''<link name="barrier_{index}">
          <pose>{(left+right)/2} {(bottom+top)/2} .9 0 0 0</pose>
          <collision name="collision"><geometry><box><size>{size}</size></box></geometry></collision>
          <visual name="visual"><geometry><box><size>{size}</size></box></geometry>
          <material><ambient>.75 .2 .12 1</ambient><diffuse>.75 .2 .12 1</diffuse></material></visual>
        </link>''')
    return f'''<?xml version="1.0"?><sdf version="1.6">
      <model name="{config['model_name']}"><static>true</static>{''.join(links)}</model></sdf>'''


class EventSchedule:
    """Testable event state machine. Requests are never inferred as successes."""
    def __init__(self,config,max_lateness_sec=.5):
        if not math.isfinite(max_lateness_sec) or max_lateness_sec<0:
            raise ValueError('event lateness tolerance must be finite and nonnegative')
        self.config=copy.deepcopy(config)
        self.max_lateness_sec=max_lateness_sec
        self.epoch=None
        self.last_time=None
        self.invalid_reasons=[]
        self.history=[]
        sensor=config['scenario']=='sensor_recovery'
        self.events=[dict(index=i,kind=kind,offset_sec=offset,state='scheduled',
                          requested_ros=None,succeeded_ros=None,finished_ros=None,success=None,
                          service_message=None,request_lateness_sec=None,response_lateness_sec=None)
                     for i,(kind,offset) in enumerate(zip(
                         ('sensor_fault_begin','sensor_fault_end') if sensor else ('spawn_obstacle','delete_obstacle'),
                         (config['activation_offset_sec'],config['release_offset_sec'])))]

    def record(self,event,now,**fields):
        self.history.append(dict(event=event,t=now,**fields))

    def start(self,epoch):
        if not math.isfinite(epoch) or epoch<0:
            raise ValueError('event epoch must be a finite source timestamp')
        if self.epoch is None:
            self.epoch=float(epoch)
            self.last_time=self.epoch
            self.record('epoch_started',self.epoch,epoch=self.epoch)
        return self.epoch

    def observe_clock(self,now):
        if not math.isfinite(now) or now<0:
            raise ValueError('invalid simulation clock')
        if self.epoch is None:
            return
        if self.last_time is not None and now<self.last_time-1e-6:
            if 'clock_rewind' not in self.invalid_reasons:
                self.invalid_reasons.append('clock_rewind')
                self.record('clock_rewind',now,previous_ros=self.last_time)
        self.last_time=now

    def next_due(self,now):
        self.observe_clock(now)
        if self.epoch is None or 'clock_rewind' in self.invalid_reasons:
            return None
        for event in self.events:
            if event['state']=='requested':
                return None
            if event['state']!='scheduled':
                continue
            due=self.epoch+event['offset_sec']
            if now+1e-9<due:
                return None
            # Never insert a barrier after its whole activation interval elapsed.
            if event['index']==0 and now>=self.epoch+self.config['release_offset_sec']:
                event.update(state='failed',success=False,finished_ros=now,
                             service_message='activation_window_missed')
                self.invalid_reasons.append('activation_window_missed')
                self.record('event_failed',now,index=event['index'],reason='activation_window_missed')
                continue
            if event['index']==0 and now-due>self.max_lateness_sec:
                event.update(state='failed',success=False,finished_ros=now,
                             service_message='activation_deadline_missed')
                self.invalid_reasons.append('activation_deadline_missed')
                self.record('event_failed',now,index=event['index'],reason='activation_deadline_missed')
                continue
            if event['index']==1 and self.events[0]['success'] is not True:
                if (event['kind']=='delete_obstacle' and
                        self.events[0]['requested_ros'] is not None):
                    # A spawn timeout has unknown physical effect. Still try
                    # deleting this run's own model at the fixed release time.
                    return event['index']
                event.update(state='failed',success=False,finished_ros=now,
                             service_message='activation_not_confirmed')
                self.invalid_reasons.append('activation_not_confirmed')
                self.record('event_failed',now,index=event['index'],reason='activation_not_confirmed')
                continue
            return event['index']
        return None

    def requested(self,index,now):
        event=self.events[index]
        if self.epoch is None or event['state']!='scheduled':
            raise ValueError('event may be requested only once after the epoch starts')
        due=self.epoch+event['offset_sec']
        if now<due-1e-9:
            raise ValueError('event requested before its fixed schedule')
        event.update(state='requested',requested_ros=now,request_lateness_sec=max(0.,now-due))
        if now-due>self.max_lateness_sec:
            self.invalid_reasons.append('late_request_'+str(index))
        self.record('event_requested',now,index=index,kind=event['kind'],scheduled_ros=due)

    def failed_without_request(self,index,now,reason):
        event=self.events[index]
        if event['state']!='scheduled':
            raise ValueError('only an unrequested event can fail without a request')
        event.update(state='failed',success=False,finished_ros=now,service_message=str(reason))
        self.invalid_reasons.append(str(reason))
        self.record('event_failed',now,index=index,reason=str(reason))

    def finished(self,index,now,success,message=''):
        event=self.events[index]
        if event['state']!='requested':
            raise ValueError('response requires a pending request')
        due=self.epoch+event['offset_sec']
        event.update(state='succeeded' if success else 'failed',success=bool(success),
                     finished_ros=now,succeeded_ros=now if success else None,
                     service_message=str(message),response_lateness_sec=max(0.,now-due))
        if not success:
            self.invalid_reasons.append('event_service_failed_'+str(index))
        if now-due>self.max_lateness_sec:
            self.invalid_reasons.append('late_response_'+str(index))
        self.record('event_finished',now,index=index,kind=event['kind'],success=bool(success),message=str(message))

    def state(self,now):
        activated=self.events[0]['success'] is True
        released=self.events[1]['success'] is True
        return dict(schema_version=1,scenario=self.config['scenario'],map_id=self.config['map_id'],
            model_name=self.config['model_name'],
            epoch=self.epoch,now=now,scheduled_activation_ros=None if self.epoch is None else self.epoch+self.config['activation_offset_sec'],
            scheduled_release_ros=None if self.epoch is None else self.epoch+self.config['release_offset_sec'],
            activated=activated,released=released,completed=activated and released,
            valid_evidence=not self.invalid_reasons,invalid_reasons=sorted(set(self.invalid_reasons)),
            rectangles=copy.deepcopy(self.config['rectangles']),
            active_rectangles=copy.deepcopy(self.config['rectangles']) if activated and not released else [],
            possibly_active_rectangles=copy.deepcopy(self.config['rectangles'])
                if self.events[0]['requested_ros'] is not None and not released else [],
            events=copy.deepcopy(self.events),history=copy.deepcopy(self.history),
            physical_timing='service requests and replies bracket physical Gazebo changes',
            sensor_timing='boundary notifications only; processed sensor frames verify actual perturbation')


def create_node():
    import rclpy
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.clock import Clock,ClockType
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
    from rclpy.task import Future
    from gazebo_msgs.srv import SpawnEntity,DeleteEntity
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    class BenchmarkEvents(Node):
        def __init__(self):
            super().__init__('benchmark_events')
            defaults=dict(scenario='partial_block',metadata='',config='',service_timeout_sec=5.,max_event_lateness_sec=.5)
            for key,value in defaults.items():
                self.declare_parameter(key,value)
            p={key:self.get_parameter(key).value for key in defaults}
            metadata=json.loads(Path(p['metadata']).read_text(encoding='utf-8'))
            loaded=json.loads(Path(p['config']).read_text(encoding='utf-8')) if p['config'] else None
            config=make_dynamic_config(p['scenario'],metadata,loaded['seed'] if loaded else 42)
            if loaded is not None and loaded!=config:
                raise ValueError('dynamic configuration differs from the frozen generator')
            if not math.isfinite(p['service_timeout_sec']) or p['service_timeout_sec']<=0:
                raise ValueError('Gazebo/bridge service timeout must be positive')
            self.schedule=EventSchedule(config,p['max_event_lateness_sec'])
            self.service_timeout=p['service_timeout_sec']
            self.pending=None
            self.service_wait={}
            self.arm_future=None
            self.arm_pending=None
            self.start_group=ReentrantCallbackGroup()
            self.start_service=self.create_service(Trigger,'/benchmark/events/start',self.start,callback_group=self.start_group)
            self.bridge_client=self.create_client(Trigger,'/benchmark/sensor_bridge/start_events')
            self.spawn_client=self.create_client(SpawnEntity,'/spawn_entity')
            self.delete_client=self.create_client(DeleteEntity,'/delete_entity')
            qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.publisher=self.create_publisher(String,'/benchmark/events',qos)
            self.steady_clock=Clock(clock_type=ClockType.STEADY_TIME)
            self.step_timer=self.create_timer(.02,self.step,clock=self.steady_clock)
            self.state_timer=self.create_timer(.25,self.publish_state,clock=self.steady_clock)
            self.runtime_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            self.get_logger().info('Fixed dynamic events; no truth, quality, estimator or costmap connection')

        def now(self):
            return self.get_clock().now().nanoseconds*1e-9

        def publish_state(self):
            state=self.schedule.state(self.now())
            state.update(start_pending=bool(self.arm_pending),runtime_module_sha256=self.runtime_hash,
                         runtime_module_path=str(Path(__file__).resolve()))
            message=String();message.data=json.dumps(state,separators=(',',':'),allow_nan=False)
            self.publisher.publish(message)

        async def start(self,request,response):
            del request
            if self.schedule.epoch is not None:
                response.success=True
                response.message=json.dumps(dict(epoch=self.schedule.epoch,idempotent=True))
                return response
            if self.arm_future is None:
                self.arm_future=Future()
                now=self.now()
                self.schedule.record('start_requested',now)
                if now<=0:
                    self.finish_arm(False,None,'simulation_clock_not_ready')
                elif self.schedule.config['scenario']=='sensor_recovery':
                    self.arm_pending=dict(future=None,deadline=time.monotonic()+self.service_timeout)
                else:
                    self.finish_arm(True,now,'fixed_obstacle_schedule_started')
            result=await self.arm_future
            response.success=result['success']
            response.message=json.dumps(result)
            return response

        def finish_arm(self,success,epoch,message):
            now=self.now()
            if success:
                self.schedule.start(epoch)
            else:
                self.schedule.invalid_reasons.append('event_start_failed')
            self.schedule.record('start_finished',now,success=success,epoch=epoch,message=message)
            self.arm_pending=None
            self.arm_future.set_result(dict(success=success,epoch=self.schedule.epoch,message=message))
            self.publish_state()

        def poll_arm(self):
            if self.arm_pending is None:
                return
            pending=self.arm_pending
            if time.monotonic()>=pending['deadline']:
                self.finish_arm(False,None,'sensor_bridge_start_timeout')
                return
            if pending['future'] is None:
                if self.bridge_client.service_is_ready():
                    try:
                        pending['future']=self.bridge_client.call_async(Trigger.Request())
                    except Exception as error:
                        self.finish_arm(False,None,'sensor_bridge_request_failed: '+str(error))
                return
            if not pending['future'].done():
                return
            try:
                response=pending['future'].result()
                details=json.loads(response.message)
                epoch=details.get('epoch')
                if (not response.success or isinstance(epoch,bool) or not isinstance(epoch,(int,float))
                        or not math.isfinite(epoch) or epoch<0):
                    raise ValueError('invalid bridge event epoch response')
                if epoch>self.now()+1e-6:
                    # DDS can deliver the service response before this node's
                    # next /clock tick. Keep the original deadline and epoch.
                    return
                self.finish_arm(True,float(epoch),'sensor_bridge_shared_epoch')
            except Exception as error:
                self.finish_arm(False,None,'sensor_bridge_start_failed: '+str(error))

        def step(self):
            self.poll_arm()
            now=self.now()
            self.schedule.observe_clock(now)
            if self.pending is not None:
                pending=self.pending
                if pending['future'].done():
                    try:
                        response=pending['future'].result()
                        self.schedule.finished(pending['index'],now,bool(response.success),response.status_message)
                    except Exception as error:
                        self.schedule.finished(pending['index'],now,False,'service_exception: '+str(error))
                    self.pending=None
                    self.publish_state()
                elif time.monotonic()>=pending['deadline']:
                    # The actual mutation may have happened despite a lost
                    # reply; report unknown effect and invalidate the trial.
                    self.schedule.finished(pending['index'],now,False,'service_timeout_effect_unknown')
                    self.pending=None
                    self.publish_state()
                else:
                    return
            index=self.schedule.next_due(now)
            if index is None:
                return
            event=self.schedule.events[index]
            if event['kind'].startswith('sensor_'):
                self.schedule.requested(index,now)
                self.schedule.finished(index,now,True,'sensor_schedule_boundary_reached_not_frame_receipt')
                self.publish_state()
                return
            client=self.spawn_client if index==0 else self.delete_client
            if not client.service_is_ready():
                # Bound missing services in source time as well; waiting does
                # not shift the preregistered event or its epoch.
                due=self.schedule.epoch+event['offset_sec']
                started=self.service_wait.setdefault(index,time.monotonic())
                if now-due>self.schedule.max_lateness_sec or time.monotonic()-started>=self.service_timeout:
                    self.schedule.failed_without_request(index,now,'gazebo_service_unavailable')
                    self.publish_state()
                return
            if index==0:
                request=SpawnEntity.Request()
                request.name=self.schedule.config['model_name']
                request.xml=obstacle_sdf(self.schedule.config)
                request.initial_pose.orientation.w=1.
                request.reference_frame='world'
            else:
                request=DeleteEntity.Request();request.name=self.schedule.config['model_name']
            self.schedule.requested(index,now)
            try:
                self.pending=dict(index=index,future=client.call_async(request),deadline=time.monotonic()+self.service_timeout)
            except Exception as error:
                self.schedule.finished(index,now,False,'service_request_failed: '+str(error))
            self.publish_state()

    return BenchmarkEvents()


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node=None
    try:
        node=create_node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
