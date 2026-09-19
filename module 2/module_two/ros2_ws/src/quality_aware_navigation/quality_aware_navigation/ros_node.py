"""ROS2 adapter for the same model used by the offline experiments."""
import math
import time
import hashlib
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import Costmap
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from quality_navigation_msgs.msg import QualityGrid as QualityGridMessage
from .grid import GridSpec, QualityGrid, load_module_one
from .model import CostModel, CostParameters


def stamp_seconds(stamp):
    if not 0 <= stamp.nanosec < 1_000_000_000:
        raise ValueError('timestamp nanosec must be in [0, 1000000000)')
    return float(stamp.sec) + float(stamp.nanosec)*1e-9


def spec_from_message(message):
    info=message.info
    if not math.isfinite(info.origin.position.z) or abs(info.origin.position.z)>1e-6:
        raise ValueError('quality grid origin must lie in the z=0 plane')
    q=info.origin.orientation
    norm=math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
    if not math.isfinite(norm) or abs(norm-1) > 1e-3 or abs(q.x)>1e-6 or abs(q.y)>1e-6:
        raise ValueError('quality grid origin must have a normalized planar orientation')
    yaw=2*math.atan2(q.z,q.w)
    return GridSpec(int(info.width),int(info.height),float(info.resolution),
                    info.origin.position.x,info.origin.position.y,yaw,message.header.frame_id)


def parse_statistics(message):
    spec=spec_from_message(message)
    grid=QualityGrid.empty(spec,message.statistics_mode)
    if message.statistics_mode not in ('cumulative','windowed'):
        raise ValueError('statistics topic requires cumulative or windowed data')
    n=len(message.indices)
    if any(len(getattr(message,key)) != n for key in ('mean','variance','sample_count','last_observed')):
        raise ValueError('statistics arrays have different lengths')
    indices=np.asarray(message.indices,dtype=np.int64)
    if np.any(indices >= spec.width*spec.height) or np.any(indices<0) or len(np.unique(indices)) != n:
        raise ValueError('statistics indices are repeated or outside the grid')
    grid.known.flat[indices]=True
    grid.mean.flat[indices]=message.mean
    grid.variance.flat[indices]=message.variance
    grid.count.flat[indices]=message.sample_count
    grid.last_observed.flat[indices]=[stamp_seconds(t) for t in message.last_observed]
    grid.validate()
    snapshot=stamp_seconds(message.header.stamp)
    if snapshot < 0 or np.any(grid.last_observed[grid.known] > snapshot+1e-6):
        raise ValueError('cell observations must not be later than their snapshot')
    return grid


def parse_mean_map(message):
    grid=QualityGrid.empty(spec_from_message(message),'mean_only')
    values=np.asarray(message.data,dtype=float)
    if values.size != grid.spec.width*grid.spec.height or np.any((values < -1)|(values > 100)):
        raise ValueError('mean quality data must contain -1 or 0..100 for every cell')
    values=values.reshape(grid.spec.shape)
    grid.known=values>=0
    grid.mean[grid.known]=values[grid.known]/100
    return grid.validate()


class QualityCostNode(Node):
    def __init__(self):
        super().__init__('quality_cost_node')
        defaults={
            'input_mode':'statistics', 'statistics_topic':'/localization_quality/quality_stats',
            'quality_map_topic':'/localization_quality/quality_map',
            'output_topic':'/localization_quality/navigation_cost',
            'mode':'full','beta':1.0,'gamma':1.0,'sample_scale':5.0,'age_scale':30.0,
            'unknown_risk':0.7,'max_cost':200,'mean_only_floor':0.7,
            'publish_period':0.5,'source_timeout':2.0,'stale_cost':200,
            'future_max_lead':0.2,'pending_wall_timeout':1.0,
            'csv_path':'','csv_as_of':-1.0,
        }
        for key,value in defaults.items():
            self.declare_parameter(key,value)
        p={key:self.get_parameter(key).value for key in defaults}
        if p['input_mode'] not in ('statistics','mean_only','csv'):
            raise ValueError('input_mode must be statistics, mean_only, or csv')
        if (not all(math.isfinite(p[key]) for key in ('publish_period','source_timeout','csv_as_of',
                                                     'future_max_lead','pending_wall_timeout'))
                or p['publish_period'] <= 0 or p['source_timeout'] <= 0 or not 0<=p['stale_cost']<=252
                or p['future_max_lead'] < 0 or p['pending_wall_timeout'] <= 0):
            raise ValueError('invalid timeouts or stale cost')
        self.settings=p
        runtime_path=Path(__file__).resolve()
        self.runtime_provenance={
            'runtime_module_path':str(runtime_path),
            'runtime_module_sha256':hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
        }
        params=CostParameters(**{key:p[key] for key in CostParameters.__dataclass_fields__})
        self.model=CostModel(params,p['mode'])
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher=self.create_publisher(Costmap,p['output_topic'],qos)
        self.diagnostics=self.create_publisher(DiagnosticArray,'/quality_navigation/diagnostics',10)
        self.grid=None
        self.source_stamp=None
        self.last_receive=None
        self.last_now=None
        self.rejected=0
        self.accepted=0
        # One complete parsed snapshot, never an incremental update. Waiting
        # does not replace the accepted grid or refresh source/receipt freshness.
        self.pending=None
        self.deferred=0
        self.deferred_accepted=0
        self.pending_replaced=0
        self.pending_expired=0
        self.pending_rewind_dropped=0
        self.last_error='waiting_for_quality'
        self.input_invalid=False
        if p['input_mode']=='statistics':
            self.subscription=self.create_subscription(QualityGridMessage,p['statistics_topic'],self.on_statistics,qos)
        elif p['input_mode']=='mean_only':
            self.subscription=self.create_subscription(OccupancyGrid,p['quality_map_topic'],self.on_mean,qos)
        else:
            self.grid=load_module_one(p['csv_path'])
            latest=float(self.grid.last_observed[self.grid.known].max(initial=0))
            self.source_stamp=latest if p['csv_as_of']<0 else p['csv_as_of']
            self.model.costs(self.grid,self.source_stamp)
            self.last_error='offline_frozen_source_clock'
        self.timer=self.create_timer(p['publish_period'],self.publish_cost)
        # ROS timers pause with /clock. A steady timer must still expire a
        # deferred sample if simulation pauses before its timestamp is reached.
        self.pending_timer=self.create_timer(0.02,self.flush_pending,
                                             clock=Clock(clock_type=ClockType.STEADY_TIME))

    def observe_clock(self,now):
        if (self.last_now is not None and now<self.last_now-0.001
                and self.settings['input_mode']!='csv'):
            self.grid=None
            self.source_stamp=None
            self.last_receive=None
            self.pending_rewind_dropped+=int(self.pending is not None)
            self.pending=None
            self.input_invalid=False
            self.last_error='clock_rewound_waiting_for_new_snapshot'
        self.last_now=now

    def reject(self,error):
        self.rejected+=1
        self.last_error=str(error)
        self.input_invalid=True
        self.get_logger().warning('Rejected quality snapshot: '+str(error))

    def accept(self,grid,stamp,received,now):
        if self.source_stamp is not None and stamp<=self.source_stamp:
            raise ValueError('duplicate or out-of-order quality snapshot')
        if stamp>now+1e-6:
            raise ValueError('snapshot in future; source and consumer clocks must match')
        if (grid.statistics_mode != 'mean_only' and
                np.any(grid.last_observed[grid.known] > now+1e-6)):
            raise ValueError('cell observations are in the consumer future')
        self.grid=grid
        self.source_stamp=stamp
        self.last_receive=received
        self.accepted+=1
        self.last_error=''
        self.input_invalid=False
        if self.pending is not None and self.pending[1]<=stamp:
            self.pending=None

    def flush_pending(self):
        now=self.get_clock().now().nanoseconds*1e-9
        self.observe_clock(now)
        if self.pending is None:return
        grid,stamp,received=self.pending
        if time.monotonic()-received>=self.settings['pending_wall_timeout']:
            self.pending=None
            self.pending_expired+=1
            self.reject('pending snapshot expired before consumer clock caught up')
        elif stamp<=now+1e-6:
            self.pending=None
            try:
                # Keep the original receive time and all cell observation times.
                self.accept(grid,stamp,received,now)
                self.deferred_accepted+=1
            except (ValueError,TypeError,OverflowError) as error:
                self.reject(error)

    def receive(self,message,parser):
        try:
            stamp=stamp_seconds(message.header.stamp)
            now=self.get_clock().now().nanoseconds*1e-9
            self.observe_clock(now)
            received=time.monotonic()
            if stamp<0 or stamp>now+self.settings['future_max_lead']+1e-6:
                raise ValueError('snapshot in future beyond bounded clock wait, or negative timestamp')
            if self.source_stamp is not None and stamp<=self.source_stamp:
                raise ValueError('duplicate or out-of-order quality snapshot')
            grid=parser(message)
            if stamp>now+1e-6:
                if self.pending is not None:
                    if stamp<=self.pending[1]:
                        raise ValueError('duplicate or out-of-order pending snapshot')
                    self.pending_replaced+=1
                self.pending=(grid,stamp,received)
                self.deferred+=1
                return
            self.accept(grid,stamp,received,now)
        except (ValueError,TypeError,OverflowError) as error:
            self.reject(error)

    def on_statistics(self,message):
        self.receive(message,parse_statistics)

    def on_mean(self,message):
        self.receive(message,parse_mean_map)

    def publish_cost(self):
        self.flush_pending()
        now_msg=self.get_clock().now()
        now=now_msg.nanoseconds*1e-9
        self.observe_clock(now)
        state='waiting_for_quality'
        if self.grid is not None:
            offline=self.settings['input_mode']=='csv'
            stale=not offline and (self.input_invalid or now-self.source_stamp>self.settings['source_timeout'] or
                                   time.monotonic()-self.last_receive>self.settings['source_timeout'])
            try:
                costs=self.model.costs(self.grid,self.source_stamp if offline else now)
                if stale:
                    costs=np.maximum(costs,self.settings['stale_cost']).astype(np.uint8)
                spec=self.grid.spec
                message=Costmap()
                message.header.stamp=now_msg.to_msg()
                message.header.frame_id=spec.frame
                message.metadata.map_load_time=message.header.stamp
                message.metadata.update_time=message.header.stamp
                message.metadata.layer='localization_quality_soft_cost'
                message.metadata.resolution=float(spec.resolution)
                message.metadata.size_x=spec.width
                message.metadata.size_y=spec.height
                message.metadata.origin.position.x=float(spec.origin_x)
                message.metadata.origin.position.y=float(spec.origin_y)
                message.metadata.origin.orientation.z=math.sin(spec.yaw/2)
                message.metadata.origin.orientation.w=math.cos(spec.yaw/2)
                message.data=costs.ravel().tolist()
                self.publisher.publish(message)
                state=('offline_frozen_source_clock' if offline else 'source_stale_conservative_fallback' if stale else self.grid.statistics_mode)
            except ValueError as error:
                self.last_error=str(error)
                state='invalid_quality_snapshot'
        diagnostic=DiagnosticStatus()
        diagnostic.name='quality_navigation/cost_mapping'
        diagnostic.level=DiagnosticStatus.OK if state in ('cumulative','windowed') else DiagnosticStatus.WARN
        diagnostic.message=state
        diagnostic.values=[KeyValue(key=k,value=str(v)) for k,v in {
            'accepted':self.accepted,'rejected':self.rejected,'last_error':self.last_error,
            'pending':int(self.pending is not None),'deferred':self.deferred,
            'deferred_accepted':self.deferred_accepted,'pending_replaced':self.pending_replaced,
            'pending_expired':self.pending_expired,'pending_rewind_dropped':self.pending_rewind_dropped,
            'model':self.model.mode,'input_mode':self.settings['input_mode'],
            'known_cells':int(self.grid.known.sum()) if self.grid is not None else 0,
            'source_stamp':self.source_stamp,'semantics':'soft_cost_not_occupancy_or_failure_probability',
            **self.runtime_provenance,
        }.items()]
        array=DiagnosticArray()
        array.header.stamp=now_msg.to_msg()
        array.status=[diagnostic]
        self.diagnostics.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node=None
    try:
        node=QualityCostNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
