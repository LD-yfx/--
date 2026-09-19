"""AMCL/encoder adapter: no Gazebo truth subscription or model-state lookup.

Estimated pose is map<-base TF evaluated at each encoder stamp. AMCL covariance
keeps its real measurement timestamp on the separate laser-localization topic.
Initial pose is a configured uncertain prior, published once AMCL is active.
"""
from collections import deque
import copy
import hashlib
import math
from pathlib import Path
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener, TransformException


def stamp_ns(stamp):
    return int(stamp.sec)*1000000000+int(stamp.nanosec)


class LocalizationAdapter(Node):
    def __init__(self):
        super().__init__('localization_adapter')
        defaults={'map_frame':'map','odom_frame':'odom','base_frame':'base_footprint',
                  'odom_topic':'/odom','amcl_pose_topic':'/amcl_pose',
                  'estimated_odom_topic':'/localization/estimated_odom',
                  'laser_odom_topic':'/laser_localization/odom',
                  'amcl_stale_timeout':0.75,'pending_timeout':0.5,'max_pending_messages':100,
                  'publish_initial_pose':True,'initial_x':-4.4,'initial_y':0.1,
                  'initial_yaw':0.05,'initial_sigma_xy':0.3,'initial_sigma_yaw':0.15}
        for name,value in defaults.items():self.declare_parameter(name,value)
        self.p={name:self.get_parameter(name).value for name in defaults}
        numeric=('amcl_stale_timeout','pending_timeout','initial_x','initial_y',
                 'initial_yaw','initial_sigma_xy','initial_sigma_yaw')
        if not all(math.isfinite(self.p[key]) for key in numeric):raise ValueError('adapter parameters must be finite')
        if self.p['amcl_stale_timeout']<=0 or self.p['pending_timeout']<=0 or self.p['max_pending_messages']<1:
            raise ValueError('adapter timeouts/queue capacity must be positive')
        if self.p['initial_sigma_xy']<=0 or self.p['initial_sigma_yaw']<=0:
            raise ValueError('initial prior must have nonzero uncertainty')
        sensor_qos=QoSProfile(depth=100,reliability=ReliabilityPolicy.BEST_EFFORT)
        pose_qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE)
        self.estimated_pub=self.create_publisher(Odometry,self.p['estimated_odom_topic'],50)
        self.laser_pub=self.create_publisher(Odometry,self.p['laser_odom_topic'],10)
        self.diagnostic_pub=self.create_publisher(DiagnosticArray,'/localization/adapter_diagnostics',10)
        initial_qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.initial_pub=self.create_publisher(PoseWithCovarianceStamped,'/initialpose',initial_qos)
        self.create_subscription(Odometry,self.p['odom_topic'],self.on_odom,sensor_qos)
        self.create_subscription(PoseWithCovarianceStamped,self.p['amcl_pose_topic'],self.on_amcl,pose_qos)
        self.tf=Buffer(cache_time=Duration(seconds=15.0),node=self)
        self.tf_listener=TransformListener(self.tf,self)
        self.amcl_state_client=self.create_client(GetState,'/amcl/get_state')
        self.state_future=None
        self.initial_sent=False
        self.pending=deque()
        self.amcl_history=deque(maxlen=100)
        self.last_odom_ns=None
        self.last_amcl_receive=None
        self.published=0
        self.dropped=0
        self.tf_failures=0
        self.last_error='waiting_for_amcl'
        self.runtime_path=str(Path(__file__).resolve())
        self.runtime_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.create_timer(0.02,self.flush)
        self.create_timer(0.25,self.initialize_prior)
        self.create_timer(0.5,self.diagnostics)

    def initialize_prior(self):
        if not self.p['publish_initial_pose'] or self.initial_sent or self.get_clock().now().nanoseconds<=0:return
        if self.state_future is None:
            if self.amcl_state_client.service_is_ready():
                self.state_future=self.amcl_state_client.call_async(GetState.Request())
            return
        if not self.state_future.done():return
        response=self.state_future.result();self.state_future=None
        if response is None or response.current_state.id!=3:return
        prior=PoseWithCovarianceStamped()
        prior.header.frame_id=self.p['map_frame'];prior.header.stamp=self.get_clock().now().to_msg()
        prior.pose.pose.position.x=float(self.p['initial_x'])
        prior.pose.pose.position.y=float(self.p['initial_y'])
        prior.pose.pose.orientation.z=math.sin(self.p['initial_yaw']/2)
        prior.pose.pose.orientation.w=math.cos(self.p['initial_yaw']/2)
        prior.pose.covariance[0]=self.p['initial_sigma_xy']**2
        prior.pose.covariance[7]=self.p['initial_sigma_xy']**2
        prior.pose.covariance[35]=self.p['initial_sigma_yaw']**2
        self.initial_pub.publish(prior)
        self.initial_sent=True
        self.get_logger().info('Published configured uncertain initial prior; no ground truth consumed')

    def on_amcl(self,message):
        if message.header.frame_id!=self.p['map_frame']:
            self.last_error='rejected_amcl_frame';return
        values=list(message.pose.covariance)
        if not all(math.isfinite(v) for v in values) or any(values[i]<-1e-10 for i in (0,7,14,21,28,35)):
            self.last_error='rejected_amcl_covariance';return
        stamp=stamp_ns(message.header.stamp)
        if self.amcl_history and stamp<=stamp_ns(self.amcl_history[-1].header.stamp):return
        self.amcl_history.append(copy.deepcopy(message))
        self.last_amcl_receive=time.monotonic()
        # Preserve the AMCL observation stamp, rather than stamping an old
        # covariance with a fresh encoder timestamp on every 50 Hz odom update.
        laser=Odometry();laser.header=message.header;laser.child_frame_id=self.p['base_frame']
        laser.pose=message.pose
        self.laser_pub.publish(laser)

    def on_odom(self,message):
        if message.header.frame_id!=self.p['odom_frame'] or message.child_frame_id!=self.p['base_frame']:
            self.last_error='rejected_encoder_frame';return
        stamp=stamp_ns(message.header.stamp)
        if self.last_odom_ns is not None and stamp<self.last_odom_ns:
            self.pending.clear();self.amcl_history.clear();self.last_amcl_receive=None
            self.initial_sent=False;self.last_error='clock_rewound_waiting_for_initialization'
        self.last_odom_ns=stamp
        if len(self.pending)>=self.p['max_pending_messages']:
            self.pending.popleft();self.dropped+=1
        self.pending.append((message,time.monotonic()))
        self.flush()

    def flush(self):
        while self.pending:
            encoder,received=self.pending[0]
            age=time.monotonic()-received
            timestamp=stamp_ns(encoder.header.stamp)
            causal=next((pose for pose in reversed(self.amcl_history)
                         if stamp_ns(pose.header.stamp)<=timestamp),None)
            if causal is None:
                if age>self.p['pending_timeout']:
                    self.pending.popleft();self.dropped+=1;continue
                return
            try:
                transform=self.tf.lookup_transform(self.p['map_frame'],self.p['base_frame'],
                                                   Time.from_msg(encoder.header.stamp))
            except TransformException as error:
                self.last_error='tf_unavailable: '+str(error)
                if age>self.p['pending_timeout']:
                    self.pending.popleft();self.dropped+=1;self.tf_failures+=1;continue
                return
            self.pending.popleft()
            estimate=Odometry();estimate.header.stamp=encoder.header.stamp
            estimate.header.frame_id=self.p['map_frame'];estimate.child_frame_id=self.p['base_frame']
            estimate.pose.pose.position.x=transform.transform.translation.x
            estimate.pose.pose.position.y=transform.transform.translation.y
            estimate.pose.pose.position.z=transform.transform.translation.z
            estimate.pose.pose.orientation=transform.transform.rotation
            estimate.pose.covariance=list(causal.pose.covariance)
            estimate.twist=encoder.twist
            self.estimated_pub.publish(estimate)
            self.published+=1
            covariance_age=(timestamp-stamp_ns(causal.header.stamp))*1e-9
            self.last_error='amcl_covariance_stale' if covariance_age>self.p['amcl_stale_timeout'] else ''

    def diagnostics(self):
        now=self.get_clock().now()
        latest=self.amcl_history[-1] if self.amcl_history else None
        source_age=(now.nanoseconds-stamp_ns(latest.header.stamp))*1e-9 if latest else math.inf
        receipt_age=time.monotonic()-self.last_amcl_receive if self.last_amcl_receive is not None else math.inf
        stale=source_age<0 or max(source_age,receipt_age)>self.p['amcl_stale_timeout']
        state='waiting_for_amcl' if latest is None else 'amcl_covariance_stale' if stale else 'tracking'
        diagnostic=DiagnosticStatus();diagnostic.name='localization/independent_amcl_adapter'
        diagnostic.level=DiagnosticStatus.WARN if stale or self.last_error else DiagnosticStatus.OK
        diagnostic.message=state
        diagnostic.values=[KeyValue(key=key,value=str(value)) for key,value in {
            'source':'amcl_scan_matching_and_encoder_tf','ground_truth_consumed':False,
            'amcl_source_age_s':source_age,'amcl_receipt_age_s':receipt_age,
            'amcl_covariance_stale':stale,'published_estimates':self.published,
            'dropped_encoder_samples':self.dropped,'tf_failures':self.tf_failures,
            'pending_samples':len(self.pending),'last_error':self.last_error,
            'initial_prior_sent':self.initial_sent,'runtime_module_path':self.runtime_path,
            'runtime_module_sha256':self.runtime_hash,
            'covariance_semantics':'last_AMCL_covariance_not_propagated_encoder_uncertainty',
        }.items()]
        array=DiagnosticArray();array.header.stamp=now.to_msg();array.status=[diagnostic]
        self.diagnostic_pub.publish(array)


def main(args=None):
    rclpy.init(args=args);node=LocalizationAdapter()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()
