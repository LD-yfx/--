"""Ideal odometry + 2-D kinematics + independent disk/rectangle collision check.

This synthetic mechanism simulator does NOT implement SLAM or localization error.
Only Nav2 publishes cmd_vel. The simulator never steers the robot towards goals.
"""
import json
import hashlib
import math
import time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from quality_navigation_msgs.msg import QualityGrid

RESOLUTION=0.1
WIDTH,HEIGHT=120,80
ORIGIN=(-6.0,-4.0)
ROBOT_RADIUS=0.20
START=(-4.3,0.0,0.0)
GOAL=(4.3,0.0)
# Geometry differs slightly above/below, making the geometric baseline unambiguous.
STATIC_RECTS=[(-6,-5.8,-4,4),(5.8,6,-4,4),(-6,6,-4,-3.8),(-6,6,3.8,4),
              (-1.3,1.3,-1.15,1.65)]
TOP_BLOCK=(-0.3,0.3,1.65,3.8)
ALL_BLOCK=(-0.3,0.3,-3.8,3.8)

def intersects_disk(x,y,r,rect):
    left,right,bottom,top=rect
    return (x-min(max(x,left),right))**2+(y-min(max(y,bottom),top))**2 <= r*r

def raster(rectangles):
    xs=ORIGIN[0]+(np.arange(WIDTH)+0.5)*RESOLUTION
    ys=ORIGIN[1]+(np.arange(HEIGHT)+0.5)*RESOLUTION
    xx,yy=np.meshgrid(xs,ys)
    occupied=np.zeros((HEIGHT,WIDTH),dtype=bool)
    for left,right,bottom,top in rectangles:
        occupied|=(xx>=left)&(xx<=right)&(yy>=bottom)&(yy<=top)
    return occupied

class KinematicSimulator(Node):
    def __init__(self):
        super().__init__('quality_kinematic_sim')
        self.x,self.y,self.yaw=START
        self.v=self.w=0.0
        self.last_command=0.0
        self.last_tick=time.monotonic()
        self.collisions=0
        self.blocked_commands=0
        self.distance=0.0
        self.quality_enabled=True
        self.top_blocked=False
        self.all_blocked=False
        self.last_collision=False
        self.sequence=0
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub=self.create_publisher(OccupancyGrid,'/map',qos)
        self.quality_pub=self.create_publisher(QualityGrid,'/localization_quality/quality_stats',qos)
        self.scan_pub=self.create_publisher(LaserScan,'/scan',10)
        self.odom_pub=self.create_publisher(Odometry,'/odom',10)
        self.state_pub=self.create_publisher(String,'/simulation/state',10)
        self.runtime_pub=self.create_publisher(String,'/simulation/runtime',qos)
        runtime=String()
        runtime.data=json.dumps({'runtime_module_path':str(Path(__file__).resolve()),
                                'runtime_module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        self.runtime_pub.publish(runtime)
        self.create_subscription(Twist,'/cmd_vel',self.command,10)
        self.create_service(SetBool,'/simulation/quality_enabled',self.set_quality)
        self.create_service(SetBool,'/simulation/top_blocked',self.set_top)
        self.create_service(SetBool,'/simulation/all_blocked',self.set_all)
        self.create_service(Trigger,'/simulation/reset',self.reset)
        self.tf=TransformBroadcaster(self)
        self.static_tf=StaticTransformBroadcaster(self)
        transforms=[]
        for parent,child in [('map','odom'),('base_link','laser')]:
            tr=TransformStamped()
            tr.header.frame_id=parent
            tr.child_frame_id=child
            tr.header.stamp=self.get_clock().now().to_msg()
            tr.transform.rotation.w=1.0
            transforms.append(tr)
        self.static_tf.sendTransform(transforms)
        self.create_timer(0.025,self.tick)
        self.create_timer(0.1,self.publish_scan)
        self.create_timer(0.5,self.publish_quality)
        self.create_timer(1.0,self.publish_map)
        self.publish_map()
        self.publish_quality()
        self.get_logger().info('SYNTHETIC mechanism test: ideal odometry, no SLAM accuracy claim')

    def rectangles(self):
        return STATIC_RECTS+([ALL_BLOCK] if self.all_blocked else [TOP_BLOCK] if self.top_blocked else [])

    def command(self,msg):
        if not math.isfinite(msg.linear.x) or not math.isfinite(msg.angular.z):
            self.v=self.w=0.0
            return
        self.v=max(-0.7,min(0.7,msg.linear.x))
        self.w=max(-1.8,min(1.8,msg.angular.z))
        self.last_command=time.monotonic()

    def set_quality(self,req,res):
        self.quality_enabled=req.data
        self.publish_quality()
        res.success=True
        res.message='synthetic lower corridor quality penalty '+str(req.data)
        return res

    def set_top(self,req,res):
        if req.data and intersects_disk(self.x,self.y,ROBOT_RADIUS,TOP_BLOCK):
            res.success=False;res.message='refusing to spawn obstacle overlapping robot';return res
        self.top_blocked=req.data
        self.publish_scan()
        res.success=True;res.message='top corridor obstacle '+str(req.data)
        return res

    def set_all(self,req,res):
        if req.data and intersects_disk(self.x,self.y,ROBOT_RADIUS,ALL_BLOCK):
            res.success=False;res.message='refusing to spawn obstacle overlapping robot';return res
        self.all_blocked=req.data
        self.publish_scan()
        res.success=True;res.message='all corridors blocked '+str(req.data)
        return res

    def reset(self,req,res):
        self.x,self.y,self.yaw=START
        self.v=self.w=0.0
        self.last_command=0.0
        self.collisions=self.blocked_commands=0
        self.distance=0.0
        self.last_collision=False
        self.top_blocked=self.all_blocked=False
        self.sequence+=1
        res.success=True;res.message='reset pose, events, and counters'
        return res

    def tick(self):
        now=time.monotonic()
        dt=min(0.05,max(0.0,now-self.last_tick));self.last_tick=now
        v,w=(self.v,self.w) if now-self.last_command<=0.4 else (0.0,0.0)
        nx=self.x+v*math.cos(self.yaw+w*dt/2)*dt
        ny=self.y+v*math.sin(self.yaw+w*dt/2)*dt
        hit=any(intersects_disk(nx,ny,ROBOT_RADIUS,r) for r in self.rectangles())
        if hit and abs(v)>1e-6:
            self.blocked_commands+=1
            if not self.last_collision:self.collisions+=1
            v=0.0
        else:
            self.distance+=math.hypot(nx-self.x,ny-self.y)
            self.x,self.y=nx,ny
        self.last_collision=hit
        self.yaw=math.atan2(math.sin(self.yaw+w*dt),math.cos(self.yaw+w*dt))
        stamp=self.get_clock().now().to_msg()
        odom=Odometry();odom.header.stamp=stamp
        odom.header.frame_id='odom';odom.child_frame_id='base_link'
        odom.pose.pose.position.x=self.x;odom.pose.pose.position.y=self.y
        odom.pose.pose.orientation.z=math.sin(self.yaw/2);odom.pose.pose.orientation.w=math.cos(self.yaw/2)
        odom.twist.twist.linear.x=v;odom.twist.twist.angular.z=w
        self.odom_pub.publish(odom)
        transform=TransformStamped();transform.header=odom.header
        transform.child_frame_id='base_link'
        transform.transform.translation.x=self.x;transform.transform.translation.y=self.y
        transform.transform.rotation=odom.pose.pose.orientation
        self.tf.sendTransform(transform)
        state={'stamp':stamp.sec+stamp.nanosec*1e-9,'x':self.x,'y':self.y,'yaw':self.yaw,
               'v':v,'w':w,'collisions':self.collisions,'blocked_commands':self.blocked_commands,
               'distance':self.distance,'quality_enabled':self.quality_enabled,
               'top_blocked':self.top_blocked,'all_blocked':self.all_blocked,'reset_sequence':self.sequence,
               'evidence':'synthetic_2d_kinematics_ideal_odometry'}
        message=String();message.data=json.dumps(state);self.state_pub.publish(message)

    def publish_map(self):
        msg=OccupancyGrid();msg.header.frame_id='map';msg.header.stamp=self.get_clock().now().to_msg()
        msg.info.resolution=RESOLUTION;msg.info.width=WIDTH;msg.info.height=HEIGHT
        msg.info.origin.position.x=ORIGIN[0];msg.info.origin.position.y=ORIGIN[1]
        msg.info.origin.orientation.w=1.0
        # Dynamic obstacles are deliberately not injected into the static map.
        msg.data=(raster(STATIC_RECTS).astype(np.int8)*100).ravel().tolist()
        self.map_pub.publish(msg)

    def publish_quality(self):
        msg=QualityGrid();msg.header.frame_id='map';msg.header.stamp=self.get_clock().now().to_msg()
        msg.info.resolution=RESOLUTION;msg.info.width=WIDTH;msg.info.height=HEIGHT
        msg.info.origin.position.x=ORIGIN[0];msg.info.origin.position.y=ORIGIN[1]
        msg.info.origin.orientation.w=1.0
        msg.statistics_mode='windowed'
        n=WIDTH*HEIGHT
        msg.indices=list(range(n))
        ys=ORIGIN[1]+(np.arange(HEIGHT)+0.5)*RESOLUTION
        mean=np.full((HEIGHT,WIDTH),0.99,dtype=np.float32)
        if self.quality_enabled:mean[ys<0.0,:]=0.03
        msg.mean=mean.ravel().tolist();msg.variance=[0.0]*n;msg.sample_count=[100]*n
        # A synthetic oracle observes all cells every publication. This is an
        # explicit artificial mechanism test, not an inference from sparse traces.
        msg.last_observed=[msg.header.stamp]*n
        self.quality_pub.publish(msg)

    def publish_scan(self):
        count=720
        angles=np.linspace(-math.pi,math.pi,count,endpoint=False)
        dx=np.cos(angles+self.yaw);dy=np.sin(angles+self.yaw)
        invx=np.divide(1.0,dx,out=np.full_like(dx,1e12),where=np.abs(dx)>1e-12)
        invy=np.divide(1.0,dy,out=np.full_like(dy,1e12),where=np.abs(dy)>1e-12)
        ranges=np.full(count,12.0)
        for left,right,bottom,top in self.rectangles():
            tx1=(left-self.x)*invx;tx2=(right-self.x)*invx
            ty1=(bottom-self.y)*invy;ty2=(top-self.y)*invy
            enter=np.maximum(np.minimum(tx1,tx2),np.minimum(ty1,ty2))
            leave=np.minimum(np.maximum(tx1,tx2),np.maximum(ty1,ty2))
            valid=(leave>=np.maximum(enter,0.0))&(enter>=0.03)
            ranges=np.where(valid,np.minimum(ranges,enter),ranges)
        msg=LaserScan();msg.header.frame_id='laser';msg.header.stamp=self.get_clock().now().to_msg()
        msg.angle_min=-math.pi;msg.angle_increment=2*math.pi/count
        msg.angle_max=msg.angle_min+(count-1)*msg.angle_increment
        msg.range_min=0.03;msg.range_max=12.0;msg.scan_time=0.1
        msg.ranges=ranges.astype(np.float32).tolist();self.scan_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args);node=KinematicSimulator()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()
