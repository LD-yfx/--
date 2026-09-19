"""Eight-connected cost-aware A*, inflated collision masks and event replanning."""
from dataclasses import dataclass
import heapq
import math
import time
import numpy as np


def inflate_obstacles(occupied, resolution, radius):
    occupied = np.asarray(occupied, dtype=bool)
    if occupied.ndim != 2 or not math.isfinite(resolution) or resolution <= 0 or not math.isfinite(radius) or radius < 0:
        raise ValueError('invalid collision map or robot geometry')
    if radius == 0:
        return occupied.copy()
    # Conservative circle vs square-cell coverage, plus map boundary clearance.
    reach = radius + resolution/math.sqrt(2)
    n = math.ceil(reach/resolution)
    out = occupied.copy()
    h,w = out.shape
    for dy in range(-n,n+1):
        for dx in range(-n,n+1):
            if math.hypot(dx,dy)*resolution > reach:
                continue
            y0,y1=max(0,dy),min(h,h+dy)
            x0,x1=max(0,dx),min(w,w+dx)
            if y1 <= y0 or x1 <= x0:
                continue
            out[y0:y1,x0:x1] |= occupied[y0-dy:y1-dy,x0-dx:x1-dx]
    yy,xx = np.indices(out.shape)
    out |= (np.minimum(xx+0.5,w-xx-0.5)*resolution < radius)
    out |= (np.minimum(yy+0.5,h-yy-0.5)*resolution < radius)
    return out


@dataclass
class PlanResult:
    path: list
    objective: float
    length: float
    expanded: int
    elapsed_ms: float

    @property
    def success(self):
        return bool(self.path)


def path_objective(path, costs, resolution=1.0, weight=3.0):
    return sum(math.hypot(b[0]-a[0], b[1]-a[1])*resolution*
               (1+weight*(float(costs[a[1],a[0]])+float(costs[b[1],b[0]]))/504.0)
               for a,b in zip(path,path[1:]))


def path_is_valid(path, blocked):
    h,w=blocked.shape
    for x,y in path:
        if not isinstance(x,(int,np.integer)) or not isinstance(y,(int,np.integer)):
            return False
        if not (0 <= x < w and 0 <= y < h) or blocked[y,x]:
            return False
    for (x,y),(nx,ny) in zip(path,path[1:]):
        if max(abs(nx-x),abs(ny-y)) > 1:
            return False
        if x != nx and y != ny and (blocked[y,nx] or blocked[ny,x]):
            return False
    return bool(path)


def astar(blocked, costs, start, goal, resolution=1.0, weight=3.0):
    began=time.perf_counter()
    blocked=np.asarray(blocked,dtype=bool)
    costs=np.asarray(costs)
    if blocked.ndim != 2 or costs.shape != blocked.shape or not np.all(np.isfinite(costs)) or np.any((costs<0)|(costs>252)):
        raise ValueError('planner requires separate collision mask and finite soft costs [0,252]')
    if not math.isfinite(weight) or weight < 0 or not math.isfinite(resolution) or resolution <= 0:
        raise ValueError('nonnegative weight and positive resolution required')
    start,goal=tuple(start),tuple(goal)
    h,w=blocked.shape
    def done(path,objective,expanded):
        length=sum(math.dist(a,b)*resolution for a,b in zip(path,path[1:]))
        return PlanResult(path,objective,length,expanded,(time.perf_counter()-began)*1000)
    if not path_is_valid([start],blocked) or not path_is_valid([goal],blocked):
        return done([],math.inf,0)
    frontier=[(math.dist(start,goal)*resolution,0.0,start)]
    distance={start:0.0}
    parent={}
    expanded=0
    while frontier:
        _,g,node=heapq.heappop(frontier)
        if g > distance[node]+1e-12:
            continue
        expanded+=1
        if node == goal:
            path=[node]
            while node in parent:
                node=parent[node]
                path.append(node)
            return done(path[::-1],g,expanded)
        x,y=node
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)):
            nx,ny=x+dx,y+dy
            if not (0<=nx<w and 0<=ny<h) or blocked[ny,nx]:
                continue
            if dx and dy and (blocked[y,nx] or blocked[ny,x]):
                continue
            step=math.hypot(dx,dy)*resolution*(1+weight*(float(costs[y,x])+float(costs[ny,nx]))/504)
            new=g+step
            if new+1e-12 < distance.get((nx,ny),math.inf):
                distance[nx,ny]=new
                parent[nx,ny]=(x,y)
                heapq.heappush(frontier,(new+math.dist((nx,ny),goal)*resolution,new,(nx,ny)))
    return done([],math.inf,expanded)


class Replanner:
    def __init__(self, resolution=1.0, weight=3.0, min_interval=1.0, improvement_fraction=0.05):
        if resolution <= 0 or weight < 0 or min_interval < 0 or not 0 <= improvement_fraction < 1:
            raise ValueError('invalid replanning parameters')
        self.resolution,self.weight=resolution,weight
        self.min_interval,self.improvement_fraction=min_interval,improvement_fraction
        self.path=[]
        self.last_switch=-math.inf

    def update(self, blocked, costs, start, goal, now):
        start,goal=tuple(start),tuple(goal)
        if not math.isfinite(now):
            raise ValueError('finite event time required')
        if now < self.last_switch:
            self.path=[]
            self.last_switch=-math.inf
        if start in self.path:
            self.path=self.path[self.path.index(start):]
        forced=(not self.path or self.path[0]!=start or self.path[-1]!=goal
                or not path_is_valid(self.path,blocked))
        if not forced and now-self.last_switch < self.min_interval:
            return self.path,'held_interval'
        candidate=astar(blocked,costs,start,goal,self.resolution,self.weight)
        if not candidate.success:
            self.path=[]
            return [],'stop_no_path'
        old_cost=math.inf if forced else path_objective(self.path,costs,self.resolution,self.weight)
        if forced or candidate.objective < old_cost*(1-self.improvement_fraction):
            self.path=candidate.path
            self.last_switch=now
            return self.path,'blocked_or_initial' if forced else 'quality_improvement'
        return self.path,'held_hysteresis'
