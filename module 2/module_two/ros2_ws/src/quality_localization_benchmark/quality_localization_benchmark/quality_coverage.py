"""Offline coverage of observed sparse QualityGrid snapshots, never a Q generator.

This describes the statistics interface, not the cost consumer's acceptance or
source-timeout state. Consumer diagnostics must be reported separately. Unknown
cells never become zero-valued quality observations. Snapshot/cell counts are
descriptive repeated observations, not independent experimental replications.

Snapshot availability is max(source time, recorder receipt time). Without the
optional received_ros field the weaker source-time-only convention is explicit.
Truth is used only offline: metadata.world_to_map is the frozen [x,y,yaw]
transform from world into map, not a fitted trajectory alignment.
"""
from bisect import bisect_right
import math


def number(value):
    if isinstance(value, bool):
        return None
    try:
        value=float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def distribution(values):
    values=sorted(value for value in values if number(value) is not None)
    def quantile(p):
        index=(len(values)-1)*p
        left=math.floor(index);right=math.ceil(index)
        return values[left]+(values[right]-values[left])*(index-left)
    return dict(n=len(values), minimum=values[0] if values else None,
                median=quantile(.5) if values else None,
                p95=quantile(.95) if values else None,
                maximum=values[-1] if values else None)


def transform_point(x,y,transform,inverse=False):
    tx,ty,yaw=transform
    c,s=math.cos(yaw),math.sin(yaw)
    if inverse:
        x,y=x-tx,y-ty
        return c*x+s*y,-s*x+c*y
    return tx+c*x-s*y,ty+s*x+c*y


def parse_snapshot(row, metadata):
    if not isinstance(row,dict):
        raise ValueError('snapshot_not_object')
    source=number(row.get('t'))
    receipt=number(row.get('received_ros')) if 'received_ros' in row else source
    if source is None or source<0 or receipt is None or receipt<0:
        raise ValueError('invalid_snapshot_time')
    width,height=row.get('width'),row.get('height')
    if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in (width,height)):
        raise ValueError('invalid_dimensions')
    resolution=number(row.get('resolution'))
    origin=row.get('origin',list(metadata.get('origin',[0.,0.]))+[0.])
    if not isinstance(origin,(list,tuple)) or len(origin)!=3 or any(number(v) is None for v in origin):
        raise ValueError('invalid_origin')
    if resolution is None or resolution<=0 or row.get('frame','map')!='map':
        raise ValueError('invalid_resolution_or_frame')
    mode=row.get('mode')
    if mode not in ('windowed','cumulative','mean_only'):
        raise ValueError('invalid_statistics_mode')
    fields=('indices','mean','variance','sample_count','last_observed')
    arrays=[row.get(field) for field in fields]
    if not all(isinstance(a,(list,tuple)) for a in arrays) or len({len(a) for a in arrays})!=1:
        raise ValueError('malformed_sparse_arrays')
    cells={}
    for index,mean,variance,count,last in zip(*arrays):
        mean,variance,last=number(mean),number(variance),number(last)
        if (isinstance(index,bool) or not isinstance(index,int) or not 0<=index<width*height
                or index in cells or isinstance(count,bool) or not isinstance(count,int) or count<1
                or mean is None or not 0<=mean<=1 or variance is None or variance<0
                or last is None or not 0<=last<=source+1e-6):
            raise ValueError('invalid_sparse_cell')
        cells[index]=dict(mean=mean,variance=variance,count=count,last_observed=last)
    return dict(t=source,available=max(source,receipt),geometry=(width,height,resolution,*map(float,origin)),
                cells=cells,mode=mode,receipt_recorded='received_ros' in row)


def point_status(snapshot,x,y):
    if snapshot is None:
        return 'no_snapshot',None
    if snapshot.get('invalid'):
        return 'invalid_snapshot',None
    width,height,resolution,ox,oy,yaw=snapshot['geometry']
    x,y=transform_point(x,y,(ox,oy,yaw),inverse=True)
    ix,iy=math.floor(x/resolution),math.floor(y/resolution)
    if not 0<=ix<width or not 0<=iy<height:
        return 'outside_grid',None
    cell=snapshot['cells'].get(iy*width+ix)
    return ('known',cell) if cell is not None else ('unknown',None)


def safe_cells(geometry,metadata):
    """Cell centres with a conservative disk clear of exact static rectangles."""
    if 'rectangles' not in metadata:
        return None
    width,height,resolution,ox,oy,yaw=geometry
    world_to_map=metadata.get('world_to_map',[0.,0.,0.])
    radius=metadata.get('reference_path_parameters',{}).get('robot_radius',
                  metadata.get('robot_collision_radius',.32))
    output=set()
    for iy in range(height):
        for ix in range(width):
            x,y=transform_point((ix+.5)*resolution,(iy+.5)*resolution,(ox,oy,yaw))
            x,y=transform_point(x,y,world_to_map,inverse=True)
            if all(math.hypot(max(a-x,x-b,0),max(c-y,y-d,0))>radius
                   for a,b,c,d in metadata['rectangles']):
                output.add(iy*width+ix)
    return output


def _pose_coverage(rows,kind,start,end,timeline,times,metadata):
    by_time={};invalid=0;previous=None;reversals=0
    for row in rows or []:
        if not isinstance(row,dict):
            invalid+=1;continue
        t,x,y=(number(row.get(k)) for k in ('t','x','y'))
        if None in (t,x,y):
            invalid+=1;continue
        if not start<=t<=end:
            continue
        if previous is not None and t<previous-1e-9:
            reversals+=1
        previous=t
        expected='world' if kind=='truth' else 'map'
        if row.get('frame',expected)!=expected:
            invalid+=1;continue
        if kind=='truth':
            x,y=transform_point(x,y,metadata.get('world_to_map',[0.,0.,0.]))
        by_time[t]=(x,y)
    stamps=sorted(by_time)
    seconds={key:0. for key in ('known','unknown','outside_grid','no_snapshot','invalid_snapshot')}
    ages=[];snapshot_ages=[]
    if not reversals:
        for i,t in enumerate(stamps):
            left=max(start,t-.05,(stamps[i-1]+t)/2 if i else start)
            right=min(end,t+.05,(t+stamps[i+1])/2 if i+1<len(stamps) else end)
            index=bisect_right(times,t)-1
            snapshot=timeline[index] if index>=0 else None
            status,cell=point_status(snapshot,*by_time[t])
            seconds[status]+=max(0.,right-left)
            if cell is not None:
                ages.append(max(0.,t-cell['last_observed']))
            if snapshot is not None and not snapshot.get('invalid'):
                snapshot_ages.append(max(0.,t-snapshot['t']))
    covered=sum(seconds.values())
    return dict(status='clock_discontinuity' if reversals else 'available' if stamps else 'unavailable',
                pose_samples=len(stamps),invalid_rows=invalid,time_reversals=reversals,
                observed_pose_seconds=covered,missing_pose_seconds=max(0.,end-start-covered),
                known_seconds=seconds['known'],seconds_by_status=seconds,
                known_fraction_of_mission_time=seconds['known']/(end-start),
                known_fraction_of_observed_pose_time=seconds['known']/covered if covered else None,
                known_observation_age_s=distribution(ages),snapshot_source_age_s=distribution(snapshot_ages),
                temporal_support_half_width_sec=.05)


def _plan_coverage(rows,start,end,timeline,times):
    output=[];invalid=0
    for row in rows or []:
        if not isinstance(row,dict):
            invalid+=1;continue
        source=number(row.get('t'))
        receipt=number(row.get('received_ros')) if 'received_ros' in row else source
        if source is None or receipt is None:
            invalid+=1;continue
        t=max(source,receipt)
        if not start<=t<=end:
            continue
        points=row.get('points')
        if (row.get('frame','map')!='map' or not isinstance(points,(list,tuple))
                or any(not isinstance(p,(list,tuple)) or len(p)<2 or
                       number(p[0]) is None or number(p[1]) is None for p in points)):
            invalid+=1;continue
        index=bisect_right(times,t)-1
        snapshot=timeline[index] if index>=0 else None
        lengths={key:0. for key in ('known','unknown','outside_grid','no_snapshot','invalid_snapshot')}
        for a,b in zip(points,points[1:]):
            length=math.hypot(b[0]-a[0],b[1]-a[1])
            cuts={0.,1.}
            if snapshot and not snapshot.get('invalid'):
                width,height,resolution,ox,oy,yaw=snapshot['geometry']
                local_a=transform_point(a[0],a[1],(ox,oy,yaw),inverse=True)
                local_b=transform_point(b[0],b[1],(ox,oy,yaw),inverse=True)
                for axis,size in ((0,width),(1,height)):
                    first,last=local_a[axis],local_b[axis]
                    if abs(last-first)<1e-15:
                        continue
                    lower=max(0,math.ceil(min(first,last)/resolution))
                    upper=min(size,math.floor(max(first,last)/resolution))
                    for edge in range(lower,upper+1):
                        fraction=(edge*resolution-first)/(last-first)
                        if 0.<fraction<1.:
                            cuts.add(fraction)
            cuts=sorted(cuts)
            for left,right in zip(cuts,cuts[1:]):
                fraction=(left+right)/2
                status,_=point_status(snapshot,a[0]+fraction*(b[0]-a[0]),a[1]+fraction*(b[1]-a[1]))
                lengths[status]+=length*(right-left)
        total=sum(lengths.values())
        output.append(dict(t=t,source_t=source,length_m=total,length_m_by_status=lengths,
                           known_length_fraction=lengths['known']/total if total else None))
    return dict(plan_count=len(output),invalid_rows=invalid,
                known_length_fraction=distribution([r['known_length_fraction'] for r in output]),
                integration='exact segment intersections with quality-grid cell boundaries',
                rows=output)


def summarize_quality_coverage(stats_rows,plans,estimates,truth,metadata,mission_start,mission_end):
    """Summarize one source-time mission; preserve absent and invalid evidence."""
    start,end=number(mission_start),number(mission_end)
    if start is None or end is None or not 0<=start<end:
        return dict(status='invalid_mission_interval',schema_version=1)
    timeline=[];invalid=[];reversals=0;previous=None
    for row in stats_rows or []:
        try:
            snapshot=parse_snapshot(row,metadata)
        except (ValueError,TypeError,OverflowError) as error:
            invalid.append(str(error))
            if isinstance(row,dict):
                t=number(row.get('t'));receipt=number(row.get('received_ros',t))
                if t is not None and receipt is not None:
                    timeline.append(dict(t=t,available=max(t,receipt),invalid=True))
            continue
        if previous is not None and snapshot['t']<previous-1e-9:
            reversals+=1
        previous=snapshot['t']
        timeline.append(snapshot)
    if reversals:
        return dict(status='clock_discontinuity',schema_version=1,
                    snapshot_time_reversals=reversals,invalid_snapshot_count=len(invalid))
    timeline.sort(key=lambda row:row['available'])
    times=[row['available'] for row in timeline]
    snapshots=[];counts=[];variances=[];ages=[];means=[];removed_means=[]
    known_to_unknown=unknown_to_known=geometry_changes=0
    previous=None;safe_cache={}
    for snapshot in timeline:
        if snapshot['available']>end:
            break
        if snapshot.get('invalid'):
            previous=None;continue
        geometry=snapshot['geometry'];cells=snapshot['cells'];indices=set(cells)
        if geometry not in safe_cache:
            safe_cache[geometry]=safe_cells(geometry,metadata)
        safe=safe_cache[geometry]
        if start<=snapshot['available']<=end:
            if previous is not None:
                if previous['geometry']==geometry:
                    removed=set(previous['cells'])-indices
                    known_to_unknown+=len(removed)
                    unknown_to_known+=len(indices-set(previous['cells']))
                    removed_means.extend(previous['cells'][index]['mean'] for index in removed)
                else:
                    geometry_changes+=1
            total=geometry[0]*geometry[1]
            snapshots.append(dict(t=snapshot['t'],available_t=snapshot['available'],mode=snapshot['mode'],
                total_cells=total,known_cells=len(cells),unknown_cells=total-len(cells),
                known_fraction=len(cells)/total,
                footprint_safe_cell_centres=len(safe) if safe is not None else None,
                known_footprint_safe_cells=len(indices & safe) if safe is not None else None,
                known_fraction_of_footprint_safe_cells=len(indices & safe)/len(safe) if safe else None))
            for cell in cells.values():
                counts.append(cell['count']);variances.append(cell['variance']);means.append(cell['mean'])
                ages.append(max(0.,snapshot['t']-cell['last_observed']))
        previous=snapshot
    receipt=all(row.get('receipt_recorded',False) for row in timeline if not row.get('invalid'))
    return dict(schema_version=1,status='available' if snapshots else 'no_mission_snapshots',
        scope='sparse_statistics_interface_only_not_consumer_acceptance_or_timeout',
        availability_time='max_source_and_recorder_receipt' if receipt and timeline else 'source_time_only_or_mixed',
        unknown_semantics='no observation; excluded from quality/count/variance/age distributions',
        snapshot_count=len(snapshots),invalid_snapshot_count=len(invalid),invalid_snapshot_reasons=invalid,
        snapshot_rows=snapshots,known_fraction=distribution([r['known_fraction'] for r in snapshots]),
        known_fraction_of_footprint_safe_cells=distribution([r['known_fraction_of_footprint_safe_cells'] for r in snapshots]),
        known_cell_snapshot_distributions=dict(sample_count=distribution(counts),variance=distribution(variances),
                                             quality_mean=distribution(means),observation_age_s=distribution(ages)),
        transitions=dict(known_to_unknown=known_to_unknown,unknown_to_known=unknown_to_known,
                         geometry_changes=geometry_changes,removed_known_mean=distribution(removed_means)),
        plans=_plan_coverage(plans,start,end,timeline,times),
        estimated_trajectory=_pose_coverage(estimates,'estimates',start,end,timeline,times,metadata),
        truth_trajectory=_pose_coverage(truth,'truth',start,end,timeline,times,metadata))
