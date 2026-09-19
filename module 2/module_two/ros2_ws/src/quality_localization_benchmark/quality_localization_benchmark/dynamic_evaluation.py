"""Pure offline dynamic-event evaluation; truth is never used for control.

Working definitions for this protocol, not general safety standards: stopped
means physical linear speed <.02 m/s AND angular speed <.05 rad/s for >=1 s.
Recovery motion exceeds either threshold for >=.25 s. Kinematics use adjacent
truth poses separated by <=.1 s, with no interpolation across longer outages.
The final five source seconds of closure are also summarized independently.

Gazebo service responses do not identify exact physical mutation times. The
possible-obstacle envelope is spawn request through confirmed delete response;
confirmed presence is spawn success through delete request. Contact intervals
use a conservative circular footprint and linear motion between valid truth
samples. Observation gaps can fragment a single physical contact episode.

This API receives no sensor frames or bridge counter history. For sensor_recovery
it reports schedule evidence only, explicitly leaving actual perturbation and
recovery unverified until the separate sensor evidence audit is attached.
"""
import math


LINEAR_STOP_MPS=.02
ANGULAR_STOP_RADPS=.05
STOP_CONFIRM_SEC=1.0
RECOVERY_CONFIRM_SEC=.25
MAX_KINEMATIC_GAP_SEC=.1
MAX_COMMAND_HOLD_SEC=.5
LATE_CLOSURE_WINDOW_SEC=5.0
EPS=1e-9
MAX_EVENT_LATENESS_SEC=.5


def finite(value):
    if isinstance(value,bool):
        return None
    try:
        value=float(value)
    except (ValueError,TypeError,OverflowError):
        return None
    return value if math.isfinite(value) else None


def merge_intervals(intervals):
    result=[]
    for left,right in sorted(intervals):
        if right<left:
            continue
        if result and left<=result[-1][1]+EPS:
            result[-1][1]=max(result[-1][1],right)
        else:
            result.append([left,right])
    return result


def clip_interval(interval,start,end):
    if interval is None:
        return None
    left,right=max(start,interval[0]),min(end,interval[1])
    return [left,right] if right>=left else None


def intersections(intervals,windows):
    return merge_intervals([overlap for interval in intervals for window in windows
                            if (overlap:=clip_interval(interval,*window)) is not None])


def duration(intervals):
    return sum(right-left for left,right in merge_intervals(intervals))


def _truth(rows,start,end):
    by_time={};invalid=duplicates=reversals=0;previous=None
    for row in rows or []:
        if not isinstance(row,dict):
            invalid+=1;continue
        timestamp=finite(row.get('t'))
        if timestamp is None:
            invalid+=1;continue
        if timestamp<start-MAX_KINEMATIC_GAP_SEC or timestamp>end+MAX_KINEMATIC_GAP_SEC:
            continue
        if previous is not None and timestamp<previous-EPS:
            reversals+=1
        previous=timestamp
        x,y,yaw=(finite(row.get(key)) for key in ('x','y','yaw'))
        if None in (x,y,yaw) or row.get('frame','world')!='world':
            invalid+=1;continue
        if timestamp in by_time:
            duplicates+=1
        by_time[timestamp]=dict(t=timestamp,x=x,y=y,yaw=yaw)
    ordered=[by_time[t] for t in sorted(by_time)] if not reversals else []
    segments=[]
    for first,last in zip(ordered,ordered[1:]):
        dt=last['t']-first['t']
        if not EPS<dt<=MAX_KINEMATIC_GAP_SEC+EPS:
            continue
        speed=math.hypot(last['x']-first['x'],last['y']-first['y'])/dt
        yaw=last['yaw']-first['yaw']
        angular=abs(math.atan2(math.sin(yaw),math.cos(yaw)))/dt
        if math.isfinite(speed) and math.isfinite(angular):
            segments.append(dict(start=first['t'],end=last['t'],first=first,last=last,
                                 speed=speed,angular=angular))
    return ordered,segments,dict(valid_pose_rows=len(ordered),invalid_rows=invalid,
                                 duplicate_timestamps=duplicates,time_reversals=reversals)


def _point_on(segment,timestamp):
    fraction=(timestamp-segment['start'])/(segment['end']-segment['start'])
    first,last=segment['first'],segment['last']
    return (first['x']+fraction*(last['x']-first['x']),
            first['y']+fraction*(last['y']-first['y']))


def point_contacts(x,y,rectangles,radius):
    return any(math.hypot(max(left-x,x-right,0),max(bottom-y,y-top,0))<=radius
               for left,right,bottom,top in rectangles)


def _point_segment_distance_squared(point,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    denominator=dx*dx+dy*dy
    fraction=max(0.,min(1.,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/denominator)) if denominator else 0.
    return (point[0]-a[0]-fraction*dx)**2+(point[1]-a[1]-fraction*dy)**2


def swept_contacts(a,b,rectangles,radius):
    """Exact segment-to-rectangle distance for a swept circular footprint."""
    for left,right,bottom,top in rectangles:
        if (max(a[0],b[0])<left-radius or min(a[0],b[0])>right+radius or
                max(a[1],b[1])<bottom-radius or min(a[1],b[1])>top+radius):
            continue
        if point_contacts(*a,[(left,right,bottom,top)],radius) or point_contacts(*b,[(left,right,bottom,top)],radius):
            return True
        lower,upper=0.,1.
        for first,last,lo,hi in ((a[0],b[0],left,right),(a[1],b[1],bottom,top)):
            delta=last-first
            if abs(delta)<1e-15:
                if not lo<=first<=hi:
                    lower,upper=1.,0.;break
            else:
                t0,t1=(lo-first)/delta,(hi-first)/delta
                lower=max(lower,min(t0,t1));upper=min(upper,max(t0,t1))
        if lower<=upper:
            return True
        if min(_point_segment_distance_squared(point,a,b) for point in
               ((left,bottom),(left,top),(right,bottom),(right,top)))<=radius*radius:
            return True
    return False


def _contacts(poses,segments,rectangles,window):
    if window is None or not rectangles:
        return []
    radius,rectangles=rectangles
    hits=[]
    for row in poses:
        if window[0]<=row['t']<=window[1] and point_contacts(row['x'],row['y'],rectangles,radius):
            hits.append([row['t'],row['t']])
    for segment in segments:
        interval=clip_interval([segment['start'],segment['end']],*window)
        if interval is not None and swept_contacts(_point_on(segment,interval[0]),
                                                  _point_on(segment,interval[1]),rectangles,radius):
            hits.append(interval)
    return merge_intervals(hits)


def _motion(segments,window):
    if window is None or window[1]<=window[0]:
        return dict(status='unavailable',interval=window,coverage=None,
                    stopped_intervals=[],moving_intervals=[],first_sustained_stop=None)
    covered=[];stopped=[];moving=[];speeds=[];angular=[]
    for segment in segments:
        interval=clip_interval([segment['start'],segment['end']],*window)
        if interval is None or interval[1]-interval[0]<=EPS:
            continue
        covered.append(interval);speeds.append(segment['speed']);angular.append(segment['angular'])
        if segment['speed']+EPS<LINEAR_STOP_MPS and segment['angular']+EPS<ANGULAR_STOP_RADPS:
            stopped.append(interval)
        else:
            moving.append(interval)
    stopped=merge_intervals(stopped);moving=merge_intervals(moving)
    sustained=next((interval for interval in stopped if interval[1]-interval[0]>=STOP_CONFIRM_SEC-EPS),None)
    covered_seconds=duration(covered)
    return dict(status='available' if covered else 'unavailable',interval=window,
        covered_seconds=covered_seconds,missing_seconds=max(0.,window[1]-window[0]-covered_seconds),
        coverage=covered_seconds/(window[1]-window[0]),
        maximum_linear_speed_mps=max(speeds) if speeds else None,
        maximum_angular_speed_radps=max(angular) if angular else None,
        moving_seconds=duration(moving),stopped_seconds=duration(stopped),
        moving_fraction_of_observed_time=duration(moving)/covered_seconds if covered_seconds else None,
        stopped_intervals=stopped,moving_intervals=moving,first_sustained_stop=sustained,
        stop_onset_delay_sec=sustained[0]-window[0] if sustained else None,
        stop_confirmed_delay_sec=sustained[0]+STOP_CONFIRM_SEC-window[0] if sustained else None)


def _commands(rows,window):
    if window is None or window[1]<=window[0]:
        return dict(status='unavailable',nonzero_messages=None)
    by_time={};invalid=0;previous=None;reversals=0
    for row in rows or []:
        if not isinstance(row,dict):
            invalid+=1;continue
        t,vx,wz=(finite(row.get(key)) for key in ('t','vx','wz'))
        if None in (t,vx,wz):
            invalid+=1;continue
        if not window[0]-MAX_COMMAND_HOLD_SEC<=t<=window[1]:
            continue
        if previous is not None and t<previous-EPS:
            reversals+=1
        previous=t
        by_time[t]=(vx,wz)
    times=sorted(by_time) if not reversals else []
    covered=[];moving=[];message_count=nonzero_count=0;peak_linear=[];peak_angular=[]
    for i,t in enumerate(times):
        vx,wz=by_time[t]
        nonzero=abs(vx)+EPS>=LINEAR_STOP_MPS or abs(wz)+EPS>=ANGULAR_STOP_RADPS
        if window[0]<=t<=window[1]:
            message_count+=1;nonzero_count+=int(nonzero)
            peak_linear.append(abs(vx));peak_angular.append(abs(wz))
        end=min(t+MAX_COMMAND_HOLD_SEC,times[i+1] if i+1<len(times) else window[1])
        interval=clip_interval([t,end],*window)
        if interval is not None:
            covered.append(interval)
            if nonzero:
                moving.append(interval)
    seconds=duration(covered)
    return dict(status='clock_discontinuity' if reversals else 'available' if times else 'unavailable',
                messages=message_count,nonzero_messages=nonzero_count if times else None,
                nonzero_seconds=duration(moving) if times else None,
                covered_seconds=seconds,coverage=seconds/(window[1]-window[0]),
                maximum_abs_linear_command_mps=max(peak_linear) if peak_linear else None,
                maximum_abs_angular_command_radps=max(peak_angular) if peak_angular else None,
                invalid_rows=invalid,time_reversals=reversals,maximum_hold_sec=MAX_COMMAND_HOLD_SEC,
                interpretation='command evidence only; nonzero command does not establish physical motion')


def _event_windows(state,start,end):
    errors=list(state.get('invalid_reasons',[]))
    if state.get('valid_evidence') is False and not errors:
        errors.append('event_node_marked_invalid')
    scenario=state.get('scenario')
    if scenario not in ('partial_block','full_block','sensor_recovery'):
        errors.append('unrecognized_dynamic_scenario')
    epoch=finite(state.get('epoch'))
    if epoch is None or epoch<0:
        errors.append('event_epoch_missing')
    events=state.get('events',[])
    if not isinstance(events,list) or len(events)!=2 or not all(isinstance(e,dict) for e in events):
        return dict(errors=errors+['two_event_records_required'],possible=None,confirmed=None,
                    uncertain=[],release=None,activation=None,events=[])
    first,last=events
    request=finite(first.get('requested_ros'));activation=finite(first.get('succeeded_ros')) if first.get('success') is True else None
    delete_request=finite(last.get('requested_ros'));release=finite(last.get('succeeded_ros')) if last.get('success') is True else None
    # An algorithm may legitimately end the mission before a later event is
    # due. Preserve that failure without relabeling it as bad instrumentation.
    if epoch is not None:
        if end>=epoch+8.+MAX_EVENT_LATENESS_SEC and (request is None or activation is None):
            errors.append('due_activation_incomplete_or_unconfirmed')
        if end>=epoch+28.+MAX_EVENT_LATENESS_SEC and (delete_request is None or release is None):
            errors.append('due_release_incomplete_or_unconfirmed')
        for actual,offset,label in ((request,8.,'activation_request'),(activation,8.,'activation_response'),
                                    (delete_request,28.,'release_request'),(release,28.,'release_response')):
            if actual is None:
                continue
            if actual<epoch+offset-EPS:
                errors.append(label+'_before_registered_offset')
            if actual>epoch+offset+MAX_EVENT_LATENESS_SEC+EPS:
                errors.append(label+'_after_registered_tolerance')
    for a,b,label in ((request,activation,'spawn_reply_before_request'),
                       (activation,delete_request,'release_request_before_activation_confirmed'),
                       (delete_request,release,'delete_reply_before_request')):
        if a is not None and b is not None and b<a-EPS:
            errors.append(label)
    expected=('sensor_fault_begin','sensor_fault_end') if scenario=='sensor_recovery' else ('spawn_obstacle','delete_obstacle')
    if tuple(e.get('kind') for e in events)!=expected:
        errors.append('event_kind_mismatch')
    possible=clip_interval([request,release if release is not None else end],start,end) if request is not None else None
    confirmed=clip_interval([activation,delete_request if delete_request is not None else end],start,end) if activation is not None else None
    uncertain=[]
    if possible is not None:
        if activation is None:
            uncertain.append(possible)
        else:
            interval=clip_interval([request,activation],*possible)
            if interval is not None and interval[1]>interval[0]:
                uncertain.append(interval)
            if delete_request is not None:
                interval=clip_interval([delete_request,release if release is not None else end],*possible)
                if interval is not None and interval[1]>interval[0]:
                    uncertain.append(interval)
    return dict(errors=errors,possible=possible,confirmed=confirmed,uncertain=merge_intervals(uncertain),
                release=release,activation=activation,request=request,events=events)


def summarize_dynamic_episode(event_state,truth_rows,command_rows,mission_start,mission_end,metadata):
    """Summarize event timing, conservative contacts, physical stops and recovery.

    Static and dynamic contact intervals use the same truth samples and can be
    unioned without double-counting a simultaneous static/dynamic intersection.
    The function makes no navigation-success or universal safety determination.
    """
    start,end=finite(mission_start),finite(mission_end)
    if start is None or end is None or not 0<=start<end:
        return dict(schema_version=1,status='invalid_mission_interval')
    state=event_state if isinstance(event_state,dict) else {}
    windows=_event_windows(state,start,end)
    poses,segments,truth_stats=_truth(truth_rows,start,end)
    radius=finite(metadata.get('robot_collision_radius'))
    if radius is None or radius<=0:
        return dict(schema_version=1,status='invalid_robot_collision_radius')
    static=metadata.get('rectangles',[]);dynamic=state.get('rectangles',[])
    for name,rectangles in (('static',static),('dynamic',dynamic)):
        if (not isinstance(rectangles,list) or any(not isinstance(r,(list,tuple)) or len(r)!=4
                or any(finite(v) is None for v in r) or r[0]>=r[1] or r[2]>=r[3] for r in rectangles)):
            return dict(schema_version=1,status='invalid_'+name+'_rectangles')
    sensor=state.get('scenario')=='sensor_recovery'
    if (sensor and dynamic) or (state.get('scenario') in ('partial_block','full_block') and not dynamic):
        return dict(schema_version=1,status='dynamic_geometry_mismatch')
    dynamic_window=None if sensor else windows['possible']
    static_contacts=_contacts(poses,segments,(radius,static),[start,end])
    dynamic_contacts=_contacts(poses,segments,(radius,dynamic),dynamic_window)
    combined=merge_intervals(static_contacts+dynamic_contacts)
    closure=_motion(segments,windows['possible'])
    confirmed=_motion(segments,windows['confirmed'])
    late_window=None
    if windows['confirmed'] is not None:
        a,b=windows['confirmed'];late_window=[max(a,b-LATE_CLOSURE_WINDOW_SEC),b]
    late=_motion(segments,late_window)
    release=windows['release']
    recovery_window=clip_interval([release,end],start,end) if release is not None else None
    recovery=_motion(segments,recovery_window)
    first_motion=next((interval for interval in recovery['moving_intervals']
                       if interval[1]-interval[0]>=RECOVERY_CONFIRM_SEC-EPS),None)
    before_release=_motion(segments,clip_interval([release-STOP_CONFIRM_SEC,release],start,end)) if release is not None else None
    before_activation=_motion(segments,clip_interval([windows['request']-STOP_CONFIRM_SEC,windows['request']],start,end)) if windows.get('request') is not None else None
    overall=_motion(segments,[start,end])
    return dict(schema_version=1,status='available' if poses else 'truth_unavailable',
        scope='offline_evaluation_only_no_truth_feedback_or_control',scenario=state.get('scenario'),
        mission_interval=[start,end],event_timing_valid=not windows['errors'],event_timing_errors=sorted(set(windows['errors'])),
        sequence_completed=bool(windows['activation'] is not None and windows['release'] is not None),
        motion_window_semantics='sensor schedule notifications, actual effect unverified' if sensor else 'possible and confirmed physical barrier presence',
        thresholds=dict(linear_stop_mps=LINEAR_STOP_MPS,angular_stop_radps=ANGULAR_STOP_RADPS,
                        stop_confirmation_sec=STOP_CONFIRM_SEC,recovery_confirmation_sec=RECOVERY_CONFIRM_SEC,
                        maximum_truth_gap_sec=MAX_KINEMATIC_GAP_SEC,late_closure_window_sec=LATE_CLOSURE_WINDOW_SEC,
                        maximum_event_lateness_sec=MAX_EVENT_LATENESS_SEC,numerical_speed_tolerance=EPS,
                        interpretation='preregistered experiment working definitions, not universal safety limits'),
        obstacle_timing=dict(possible_presence_interval=dynamic_window,
                             confirmed_presence_interval=None if sensor else windows['confirmed'],
                             uncertain_intervals=[] if sensor else windows['uncertain'],rectangles=dynamic,
                             interpretation='request/response times bracket physical mutation; replies are not exact mutation timestamps'),
        contacts=dict(available=bool(poses),radius_m=radius,
            geometry='circular footprint swept along bounded adjacent truth segments in world coordinates',
            contact_time_semantics='whole intersecting sample interval is conservatively counted; gaps may split physical episodes',
            static_contact_intervals=static_contacts,dynamic_contact_intervals=dynamic_contacts,
            combined_contact_intervals=combined,
            static_contact_count=len(static_contacts) if poses else None,
            dynamic_contact_count=len(dynamic_contacts) if poses else None,
            combined_contact_count=len(combined) if poses else None,
            dynamic_contacts_during_uncertain_physics=intersections(dynamic_contacts,windows['uncertain']),
            dynamic_contacts_during_confirmed_presence=intersections(dynamic_contacts,[windows['confirmed']]) if windows['confirmed'] else []),
        truth_statistics=truth_stats,mission_kinematic_coverage=overall.get('coverage'),
        possible_closure_motion=closure,confirmed_closure_motion=confirmed,late_closure_motion=late,
        closure_commands=_commands(command_rows,windows['possible']),
        late_closure_commands=_commands(command_rows,late_window),
        already_stopped_before_activation=bool(before_activation and before_activation.get('first_sustained_stop')),
        recovery=dict(release_confirmed_ros=release,window=recovery_window,
                      first_sustained_motion_interval=first_motion,
                      motion_onset_delay_sec=first_motion[0]-release if first_motion else None,
                      motion_confirmed_delay_sec=first_motion[0]+RECOVERY_CONFIRM_SEC-release if first_motion else None,
                      stopped_for_one_second_before_release=bool(before_release and before_release.get('first_sustained_stop')),
                      kinematics=recovery),
        sensor_effect_evidence=dict(applicable=sensor,actual_sensor_effect_verified=False,
            status='requires_independent_bridge_counters_and_source_frames' if sensor else 'not_applicable',
            scheduled_window=[finite(state.get('scheduled_activation_ros')),finite(state.get('scheduled_release_ros'))] if sensor else None,
            boundary_records=windows['events'] if sensor else [],
            required_evidence=['faulted scan/image counts during epoch+8..28',
                'paired raw/output source frames show registered perturbation and subsequent recovery',
                'no unplanned passthrough, clock reset, or missing bridge diagnostic coverage'],
            interpretation='event-state success alone never establishes sensor degradation or recovery'))
