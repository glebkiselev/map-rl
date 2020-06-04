import json
import os
from copy import deepcopy

import math


def get_agent(planner_data):
    agents = [name for name in planner_data['global-start'] if 'ag' in name]
    act_ag = None
    start = planner_data['global-start']['objects']
    goal = planner_data['global-finish']['objects']
    for ag in agents:
        if planner_data['global-start'][ag] != planner_data['global-finish'][ag]:
            act_ag = ag
            break
        elif planner_data['global-start']['objects'][ag] != planner_data['global-finish']['objects'][ag]:
            act_ag = ag
            break

    return act_ag, {'start_x': start[act_ag]['x'],
             'start_y': start[act_ag]['y'],
             'goal_x': goal[act_ag]['x'],
             'goal_y': goal[act_ag]['y'],
             'holding_start': planner_data['global-start'][act_ag]['holding']['cause'][1]
             if 'holding' in planner_data['global-start'][act_ag] else None,
             'holding_goal': planner_data['global-finish'][act_ag]['holding']['cause'][1]
             if 'holding' in planner_data['global-finish'][act_ag] else None,
             'r': start[act_ag]['r']}


def check_window_size(ut, window_size):
    if ut['agent']['goal_x'] - ut['agent']['start_x'] > window_size or ut['agent']['goal_y'] - ut['agent']['start_y'] > window_size:
        got_x = False
        got_y = False
        sub_tasks = []
        ag_st = deepcopy([ut['agent']['start_x'], ut['agent']['start_y']])
        ag_g = deepcopy([ut['agent']['goal_x'], ut['agent']['goal_y']])
        hs = ut['agent']['holding_start']
        hg = ut['agent']['holding_goal']
        if hs and hg: # this situation can be only in move action
            move_block = hs
        else:
            move_block = None
        while not got_x and not got_y:
            task = {}
            task['agent'] = {}
            task['map'] = deepcopy(ut['map'])
            task['blocks'] = deepcopy(ut['blocks'])
            if ag_st[0] + window_size < ag_g[0]:
                task['agent']['start_x'] = ag_st[0]
                task['agent']['goal_x'] = ag_st[0] + window_size
                ag_st[0] = task['agent']['goal_x']
            else:
                task['agent']['start_x'] = ag_st[0]
                task['agent']['goal_x'] = ag_g[0]
                got_x = True
            if ag_st[1] + window_size < ag_g[1]:
                task['agent']['start_y'] = ag_st[1]
                task['agent']['goal_y'] = ag_st[1] + window_size
                ag_st[1] = task['agent']['goal_y']
            else:
                task['agent']['start_y'] = ag_st[1]
                task['agent']['goal_y'] = ag_g[1]
                got_y = True


    return united_task


def parse(from_path, to_path, multiple=True, window_size=30):
    file_paths = []
    if multiple:
        if not os.path.exists(to_path):
            os.mkdir(to_path)
        for filename in os.listdir(from_path):
            file_paths.append(from_path + filename)
    else:
        file_paths.append(from_path)
    fp = []
    for i in range(len(file_paths)):
        files = []
        for file in file_paths:
            if f'{i}.json' in file:
                files.append(file)
        fp.append(min(files, key = lambda files:len(files)))
    united_task = None
    united_tasks_indices = None
    count = 0
    for one_file_path in fp:
        with open(one_file_path, 'r') as read:
            planner_data = json.load(read)
        full_rl_data = {'map': {'walls': None}}
        map_size = planner_data['map']['map-size']
        full_rl_data['map']['rows'] = map_size[0]
        full_rl_data['map']['cols'] = map_size[1]
        start = planner_data['global-start']['objects']
        goal = planner_data['global-finish']['objects']
        # On each step only 1 agent has an action -> Let's find it!
        act_ag, agent = get_agent(planner_data)
        blocks = {}
        block_names = list(start.keys()) + ([agent['holding_start']] if agent['holding_start'] is not None else [])
        block_names = [name for name in block_names if 'block-' in name]
        for key in block_names:
            blocks[key] = {}
            if key == agent['holding_start'] and key !=agent['holding_goal']:  # task is to put down a block
                s_item = start[act_ag]
                g_item = goal[key]
                s_item['r'] = g_item['r']
            elif key == agent['holding_goal'] and key !=agent['holding_start']:  # task is to pick up a block
                s_item = start[key]
                g_item = goal[act_ag]
                g_item['r'] = s_item['r']
            elif key == agent['holding_start'] and key ==agent['holding_goal']: # task to move with block
                s_item = start[act_ag]
                g_item = goal[act_ag]
                g_item['r'] = 1 #todo check anywhere
                s_item['r'] = 1
            else:
                s_item = start[key]
                g_item = goal[key]
            blocks[key] = {'start_x': s_item['x'],
                           'start_y': s_item['y'],
                           'goal_x': g_item['x'],
                           'goal_y': g_item['y'],
                           'r': s_item['r']}
        full_rl_data['agent'] = agent
        start_cond = planner_data['global-start']['conditions']
        goal_cond = planner_data['global-finish']['conditions']
        conditions = {'start': {block: [] for block in block_names},
                      'goal': {block: [] for block in block_names}}
        conditions['start'] = rewrite_conditions(start_cond, conditions['start'])
        conditions['goal'] = rewrite_conditions(goal_cond, conditions['goal'])
        full_rl_data['blocks'] = change_order_via_conditions(blocks, conditions)
        if not multiple:
            name = one_file_path.split('/')[-1]
            with open(to_path + 'parsed_' + name, 'w+') as write:
                write.write(json.dumps(full_rl_data, indent=4))
            return
        if united_task is None:
            united_task = full_rl_data
            united_tasks_indices = str(count)
            count += 1
            continue
        if is_in_window(united_task, full_rl_data, window_size): # check if few planner steps in 1 RL window
            united_task = join_tasks(united_task, full_rl_data) # previous full_rl_data
            united_tasks_indices += str(count)
        else:
            name = 'tasks_' + united_tasks_indices + '.json'
            united_task = check_window_size(united_task, window_size)
            with open(to_path + 'parsed_' + name, 'w+') as write:
                write.write(json.dumps(crop_task_map(united_task), indent=4))
            united_task = full_rl_data
            united_tasks_indices = str(count)
        count += 1
    name = 'tasks_' + united_tasks_indices + '.json'
    with open(to_path + 'parsed_' + name, 'w+') as write:
        write.write(json.dumps(crop_task_map(united_task), indent=4))


def is_in_window(old_task, new_task, window):
    if (old_task['agent']['holding_start'] and not old_task['agent']['holding_goal']) or \
            (new_task['agent']['holding_start'] and not new_task['agent']['holding_goal']) or \
        (not old_task['agent']['holding_start'] and  old_task['agent']['holding_goal'])or \
            (not new_task['agent']['holding_start'] and  new_task['agent']['holding_goal']) or\
                (not old_task['agent']['holding_goal'] and new_task['agent']['holding_start']) or\
            (old_task['agent']['holding_goal'] and not new_task['agent']['holding_start']):
        return False
    if old_task['agent']['holding_goal'] is not None:
        if old_task['agent']['holding_start'] == old_task['agent']['holding_goal'] \
            == new_task['agent']['holding_goal'] == new_task['agent']['holding_start']:
            if not ((new_task['agent']['start_x'] == new_task['agent']['goal_x']) and (new_task['agent']['start_y'] == new_task['agent']['goal_y'])):
                return False # all movement with blocks are in separate files iff the movement was.
    old_xs, old_ys = get_changing_points(old_task)
    new_xs, new_ys = get_changing_points(new_task)
    minx, maxx, miny, maxy = bounding_rect_points(old_xs+new_xs, old_ys+new_ys)
    return maxx - minx <= window and maxy - miny <= window


def bounding_rect_points(xs, ys):
    return min(xs), max(xs), min(ys), max(ys)


def join_tasks(old_task, new_task):
    assert set(old_task['blocks'].keys()) == set(new_task['blocks'].keys()), 'block names are not identical'
    assert old_task['agent']['goal_x'] == new_task['agent']['start_x'], 'agent steps in bad order'
    block_names = list(old_task['blocks'].keys())
    for name in block_names:
        try:
            assert old_task['blocks'][name] == new_task['blocks'][name], f'block {name} coords are different in tasks'
        except AssertionError:
            if (new_task['blocks'][name]['start_x'] == new_task['blocks'][name]['goal_x']) and \
                (new_task['blocks'][name]['start_y'] == new_task['blocks'][name]['goal_y']):
                result_task = old_task
                result_task = crop_task_map(result_task)
                return result_task
            else:
                raise AssertionError
    result_task = new_task
    result_task['agent']['start_x'] = old_task['agent']['start_x']
    result_task = crop_task_map(result_task)
    return result_task


def crop_task_map(task):
    task['map']['full_rows'] = task['map']['rows']
    task['map']['full_cols'] = task['map']['cols']
    xs, ys = get_changing_points(task)
    minx, maxx, miny, maxy = bounding_rect_points(xs, ys)
    asx, asy = task['agent']['start_x'], task['agent']['start_y']
    agx, agy = task['agent']['goal_x'], task['agent']['goal_y']
    if (asx, asy) != (agx, agy):
        task['agent']['start_x'] = asx - minx
        task['agent']['start_y'] = asy - miny
        task['agent']['goal_x'] = agx - minx
        task['agent']['goal_y'] = agy - miny
        task['agent']['coord_mode'] = 'cropped'
    else:
        task['agent']['coord_mode'] = 'full'
    for block in list(task['blocks'].values()):
        sx, sy = block['start_x'], block['start_y']
        gx, gy = block['goal_x'], block['goal_y']
        if (sx, sy) != (gx, gy):
            block['start_x'] = sx - minx
            block['start_y'] = sy - miny
            block['goal_x'] = gx - minx
            block['goal_y'] = gy - miny
            block['coord_mode'] = 'cropped'
            task['agent']['start_x'] = asx - minx
            task['agent']['start_y'] = asy - miny
            task['agent']['goal_x'] = agx - minx
            task['agent']['goal_y'] = agy - miny
            task['agent']['coord_mode'] = 'cropped'
        else:
            block['coord_mode'] = 'full'
    task['map']['rows'] = maxx - minx + 1
    task['map']['cols'] = maxy - miny + 1
    task['map']['start_x'] = minx
    task['map']['start_y'] = miny
    return task


def get_changing_points(task):
    xs = []
    ys = []
    agent_in_points = False
    sx, sy = task['agent']['start_x'], task['agent']['start_y']
    gx, gy = task['agent']['goal_x'], task['agent']['goal_y']
    if (sx, sy) != (gx, gy):
        xs.extend([sx, gx])
        ys.extend([sy, gy])
    blocks = list(task['blocks'].values())
    for block in blocks:
        sx, sy = block['start_x'], block['start_y']
        gx, gy = block['goal_x'], block['goal_y']
        if (sx, sy) != (gx, gy):
            xs.extend([sx, gx])
            ys.extend([sy, gy])
            if not agent_in_points:
                agent_in_points = True
                asx, asy = task['agent']['start_x'], task['agent']['start_y']
                agx, agy = task['agent']['goal_x'], task['agent']['goal_y']
                xs.extend([asx, agx])
                ys.extend([asy, agy])
    return xs, ys


def rewrite_conditions(dict_from, dict_to):
    for key, value in dict_from.items():
        if 'onground' in key:
            dict_to[value['cause'][0]].append('onground')
        elif 'clear' in key:
            dict_to[value['cause'][0]].append('clear')
        elif 'on' in key:
            dict_to[value['cause'][0]].append({'on': value['cause'][1]})
    return dict_to


def contains_on(block_cond):
    for cond in block_cond:
        if type(cond) is dict and 'on' in cond:
            return cond['on']
    return None


def not_clear(conditions, block_name):
    for block, value in conditions.items():
        for cond in value:
            if type(cond) is dict and 'on' in cond and cond['on'] == block_name:
                return block
    return None


def change_order_via_conditions(blocks, conditions):
    block_names = list(blocks.keys())
    blocks_queue = []
    for name in block_names:
        if conditions['start'][name] == ['clear', 'onground'] and conditions['goal'][name] == ['clear', 'onground']:
            blocks_queue.append(name)
    for name in block_names:
        if name in blocks_queue:
            continue
        if contains_on(conditions['goal'][name]):
            base_block = contains_on(conditions['goal'][name])
            while contains_on(conditions['goal'][base_block]):
                base_block = contains_on(conditions['goal'][base_block])
            blocks_queue.append(base_block)
            top_block = base_block
            while not_clear(conditions['goal'], top_block):
                top_block = not_clear(conditions['goal'], top_block)
                blocks_queue.append(top_block)
        elif conditions['start'][name] == [] or conditions['goal'][name] == []:
            blocks_queue.append(name)
    return {name: blocks[name] for name in blocks_queue}
