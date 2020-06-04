import logging
from collections import defaultdict

import itertools
from functools import reduce

from mapcore.swm.src.components.semnet import Sign
from mapmulti.grounding.planning_task import MaPlanningTask
from mapcore.planning.grounding.pddl_grounding import signify_predicates, pred_resonate, _update_predicates, \
    _define_situation, _update_exp_signs, _create_type_map, _create_subtype, task_signs
from mapcore.planning.search.mapsearch import mix_pairs


def ground(problem, plagent, exp_signs=None):
    domain = problem.domain
    actions = domain.actions.values()
    predicates = domain.predicates.values()
    constraints = problem.constraints

    # Objects
    objects = problem.objects
    objects.update(domain.constants)
    logging.debug('Objects:\n%s' % objects)

    # Remove old type_map
    if exp_signs:
        objects = _update_exp_signs(exp_signs, objects)

    # Create a map from types to objects
    type_map = _create_type_map(objects)
    logging.debug("Type to object map:\n%s" % type_map)

    # Create type subtype map
    subtype_map = _create_subtype(domain.types)

    obj_signifs = {}
    # add agents, because on significances they have a inner link to I and They signs
    obj_means = {}

    # Check logic in exp
    if exp_signs:
        signs = exp_signs
        I_sign = signs['I']
        They_sign = signs['They']
        obj_means[I_sign] = I_sign.meanings[1]
        obj_signifs[I_sign] = I_sign.significances[1]
        obj_means[They_sign] = They_sign.meanings[1]
        obj_signifs[They_sign] = They_sign.significances[1]
        signs['situation'] = exp_signs['situation']
    else:
        signs = {}
        I_sign = Sign("I")
        obj_means[I_sign] = I_sign.add_meaning()
        obj_signifs[I_sign] = I_sign.add_significance()
        signs[I_sign.name] = I_sign
        They_sign = Sign("They")
        obj_means[They_sign] = They_sign.add_meaning()
        obj_signifs[They_sign] = They_sign.add_significance()
        signs[They_sign.name] = They_sign
        signs['situation'] = Sign('situation')

    for obj in objects:
        obj_sign = Sign(obj)
        obj_signifs[obj_sign] = obj_sign.add_significance()
        signs[obj] = obj_sign
        if obj_sign.name == plagent:
            connector = obj_signifs[obj_sign].add_feature(obj_signifs[I_sign], zero_out=True)
            I_sign.add_out_significance(connector)
            obj_means[obj_sign] = obj_sign.add_meaning()

    for tp, objects in type_map.items():
        if exp_signs:
            tp_sign = signs[tp.name]
        else:
            tp_sign = Sign(tp.name)
        for obj in objects:
            obj_signif = obj_signifs[signs[obj]]
            tp_signif = tp_sign.add_significance()
            connector = tp_signif.add_feature(obj_signif, zero_out=True)
            signs[obj].add_out_significance(connector)
        if not exp_signs:
            signs[tp.name] = tp_sign

    others = set()
    for id in range(1, len(signs['agent'].significances)+1):
        other_ag = signs['agent'].significances[id].get_signs()
        if signs[plagent] not in other_ag:
            others |= other_ag

    for subagent in others:
        if subagent.name != plagent:
            if not They_sign in subagent.significances[1].get_signs():
                signif = obj_signifs[They_sign]
                if signif.is_empty():
                    They_signif = signif
                else:
                    They_signif = They_sign.add_significance()
                connector = subagent.significances[1].add_feature(They_signif, zero_out=True)
                They_sign.add_out_significance(connector)
                obj_means[subagent] = subagent.add_meaning()

    if not exp_signs:
        updated_predicates = _update_predicates(predicates, actions)
        signify_predicates(predicates, updated_predicates, signs, subtype_map, domain.constants)
        signify_actions(actions, constraints, signs, plagent, obj_means, obj_signifs)
        signify_connection(signs)

    start_situation, pms = _define_situation('*start*', problem.initial_state, signs, 'image')
    goal_situation, pms = _define_situation('*finish*', problem.goal, signs, 'image')
    # if problem.name.startswith("blocks"):
    #     list_signs = task_signs(problem)
    #     _expand_situation_ma_blocks(goal_situation, signs, pms, list_signs)  # For task
    return MaPlanningTask(problem.name, signs, start_situation, goal_situation)

def signify_actions(actions, constraints, signs, agent, obj_means, obj_signifs):
    for action in actions:
        act_sign = Sign(action.name)
        act_signif = act_sign.add_significance()

        def update_significance(predicate, signature, effect=False):
            pred_sign = signs[predicate.name]
            if len(pred_sign.significances) > 1:
                pred_cm = pred_resonate('significance', pred_sign, predicate, signs, signature)
            elif len(pred_sign.significances) == 0:
                pred_cm = pred_sign.add_significance()
            else:
                pred_cm = pred_sign.significances[1]
            connector = act_signif.add_feature(pred_cm, effect=effect)
            pred_sign.add_out_significance(connector)
            if len(predicate.signature) == 1:
                fact = predicate.signature[0]
                role_sign = signs[fact[1][0].name + fact[0]]
                conn = act_signif.add_feature(role_sign.significances[1], connector.in_order, effect=effect,
                                              zero_out=True)
                role_sign.add_out_significance(conn)
            elif not len(predicate.signature) == 0:
                if not predicate.signature[0][1][0].name == predicate.signature[1][1][0].name:
                    for role_sign in pred_cm.get_signs():
                        connector_new = act_signif.add_feature(role_sign.significances[1], connector.in_order,
                                                               effect=effect,
                                                               zero_out=True)
                        role_sign.add_out_significance(connector_new)

        for predicate in action.precondition:
            update_significance(predicate,  action.signature, False)
        for predicate in action.effect.addlist:
            update_significance(predicate, action.signature, True)
        signs[action.name] = act_sign

        if constraints:
            if not action.agents:
                nonspecialized(constraints, act_signif, signs, agent, obj_means, obj_signifs)
            else:
                specialized(action, signs, obj_means, obj_signifs, act_signif, agent, constraints)

        else:
            simple(signs, obj_means, act_signif)

def signify_connection(signs):
    Send = Sign("Send")
    send_signif = Send.add_significance()
    Broadcast = Sign("Broadcast")
    brdct_signif = Broadcast.add_significance()
    connector = brdct_signif.add_feature(send_signif)
    Send.add_out_significance(connector)
    Approve = Sign("Approve")
    approve_signif = Approve.add_significance()
    connector = approve_signif.add_feature(send_signif)
    Send.add_out_significance(connector)
    signs[Send.name] = Send
    signs[Broadcast.name] = Broadcast
    signs[Approve.name] = Approve

    They_sign = signs["They"]

    They_signif = They_sign.add_significance()
    brdct_signif = Broadcast.add_significance()
    connector = They_signif.add_feature(brdct_signif)
    Broadcast.add_out_significance(connector)

    approve_signif = Approve.add_significance()
    They_signif = They_sign.add_significance()
    connector = They_signif.add_feature(approve_signif)
    Approve.add_out_significance(connector)

    brdct_signif = Broadcast.add_significance()
    executer = brdct_signif.add_feature(Broadcast.name.lower(), effect=True, actuator=True)
    Send.add_out_significance(executer)

    approve_signif = Approve.add_significance()
    executer = approve_signif.add_feature(Approve.name.lower(), effect=True, actuator=True)
    Send.add_out_significance(executer)

def simple(signs, obj_means, act_signif):
    agents = set()
    I_sign = signs['I']
    agents.add(I_sign)
    agents |= {cm.sign for cm in signs['They'].spread_up_activity_obj('significance', 1)}
    for agent in agents:
        act_mean = act_signif.copy('significance', 'meaning')
        connector = act_mean.add_feature(obj_means[agent])
        efconnector = act_mean.add_feature(obj_means[agent], effect=True)
        agent.add_out_meaning(connector)


def specialized(action, signs, obj_means, obj_signifs, act_signif, agent, constraints):
    agent_signs = []
    agents_with_constraints = []
    for ag in action.agents:
        for cm in signs[ag].significances.values():
            agent_signs.extend(list(cm.get_signs())) #add current agents's signs
    for ag in constraints.keys():
        agents_with_constraints.append(signs[ag])
    agent_roles = {}
    # receiving all agent's roles.
    for ag in agent_signs:
        for a in ag.get_role():
            if a.name != "object":
                agent_roles.setdefault(ag, set()).update(a.get_role())
    for ag in agent_roles.keys():
        act_mean = act_signif.copy('significance', 'meaning')
        role_signs = [sign for sign in act_mean.get_signs() if sign in agent_roles[ag]]
        for role_sign in role_signs: # changing agent's roles to agents cms
            if role_sign in act_mean.get_signs():
                ag_mean = obj_means[ag]
                act_mean.replace('meaning', role_sign, ag_mean)
        if ag in agents_with_constraints:
            predicates = []
            for _, preds in constraints[ag.name].items():
                predicates.extend(preds)
            role_signifs = {}

            non_agent_preds = []
            agent_preds = []
            for pred in predicates:
                for signa in pred.signature:
                    if signa[0] == ag.name:
                        agent_preds.append(pred) # predicates with agent signature
                        break
                    else:
                        continue
                else:
                    non_agent_preds.append(pred) # predicates without agent signature

            agent_signs= []
            for predicate in agent_preds:
                agent_signs.extend([signa[0] for signa in predicate.signature if signa[0] != ag.name])
            for pred in non_agent_preds.copy():
                signatures = []
                for signa in pred.signature:
                    signatures.append(signa[0])
                if not any(signa in agent_signs for signa in signatures):
                    non_agent_preds.remove(pred)
            non_agent_preds_signs = {signs[pred.name] for pred in non_agent_preds}
            for event in itertools.chain(act_mean.cause, act_mean.effect):
                event_signs = event.get_signs()
                if ag in event_signs:
                    pred_signs = [pred for pred in predicates if signs[pred.name] in event_signs]
                    if pred_signs:
                        event_signs.remove(ag)
                        for pred in pred_signs:
                            role_signature = {sign for sign in event_signs if sign != signs[pred.name]}
                            for role in role_signature:
                                pred_roles = [signif.get_signs() for _, signif in role.significances.items()]
                                pred_role = set()
                                while len(pred_roles):
                                    for role1 in pred_roles.copy():
                                        role2 = list(role1)[0]
                                        if not role2 in pred_role:
                                            pred_role.add(role2)
                                            pred_roles.remove(role1)
                                        else:
                                            pred_roles.remove(role1)
                                # matrixe role to predicate's type
                                role_signifs.setdefault(role, set()).update(pred_role)
                            # change type sign to object sign
                            for key, pred_signats in role_signifs.items():
                                for pred_signat in pred_signats.copy():
                                    for signa in pred.signature:
                                        if signa[1][0].name == pred_signat.name:
                                            pred_signats.remove(pred_signat)
                                            pred_signats.add(signs[signa[0]])
                elif event_signs & non_agent_preds_signs:
                    for predicate in non_agent_preds:
                        pred_signats = {signs[signa[0]] for signa in predicate.signature}
                        if role_signifs:
                            used_signs = reduce(lambda x, y: x|y, role_signifs.values())
                        else:
                            used_signs = set()
                        new_signs = pred_signats - used_signs
                        for pred_sign in new_signs:
                            attributes = pred_sign.find_attribute()
                            key_at = None
                            depth = 2
                            while depth > 0:
                                for attribute in attributes:
                                    if not attribute in event_signs:
                                        attribute = attribute.find_attribute()
                                        for at in attribute.copy():
                                            if at in event_signs:
                                                key_at = at
                                                break
                                        else:
                                            depth-=1
                                            break
                                break
                            if key_at:
                                role_signifs.setdefault(key_at, set()).add(pred_sign)


            if 'truck' not in action.name and 'airplane' not in action.name:
                obj_signs = signs['block?x'], signs['block?y']
                for obj_sign in obj_signs:
                    if obj_sign in role_signifs.keys():
                        role_signifs.pop(obj_sign)


            pairs = mix_pairs(role_signifs)
            if pairs:
                for pair in pairs:
                    act_mean_constr = act_mean.copy('meaning', 'meaning')
                    for role_sign, obj in pair.items():
                        if obj not in obj_means:
                            obj_mean = obj_signifs[obj].copy('significance', 'meaning')
                        else:
                            obj_mean = obj_means[obj]
                        act_mean_constr.replace('meaning', role_sign, obj_mean)
                    if ag.name == agent:
                        I_sign = signs["I"]
                        connector = act_mean_constr.add_feature(obj_means[I_sign])
                        efconnector = act_mean_constr.add_feature(obj_means[I_sign], effect=True)
                        I_sign.add_out_meaning(connector)
                    else:
                        ag_mean = obj_means[ag]
                        connector = act_mean_constr.add_feature(ag_mean)
                        efconnector = act_mean_constr.add_feature(ag_mean, effect=True)
                        ag.add_out_meaning(connector)

        else:
            if ag.name == agent:
                I_sign = signs["I"]
                connector = act_mean.add_feature(obj_means[I_sign])
                efconnector = act_mean.add_feature(obj_means[I_sign], effect=True)
                I_sign.add_out_meaning(connector)
            else:
                ag_mean = obj_means[ag]
                connector = act_mean.add_feature(ag_mean)
                efconnector = act_mean.add_feature(ag_mean, effect=True)
                ag.add_out_meaning(connector)

def nonspecialized(constraints, act_signif, signs, agent, obj_means, obj_signifs):
    for ag, constraint in constraints.items():
        predicates = set()
        signatures = []
        for _, values in constraint.items():
            for value in values:
                predicates.add(value.name)
                signatures.append(value.signature)

        variants = []
        csignatures = signatures.copy()
        for signa in signatures:
            csignatures.remove(signa)
            signat = [signat[0] for signat in csignatures if signat[1] == signa[1]]
            if len(signat):
                signat.append(signa[0])
                if not signat in variants:
                    variants.append(signat)
        for variant in variants:
            act_mean = act_signif.copy('significance', 'meaning')
            for event in itertools.chain(act_mean.cause, act_mean.effect):
                ev_signs = [connector.out_sign for connector in event.coincidences]
                if any(signs[pred] in ev_signs for pred in predicates):
                    for var in variant:
                        role_sign = [role_sign for role_sign in list(signs[var[1][0].name].get_role()) if
                                     role_sign in ev_signs]
                        if role_sign:
                            role_sign = role_sign[0]
                            if signs[var[0]] in obj_means:
                                obj_mean = obj_means[signs[var[0]]]
                            else:
                                obj_mean = obj_signifs[signs[var[0]]].copy('significance', 'meaning')
                            act_mean.replace('meaning', role_sign, obj_mean)
                            # event.replace('meaning', role_sign, obj_means[var[0]], [])
            if ag == agent:
                I_sign = signs["I"]
                connector = act_mean.add_feature(obj_means[I_sign])
                efconnector = act_mean.add_feature(obj_means[I_sign], effect=True)
                I_sign.add_out_meaning(connector)
            else:
                ag_mean = obj_means[signs[ag]]
                connector = act_mean.add_feature(ag_mean)
                efconnector = act_mean.add_feature(ag_mean, effect=True)
                signs[ag].add_out_meaning(connector)

def _expand_situation_ma_blocks(goal_situation, signs, pms, list_signs):
    ont_image = signs['ontable'].add_image()
    a_image = pms[signs[list_signs[1]]]
    connector = goal_situation.images[1].add_feature(ont_image)
    conn = goal_situation.images[1].add_feature(a_image, connector.in_order)
    signs['ontable'].add_out_image(conn)
    signs[list_signs[1]].add_out_image(conn)
    cl_image = signs['clear'].add_image()
    d_image = pms[signs[list_signs[0]]]
    connector = goal_situation.images[1].add_feature(cl_image)
    conn = goal_situation.images[1].add_feature(d_image, connector.in_order)
    signs['clear'].add_out_image(conn)
    signs[list_signs[0]].add_out_image(conn)


