import logging

import mapcore.swm.src.components.sign_task as st
import random
from mapcore.swm.src.components.semnet import Sign
from mapcore.planning.search.mapsearch import mix_pairs
from mapcore.planning.search.mapsearch import MapSearch as MScore
from copy import copy
import itertools


MAX_CL_LV = 1
'''
Activity coefficient - the average value of 
the depth of activity spreading in the sign 
world model for the present task
'''
A_C = 4

class MapSearch(MScore):
    def __init__ (self, task, TaskType, backward):
        self.world_model = task.signs
        self.MAX_ITERATION = 6
        self.exp_acts = []
        self.exp_sits = set()
        self.backward = backward
        self.TaskType = TaskType
        self.scenario = None
        self.final_plans = []
        if TaskType == 'mapddl':
            self.MAX_ITERATION = 8
            if self.backward:
                self.check_pm = task.start_situation.images[1]
                self.active_pm = task.goal_situation.images[1]
            else:
                self.check_pm = task.goal_situation.images[1]
                self.active_pm = task.start_situation.images[1]
        elif task.subtasks and TaskType == 'mahddl':
            self.scenario = task.subtasks[0].meanings[1].spread_down_htn_activity_act('meaning', A_C)
            self.MAX_ITERATION = len(self.scenario)
            self.active_pm = task.start_situation.images[1]
            self.check_pm = task.goal_situation
        else:
            logging.debug("Cant find the goal situation in task %s" % task.name)
        self.precedents = set()

    def search_plan(self):
        self.I_sign, self.I_obj, self.agents = self.__get_agents()
        self.precedent_activation()
        plans = self._map_iteration(self.active_pm, iteration=0, current_plan=[])
        return plans

    def applicable_search(self, meanings, active_pm):
        applicable_meanings = set()
        for agent, cm in meanings:
            if (agent, cm) in self.precedents:
                expandable = True
            else:
                expandable = False
            result, checked = self._check_activity(cm, active_pm, self.backward, expandable=expandable)
            if result:
                maxlen = max([len(el) for el in checked.spread_down_activity('meaning', A_C)])
                if maxlen >= 2:
                    applicable_meanings.add((agent, checked))
        return applicable_meanings

    def _map_iteration(self, active_pm, iteration, current_plan, prev_state = []):
        logging.debug('STEP {0}:'.format(iteration))
        logging.debug('\tSituation {0}'.format(active_pm.longstr()))

        if iteration >= self.MAX_ITERATION:
            logging.debug('\tMax iteration count')
            return None

        precedents = self.precedent_search(active_pm)

        act_matrice = None
        if self.scenario:
            act_matrice = self.scenario[iteration]

        active_chains = active_pm.spread_down_activity('image', A_C)
        active_signifs = dict()
        active_pm = active_pm.sign.images[1].copy('image', 'meaning')

        for chain in active_chains:
            pm = chain[-1]
            active_signif = pm.sign.spread_up_activity_act('significance', A_C)
            for signif in active_signif:
                if signif.sign not in active_signifs:
                    active_signifs.setdefault(signif.sign, set()).update({el for el in active_signif if el.sign.name == signif.sign.name})
        if act_matrice:
            active_signifs = {key:value for key, value in active_signifs.items() if key == act_matrice.sign}

        meanings = []
        for pm_signif, ams in active_signifs.items():
            chains = []
            for am in ams:
                chains.extend(am.spread_down_activity('significance', A_C))
            merged_chains = []
            for chain in chains:
                for achain in active_chains:
                    if chain[-1].sign == achain[-1].sign and len(chain) > 2 and chain not in merged_chains:
                        merged_chains.append(chain)
                        break
            scripts = self._generate_meanings(merged_chains, act_matrice)
            logging.debug("Generated {0} scripts for {1} action on step {2}".format(str(len(scripts)), pm_signif.name, str(iteration)))
            meanings.extend(scripts)

        applicable_meanings = self.applicable_search(precedents + meanings, active_pm)

        if self.check_pm:
            candidates = self._meta_check_activity(active_pm, applicable_meanings, [x for x, _, _, _ in current_plan])
        else:
            candidates = [(0, script.sign.name, script, agent) for agent, script in applicable_meanings]

        if not candidates:
            logging.debug('\tNot found applicable scripts ({0})'.format([x for _, x, _, _ in current_plan]))
            return None

        logging.debug('\tFound {0} variants'.format(len(candidates)))
        final_plans = []

        logging.info("Текущая длина плана: {0}. Было найдено возможных действий: {1}".format(len(current_plan), len(candidates)))

        for counter, name, script, ag_mask in candidates:
            logging.debug('\tChoose {0}: {1} -> {2}'.format(counter, name, script))
            plan = copy(current_plan)
            subplan = None
            next_pm = self._time_shift_forward(active_pm, script, backward=self.backward)
            if script.sign.images:
                acts = []
                for act in script.sign.images[1].spread_down_activity('image', 2):
                    if act[1] not in acts:
                        acts.append(act[1])
                self.exp_sits.add(next_pm)
                subplan = self.hierarchical_exp_search(active_pm, next_pm, iteration, prev_state, acts)
                min_length = min([len(pl) for pl in subplan])
                subplan = list(filter(lambda x: len(x) == min_length, subplan))[0]
                if not subplan:
                    print('CANT expand %s' % name)
            if not subplan:
                plan.append((active_pm.sign.images[1], name, script, ag_mask))
            else:
                plan.extend(subplan)
                logging.info(
                    'Сложное действие {0} уточнено до подплана: {1}'.format(script.sign.name, [part[1] for part in subplan]))
                prev_state.append(active_pm.sign.images[1])
            _isbuild = False
            if self.check_pm:
                if next_pm.includes('image', self.check_pm):
                    _isbuild = True
            elif self.check_pm is None and iteration == self.MAX_ITERATION - 1:
                _isbuild = True
            if _isbuild:
                final_plans.append((plan, next_pm.sign))
                plan_actions = [(ag_mask.name, act.sign.name) for _, _, act, ag_mask in plan]
                logging.info("Цель достигнута. Длина найденного плана: {0}".format(len(plan)))
                logging.info(plan_actions)
            else:
                recursive_plans = self._map_iteration(next_pm, iteration + 1, plan, prev_state)
                if recursive_plans:
                    final_plans.extend(recursive_plans)
        return final_plans


    def hierarch_acts(self):
        """
        This function implements experience actions search in agent's world model
        :return:
        """
        exp_acts = {}
        for name, sign in self.world_model.items():
            if sign.meanings and sign.images:
                for index, cm in sign.meanings.items():
                    if cm.is_causal():
                        exp_acts.setdefault(sign, {})[index] = cm

        applicable_meanings = {}
        used = {key: {} for key in exp_acts.keys()}
        for agent in self.agents:
            for conn in agent.out_meanings:
                if conn.in_sign in exp_acts and not conn.in_index in used[conn.in_sign]:
                    if conn.in_index in exp_acts[conn.in_sign]:
                        applicable_meanings.setdefault(conn.in_sign, []).append((agent, exp_acts[conn.in_sign][conn.in_index]))
                        used.setdefault(conn.in_sign, {})[conn.in_index] = getattr(conn.in_sign, 'meanings')[conn.in_index]


        for key1, value1 in exp_acts.items():
            for key2, value2 in value1.items():
                if not key2 in used[key1]:
                    applicable_meanings.setdefault(key1, []).append(
                        (None, value2))

        return applicable_meanings

    # def hierarchical_exp_search(self, active_pm, check_pm, iteration, prev_state, acts, cur_plan = []):
    #     """
    #     create a subplan using images info
    #     :param script: parametrs to generate plan
    #     :return:plan
    #     """
    #     if iteration > len(acts):
    #         return None
    #     if not cur_plan:
    #         logging.info('Clarify experience plan')
    #     try:
    #         if self.backward:
    #             act = acts[-(iteration+1)].sign
    #         else:
    #             act = acts[iteration].sign
    #     except IndexError:
    #         return None
    #     finall_plans = []
    #
    #     try:
    #         active_pm.spread_down_activity('meaning', 5)
    #     except Exception:
    #         active_pm = active_pm.sign.images[1].copy('image', 'meaning')
    #
    #
    #     applicable = self.applicable_search([action for action in self.exp_acts[act] if
    #                                          (action[0] is not None and len(action[1].cause))], active_pm)
    #
    #     if not applicable:
    #         logging.debug('No applicable actions was found')
    #         return None
    #
    #     for action in applicable:
    #         plan = copy(cur_plan)
    #         next_pm = self._time_shift_forward(active_pm, action[1], self.backward)
    #         included_sit = [sit for sit in self.exp_sits if sit.includes('image', next_pm)]
    #         if included_sit:
    #             plan.append(
    #                 (active_pm.sign.images[1], action[1].sign.name, action[1], action[0]))
    #             logging.info('Прецедент уточнен. Добавлено поддействие: %s' % action[1].sign.name)
    #         else:
    #             continue
    #         if next_pm.includes('image', check_pm):
    #                 if plan:
    #                     finall_plans.append(plan)
    #                 else:
    #                     finall_plans.append(plan)
    #                     break
    #         else:
    #             plan = self.hierarchical_exp_search(next_pm, check_pm, iteration+1, prev_state, acts, plan)
    #             if plan:
    #                 finall_plans.extend(plan)
    #     return finall_plans

    def __get_agents(self):
        agent_back = set()
        I_sign = self.world_model['I']
        agent_back.add(I_sign)
        I_objects = [con.in_sign for con in I_sign.out_significances if con.out_sign.name == "I"]
        if I_objects:
            I_obj = I_objects[0]
        else:
            I_obj = None
        They_sign = self.world_model["They"]
        agents = They_sign.spread_up_activity_obj("significance", 1)
        for cm in agents:
            agent_back.add(cm.sign)
        return I_sign, I_obj, agent_back

    def _generate_meanings(self, chains, sm=None):
        def __get_role_index(chain):
            index = 0
            rev_chain = reversed(chain)
            for el in rev_chain:
                if len(el.cause) == 0:
                    continue
                elif len(el.cause) == 1:
                    if len(el.cause[0].coincidences) ==1:
                        index = chain.index(el)
                    else:
                        return index
                else:
                    return index
            return None

        def __generator(combinations, pms, pm_signs, pm, agent):
            for combination in combinations:
                cm = pm.copy('meaning', 'meaning')
                for role_sign, obj_pm in combination.items():
                    if role_sign in pm_signs:
                        if obj_pm.sign in pm_signs:
                            continue
                        obj_cm = obj_pm.copy('significance', 'meaning')
                        cm.replace('meaning', role_sign, obj_cm)
                if not pms:
                    pms.append((agent, cm))
                else:
                    for _, pmd in copy(pms):
                        if pmd.resonate('meaning', cm):
                            break
                    else:
                        pms.append((agent, cm))
            return pms

        replace_map = {}
        main_pm = None
        for chain in chains:
            role_index = __get_role_index(chain)
            if role_index:
                if not chain[role_index].sign in replace_map:
                    replace_map[chain[role_index].sign] = [chain[-1]]
                else:
                    if not chain[-1] in replace_map[chain[role_index].sign]:
                        replace_map[chain[role_index].sign].append(chain[-1])
                main_pm = chain[0]

        connectors = [agent.out_meanings for agent in self.agents]

        main_pm_len = len(main_pm.cause) + len(main_pm.effect) + 2

        mapped_actions = {}
        for agent_con in connectors:
            for con in agent_con:
                if con.in_sign == main_pm.sign:
                    mapped_actions.setdefault(con.out_sign, set()).add(con.in_sign.meanings[con.in_index])

        new_map = {}
        rkeys = {el for el in replace_map.keys()}
        pms = []

        for agent, lpm in mapped_actions.items():
            for pm in lpm.copy():
                if len(pm.cause) + len(pm.effect) != main_pm_len:
                    lpm.remove(pm)
                    continue
                pm_signs = set()
                pm_mean = pm.spread_down_activity('meaning', 3)
                for pm_list in pm_mean:
                    pm_signs |= set([c.sign for c in pm_list])
                role_signs = rkeys & pm_signs
                if not role_signs:
                    lpm.remove(pm)
                    if not pms:
                        pms.append((agent, pm))
                    else:
                        for _, pmd in copy(pms):
                            if pmd.resonate('meaning', pm):
                                break
                        else:
                            pms.append((agent, pm))
        if not sm:
            for agent, lpm in mapped_actions.items():
                for pm in lpm:
                    if len(pm.cause) + len(pm.effect) != main_pm_len:
                        continue
                    pm_signs = set()
                    pm_mean = pm.spread_down_activity('meaning', 3)
                    for pm_list in pm_mean:
                        pm_signs |= set([c.sign for c in pm_list])
                    role_signs = rkeys & pm_signs
                    for role_sign in role_signs:
                        new_map[role_sign] = replace_map[role_sign]
                    combinations = mix_pairs(new_map)
                    pms = __generator(combinations, pms, pm_signs, pm, agent)
        else:
            sm_chains = sm.spread_down_activity('meaning', A_C)
            sm_signs = set()
            for chain in sm_chains:
                sm_signs |= set([c.sign for c in chain])
            changed_signs = set()
            for role, rms in replace_map.items():
                changed_signs |= {cm.sign for cm in rms if cm.sign in sm_signs}
            for agent, actions in mapped_actions.items():
                for action in actions:
                    act_chains = action.spread_down_activity('meaning', A_C)
                    act_signs = set()
                    for chain in act_chains:
                        act_signs |= set([c.sign for c in chain])
                    combinations = mix_pairs(replace_map)
                    pms = __generator(combinations, pms, act_signs, action, agent)
            for pair in pms.copy():
                pm_chains = pair[1].spread_down_activity('meaning', A_C)
                pm_signs = set()
                for chain in pm_chains:
                    pm_signs |= set([c.sign for c in chain])
                if not pm_signs.issuperset(changed_signs):
                    pms.remove(pair)
        return pms

    # def _check_activity(self, pm, next_cm, backward = False, prec_search = False, expandable = True):
    #     if len(pm.cause) and len(pm.effect):
    #         result = True
    #     else:
    #         result = False
    #
    #     bigger = next_cm.cause
    #     smaller = self._applicable_events(pm, backward)
    #     if prec_search: bigger, smaller = smaller, bigger
    #
    #     for event in smaller:
    #         for fevent in bigger:
    #             if event.resonate('meaning', fevent, True):
    #                 break
    #         else:
    #             result = False
    #             break
    #
    #     if expandable:
    #         if not result:
    #             expanded = pm.expand('meaning')
    #             if not len(expanded.effect) == 0:
    #                 result = self._check_activity(expanded, next_cm, backward, prec_search)
    #                 return result
    #             else:
    #                 expanded.sign.remove_meaning(expanded)
    #                 return False, pm
    #     return result, pm

    # def _meta_check_activity(self, active_pm, scripts, prev_pms):
    #     heuristic = []
    #     for agent, script in scripts:
    #         estimation = self._time_shift_forward(active_pm, script, self.backward)
    #         for prev in prev_pms:
    #             if estimation.resonate('image', prev, False, False):
    #                 break
    #         else:
    #             counter = 0
    #             for event in self._applicable_events(estimation):
    #                 for ce in self._applicable_events(self.check_pm):
    #                     if event.resonate('image', ce):
    #                         counter += 1
    #                         break
    #             heuristic.append((counter, script.sign.name, script, agent))
    #     if heuristic:
    #         best_heuristics = max(heuristic, key=lambda x: x[0])
    #         return list(filter(lambda x: x[0] == best_heuristics[0], heuristic))
    #     else:
    #         return None

    def _applicable_events(self, pm, effect = False):
        """
        Search only in events without connecting to agents
        :param pm:
        :param effect:
        :return:
        """
        applicable = []
        if effect:
            search_in_part = pm.effect
        else:
            search_in_part = pm.cause
        for event in search_in_part:
            if len(event.coincidences) == 1:
                flag = False
                for connector in event.coincidences:
                    if connector.out_sign in self.agents:
                        flag = True
                if flag:
                    continue
            applicable.append(event)
        return applicable

    def recursive_files(self, direct, ext):
        """
        Recursive domain search
        :param direct:
        :param ext:
        :return:
        """
        import os
        extfiles = []
        for root, subfolder, files in os.walk(direct):
            for file in files:
                if file.endswith(ext):
                    extfiles.append(os.path.join(root, file))
            for sub in subfolder:
                extfiles.extend(self.recursive_files(os.path.join(root, sub), ext))
            return extfiles

    # def _time_shift_forward(self, active_pm, script, backward = False):
    #     """
    #     Next situation synthesis
    #     :param active_pm: meaning of active situation
    #     :param script: meaning of active action
    #     :param backward: planning style
    #     :return:
    #     """
    #     next_situation = Sign(st.SIT_PREFIX + str(st.SIT_COUNTER))
    #     self.world_model[next_situation.name] = next_situation
    #     pm = next_situation.add_meaning()
    #     st.SIT_COUNTER += 1
    #     copied = {}
    #     for event in active_pm.cause:
    #         for es in self._applicable_events(script, effect=backward):
    #             if event.resonate('meaning', es):
    #                 break
    #         else:
    #             pm.add_event(event.copy(pm, 'meaning', 'meaning', copied))
    #     for event in self._applicable_events(script, effect=not backward):
    #         pm.add_event(event.copy(pm, 'meaning', 'meaning', copied))
    #     pm = pm.copy('meaning', 'image')
    #     global_situation = self.world_model['situation']
    #     global_cm = global_situation.add_image()
    #     connector = global_cm.add_feature(pm)
    #     next_situation.add_out_image(connector)
    #     return pm
















