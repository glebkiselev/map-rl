"""
This file contain functions to parse branches and  leaves

"""

import re
from mapcore.planning.parsers.branch_parser import *

def parse_constraints(constraints, init, objects):
    """
    Constraints contain predicates that only become true for a specific agent or group.
    :return: return the dictionary with agents and suitable predicates
    """
    suitable = {}
    # link keywords
    # search suitable predicates in init and create new
    # PredicatesStmt(name, signatures) name - ['name'] signa - ['signa (ag1)']
    w_ch = 'implies'
    agents = [el[0] for el in objects if el[1] == 'agent']
    changer = []
    for st, end, m in tree_sample(constraints):
        part = constraints[st:end]
        if part.startswith(w_ch):
            changer.append(part)

    for chan in changer:
        agent_preds = []
        for st, end, _ in tree_sample(chan):
            predicate_descr = chan[st:end].strip()
            name = re.findall('^\w+', predicate_descr)[0]
            signatures = [sign for sign in re.findall('\w+', predicate_descr) if sign != name]
            agent_preds.append((name, *signatures))
        to_change = []
        for pr1 in agent_preds:
            for pr2 in init:
                if pr1[0] == pr2.name:
                    diff = pred_difference(pr1[1:], pr2.signature, agents)
                    if diff:
                        to_change.extend(diff)
        agent = None
        for pred in agent_preds:
            for el in pred:
                if el in agents:
                    agent = el
                    break
            if agent:
                break
        for pred in agent_preds:
            for ch in to_change:
                if ch[0] in pred:
                    signature = []
                    for el in pred[1:]:
                        if el != ch[0]:
                            signature.append(el)
                        else:
                            signature.append(ch[1])
                    suitable.setdefault(agent, []).append(PredicatesStmt(pred, signature))

    return suitable

def pred_difference(signa1, signa2, agents):
    to_change = []
    flag = False
    for el in signa1:
        if el in agents:
            if el in signa2:
                flag = True
            else:
                flag = False
                break
            continue
        if el in signa2:
            flag = True
            continue
        else:
            el2 = signa2[signa1.index(el)]
            to_change.append((el, el2))
    if flag:
        return to_change
    else:
        return False
