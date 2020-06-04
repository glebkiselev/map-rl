import importlib
import logging
import time
import os
from multiprocessing import Pipe, Process
from multiprocessing import log_to_stderr
import pickle

from mapcore.planning.agent.planning_agent import PlanningAgent
from mapmulti.agent.messagen import reconstructor
from mapmulti.agent.messagen import Tmessage

class MAgent(PlanningAgent):
    def __init__(self):
        super().__init__()

    # Initialization pddl agent
    def multinitialize(self, name, agents, problem, TaskType, backward):
        """
        This function allows agent to be initialized. We do not use basic __init__ to let
        user choose a valid variant of agent.
        :param problem: problem
        :param ref: the dynamic value of plan clarification
        """
        self.name = name
        self.problem = problem
        self.solution = []
        self.allsolutions = []
        self.backward = backward
        self.others = {ag for ag in agents if ag != name}
        self.task = None
        self.TaskType = TaskType

    # Grounding tasks
    def get_task(self):
        """
        This functions is needed to load swm.
        :return: task - sign representation of the problem.
        """
        logging.info('Grounding start: {0}'.format(self.problem.name))
        signs = self.load_swm(type = 'classic')
        if self.TaskType == 'mahddl':
            from mapmulti.grounding import hddl_grounding
            self.task = hddl_grounding.ground(self.problem, self.name, signs)
        elif self.TaskType == 'mapddl':
            from mapmulti.grounding import pddl_grounding
            self.task = pddl_grounding.ground(self.problem, self.name, signs)
        else:
            Exception('Wrong TaskType! (mahddl or mapddl)')
        logging.info('Grounding end: {0}'.format(self.problem.name))
        logging.info('{0} Signs created'.format(len(self.task.signs)))
        if signs:
            return len(signs) - len(self.task.signs)
        else:
            return 0

    def sol_to_acronim(self, solution):
        acronim = ''
        if solution[-1][1] == 'approve' or solution[-1][1] == 'broadcast':
            solution = solution[:-1]
        for act in solution:
            if act[3]:
                if act[3].name == 'I':
                    name = self.name
                else:
                    name = act[3].name
            else:
                name = self.name
            acronim += act[1] + ' ' + name + ';'
        return acronim


    def search_solution(self):
        """
        This function is needed to synthesize all plans, choose the best one and
        save the experience.
        """
        from mapmulti.search.mapsearch import MapSearch as Search
        logging.info('Search start: {0}, Start time: {1} by agent: {2}'.format(self.task.name, time.clock(), self.name))
        if len(self.others) > 1:
            method = 'Broadcast'
            cm = self.task.signs[method].significances[1].copy('significance', 'meaning')
        elif len(self.others) == 1:
            method = 'Approve'
            cm = self.task.signs[method].significances[1].copy('significance', 'meaning')
        else:
            method = 'save_achievement'
            cm = None
        search = Search(self.task, self.TaskType, self.backward)
        self.allsolutions = search.search_plan()
        self.solution = self.sort_plans([sol[0] for sol in self.allsolutions])
        if self.backward:
            self.solution = list(reversed(self.solution))
        self.solution.append((self.task.signs[method].add_meaning(), method, cm, self.task.signs["I"]))
        mes = Tmessage(self.solution, self.name)
        message = getattr(mes, method.lower())()
        return message

    def save_solution(self, solution):
        solacr = ''
        file_name = None
        for sol in solution.split(';')[:-1]:
            solacr+=sol.strip() + ';'
        for solution, goal in self.allsolutions:
            if self.backward:
                solution = list(reversed(solution))
            acronim = self.sol_to_acronim(solution)
            if acronim == solacr:
                if not self.task.goal_situation:
                    self.task.goal_situation = goal
                file_name = self.task.save_signs(solution)
                if file_name:
                    logging.info('Agent ' + self.name + ' finished all works')
                break
        else:
            logging.info("Agent {0} can not find the right solution to save!".format(self.name))
        if not file_name:
            for f in os.listdir(os.getcwd()):
                if f.startswith('wmodel_'):
                    if f.split(".")[0].endswith(self.name) or f.split(".")[0].endswith('agent'):
                        file_name = f
                        break
        import platform
        if platform.system() != 'Windows':
            delim = '/'
        else:
            delim = '\\'
        file_name = os.getcwd() +delim+ file_name
        return file_name



class DecisionStrategies:
    def __init__(self, solutions):
        self.plans = solutions

    def auction(self):
        solutions = {}
        auct = {}
        maxim = 1
        for agent, sol in self.plans.items():
            _, plan = reconstructor(sol)
            clear_plan = ''
            for el in plan.strip().split(';')[:-2]:
                clear_plan+=el + ';'
            solutions[agent] = clear_plan
        for agent, plan in solutions.items():
            if not plan in auct:
                auct[plan] = 1
            else:
                iter = auct[plan]
                auct[plan] = iter+1
                if iter+1 > maxim:
                    maxim = iter+1

        plan = [plan for plan, count in auct.items() if count==maxim][0]

        agents = []
        for agent, pl in solutions.items():
            if pl == plan:
                agents.append(agent)
        return agents, plan



def agent_activation(agpath, agtype, name, agents, problem, backward, TaskType, childpipe):
    # init agent
    class_ = getattr(importlib.import_module(agpath), agtype)
    workman = class_()
    workman.multinitialize(name, agents, problem, TaskType, backward)
    # load SWM and calculate the amount of new signs
    new_signs = workman.get_task()
    childpipe.send((name, new_signs))
    # load info about the major agent
    major_agent = childpipe.recv()
    # search solution and send it to major agent
    solution = workman.search_solution()
    childpipe.send(solution)
    if name == major_agent:
        # receive solution and create an auction
        solutions = childpipe.recv()
        logging.info("Solutions received by major agent %s" % name)
        keeper = DecisionStrategies(solutions)
        # can be changed to any other strategy
        agents, solution = keeper.auction()
        # ask agents whose plan won to save their solutions, to other agents - save won agent solution (find in their plans the won plan).
        childpipe.send(solution)
    # Save solution
    solution_to_save = childpipe.recv()
    file_name = workman.save_solution(solution_to_save)
    solution = [sol for sol in workman.allsolutions if sol[0] == workman.solution[:-1] or list(reversed(sol[0])) == workman.solution[:-1]][0]
    if workman.backward:
        solution = (list(reversed(solution[0])), solution[1])
    agent_solution = pickle.dumps(solution)
    childpipe.send((agent_solution,file_name))


class Manager:
    def __init__(self, agents, problem, agpath = 'mapmulti.agent.agent_search', agtype = 'MAgent', backward = False, TaskType = 'mapddl'):
        self.problem = problem
        self.solution = []
        self.finished = None
        self.agtype = agtype
        self.agpath = agpath
        self.backward = backward
        self.agents = agents
        self.logger = log_to_stderr()
        self.logger.setLevel(logging.INFO)
        self.TaskType = TaskType

    def manage_agents(self):

        allProcesses = []

        for ag in self.agents:
            parent_conn, child_conn = Pipe()
            p = Process(target=agent_activation,
                        args=(self.agpath, self.agtype,ag, self.agents, self.problem, self.backward, self.TaskType, child_conn, ))
            allProcesses.append((p, parent_conn))
            p.start()

        group_experience = []
        for pr, conn in allProcesses:
            group_experience.append((conn.recv(), conn))

        # Select the major (most experienced) agent
        most_exp = 0
        for info, _ in group_experience:
            if info[1] > most_exp:
                most_exp = info[1]

        major = [info[0] for info, conn in group_experience if info[1] == most_exp][0]

        # Major agent will create an auction and send back the best solution.
        for pr, conn in allProcesses:
            conn.send(major)

        solutions = {}
        # Receive solutions
        for info, conn in group_experience:
            solutions[info[0]] = conn.recv()

        # Send solutions to the major agent and receive final solution
        final_solution = None
        for info, conn in group_experience:
            if info[0] == major:
                conn.send(solutions)
                final_solution = conn.recv()
                break

        # Send final solution to all agents and get paths to experience files
        exp_path = {}
        for info, conn in group_experience:
            conn.send(final_solution)
            solution, path = conn.recv()
            exp_path[info[0]] = (pickle.loads(solution), path)

        for pr, conn in allProcesses:
            pr.join()

        return exp_path