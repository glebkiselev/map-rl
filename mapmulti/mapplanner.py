import logging
import os

from mapcore.planning.mapplanner import MapPlanner as MPcore
from mapmulti.agent.planning_agent import Manager
SOLUTION_FILE_SUFFIX = '.soln'

import platform

if platform.system() != 'Windows':
    delim = '/'
else:
    delim = '\\'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("process-main")

class MapPlanner(MPcore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def action_agents(self, problem):
        agents = set()
        for _, action in problem.domain.actions.items():
            for ag in action.agents:
                for obj, type in problem.objects.items():
                    if type.name == ag:
                        agents.add(obj)
        return agents

    def _parse_mapddl(self):
        """
        multiagent pddl parser and agents search
        """

        problem = self._parse_pddl()
        act_agents = self.action_agents(problem)
        logging.debug('Agents found in actions: {0}'.format(len(act_agents)))
        agents = set()
        if problem.constraints:
            if len(act_agents):
                agents |= act_agents
            else:
                for constr in problem.constraints:
                    agents.add(constr)
                logging.debug('Agents found in constraints: {0}'.format(len(agents)))
        elif act_agents:
            agents |= act_agents
        else:
            for item, type in problem.objects.items():
                if type.name == 'agent':
                    agents.add(item)
            if not agents:
                agents.add("I")
                logging.debug("Can not find any agents. Thinking that it is only 1.")
        return agents, problem

    def _parse_mahddl(self):
        """
        multiagent HTN-based parser and agents search
        :return: the final solution
        """
        from mapmulti.parsers.hddl_parser import maHDDLParser
        parser = maHDDLParser(self.domain, self.problem)
        logging.info('Распознаю классическую многоагентную задачу...')
        logging.debug('Распознаю домен {0}'.format(self.domain))
        domain = parser.ParseDomain(parser.domain)
        logging.debug('Распознаю проблему {0}'.format(self.problem))
        problem = parser.ParseProblem(parser.problem, domain)
        logging.debug('{0} Predicates parsed'.format(len(domain['predicates'])))
        logging.debug('{0} Actions parsed'.format(len(domain['actions'])))
        logging.debug('{0} Methods parsed'.format(len(domain['methods'])))
        agents = list(problem.constraints.keys())
        return agents, problem

    def find_domain(self, domain, path, number):
        """
        Domain search function
        :param path: path to current task
        :param number: task number
        :return:
        """
        ext = '.pddl'
        if self.TaskType == 'mahddl':
            ext = '.hddl'
        task = 'task' + number + ext
        domain += ext
        if not domain in os.listdir(path):
            domain2 = self.search_upper(path, domain)
            if not domain2:
                raise Exception('domain not found!')
            else:
                domain = domain2
        else:
            domain = path + domain
        if not task in os.listdir(path):
            raise Exception('task not found!')
        else:
            problem = path + task

        return domain, problem

    def search(self):
        if self.TaskType == 'hddl':
            problem = self._parse_hddl()
            agents = set('I')
        elif self.TaskType == 'pddl':
            problem = self._parse_pddl()
            agents = set('I')
        elif self.TaskType == 'mapddl':
            agents, problem = self._parse_mapddl()
        elif self.TaskType == 'mahddl':
            agents, problem = self._parse_mahddl()
        else:
            raise Exception('You are using multiagent lib without extensions. Tasks can be pddl, hddl, mapddl or mahddl!!!')
        logger.info('Parsing was finished...')
        manager = Manager(agents, problem, self.agpath, TaskType = self.TaskType, backward=self.backward)
        solution = manager.manage_agents()

        return solution

