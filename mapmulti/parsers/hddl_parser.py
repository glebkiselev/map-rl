import re

import mapmulti.parsers.branch_parser as bch
from mapcore.planning.parsers.hddl_parser import HTNParser, Problem

class maHDDLParser(HTNParser):
    def __init__(self, domain_file, problem_file):
        super().__init__(domain_file, problem_file)

    def ParseBlock(self, descr):
        block = {}
        tokens = self.get_tokens(descr)
        flag = False
        my_token = self.tokenizer(tokens)
        start_token = next(my_token)
        while not flag:
            try:
                if start_token not in self.utokens:
                    start_token = next(my_token)
                    continue
                next_token = next(my_token)
                while next_token not in self.utokens:
                    next_token = next(my_token)
                    continue
                part = [''.join(el) for el in descr.split(start_token)[1].split(next_token)][0]
                while part[-1] != ')':
                    part = part[:-1]
                else:
                    part = part[:-1]
                parsed = getattr(bch, 'parse_'+start_token[1:])(part)
                if isinstance(parsed, list):
                    block[start_token[1:]] = parsed
                else:
                    block.setdefault(start_token[1:]+'s', []).append(parsed)

                if next_token != start_token:
                    self.utokens.remove(start_token)
                start_token = next_token
                descr = descr.split(part)[1]

            except StopIteration:
                part = [''.join(el) for el in descr.split(start_token)][1]
                while part[-1] != ')':
                    part = part[:-1]
                else:
                    part = part[:-1]
                if start_token != ':constraints':
                    parsed = getattr(bch, 'parse_'+start_token[1:])(part)
                    block.setdefault(start_token[1:] + 's', []).append(parsed)
                else:
                    parsed = getattr(bch, 'parse_' + start_token[1:])(part, block['init'], block['objects'])
                    block.setdefault(start_token[1:], {}).update(parsed)
                self.utokens.remove(start_token)
                flag = True
        return block

    def ParseProblem(self, descr, domain):
        self.utokens = [':objects', ':htn', ':init', ':constraints']
        task = self.ParseBlock(descr)
        problem_name = re.search('problem(.*)\)', descr)
        problem_name = problem_name.group(1)
        name = problem_name.strip()
        return MaProblem(name, domain, task)


class MaProblem(Problem):
    def __init__(self, name, domain, task):
        """
        name: The name of the problem
        domain: The domain in which the problem has to be solved
        objects: A dict name->type of objects that are used in the problem
        init: A list of predicates describing the initial state
        htns: high level actions that need to be completed in a current task
        constraints: constraints to each agent of a task
        """
        super().__init__(name, domain, task)
        self.constraints = task['constraints']

