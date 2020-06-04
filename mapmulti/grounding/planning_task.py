from mapcore.planning.grounding.planning_task import PlanningTask

DEFAULT_FILE_PREFIX = 'wmodel_'
DEFAULT_FILE_SUFFIX = '.swm'

SIT_COUNTER = 0
SIT_PREFIX = 'situation_'
PLAN_PREFIX = 'action_'


class MaPlanningTask(PlanningTask):
    def __init__(self, name, signs, start_situation, goal_situation, subtasks=None):
        super().__init__(name, signs, start_situation, goal_situation, subtasks)
        self.They_signs = [con.in_sign for con in self.signs["They"].out_significances]
        self.agents.extend(self.They_signs)



