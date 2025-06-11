from hprl_karel_env.karel_option import Karel_world
import numpy as np
import ipdb

branch_table = {
    'while': 0,
    'repeat': 1,
    'if': 2,
    'ifelse': 3,
    'else': 4,
}

function_table = {
    'frontIsClear': 0,
    'leftIsClear': 1,
    'rightIsClear': 2,
    'markersPresent': 3,
    'noMarkersPresent': 4,
    'not': 5,
}

class HPRLFunctionCall(Karel_world):
    def __init__(self, s=None, make_error=True, env_task="program", task_definition='program' ,reward_diff=False, final_reward_scale=True):
        super(HPRLFunctionCall, self).__init__(s=s, make_error=make_error, env_task=env_task, task_definition=task_definition, reward_diff=reward_diff, final_reward_scale=final_reward_scale)
        self.function_calls = []
        self.branch_calls = []
        self.function_history = []
        self.branch_history = []

    def set_new_state(self, s, metadata=None):
        super(HPRLFunctionCall, self).set_new_state(s, metadata=metadata)
        self.function_calls = []
        self.branch_calls = []
        self.function_history = []
        self.branch_history = []

    def add_function_call(self, function_call):
        self.function_calls.append(function_call)

    def add_branch_call(self, branch_call):
        self.branch_calls.append(branch_call)

    def add_callee(self):
        # function call to multi-binary
        function = np.zeros((len(function_table),), dtype=bool)
        for f in self.function_calls:
            function[function_table[f]] = 1
        self.function_history.append(function)
        self.function_calls = []

        # branch call to multi-binary
        branch = np.zeros((len(branch_table),), dtype=bool)
        for b in self.branch_calls:
            branch[branch_table[b]] = 1
        self.branch_history.append(branch)
        self.branch_calls = []