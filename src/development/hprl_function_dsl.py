"""
Domain specific language for Karel Environment

Code is adapted from https://github.com/carpedm20/karel
"""

import numpy as np

from hprl_karel_env.dsl.dsl_base_option_v2 import DSLBase, MIN_INT, MAX_INT, INT_PREFIX
from hprl_karel_env.dsl.dsl_data_option_v2 import DSLData


class DSLProb_option_v2(DSLBase, DSLData):
    t_ignore = ' \t\n'

    t_M_LBRACE = 'm\('
    t_M_RBRACE = 'm\)'

    t_C_LBRACE = 'c\('
    t_C_RBRACE = 'c\)'

    t_R_LBRACE = 'r\('
    t_R_RBRACE = 'r\)'

    t_W_LBRACE = 'w\('
    t_W_RBRACE = 'w\)'

    t_I_LBRACE = 'i\('
    t_I_RBRACE = 'i\)'

    t_E_LBRACE = 'e\('
    t_E_RBRACE = 'e\)'

    t_DEF = 'DEF'
    t_RUN = 'run'
    t_WHILE = 'WHILE'
    t_REPEAT = 'REPEAT'
    t_IF = 'IF'
    t_IFELSE = 'IFELSE'
    t_ELSE = 'ELSE'
    t_NOT = 'not'

    def __init__(self, seed=None, environment='karel'):
        DSLData.__init__(self, environment=environment)
        DSLBase.__init__(self, seed=seed)

    #########
    # lexer
    #########

    def t_INT(self, t):
        r'R=\d+'

        value = int(t.value.replace(INT_PREFIX, ''))
        if not (MIN_INT <= value <= MAX_INT):
            raise Exception(" [!] Out of range ({} ~ {}): `{}`".
                            format(MIN_INT, MAX_INT, value))

        t.value = value
        return t

    def random_INT(self):
        return "{}{}".format(
            INT_PREFIX,
            self.rng.randint(MIN_INT, MAX_INT + 1))

    def t_error(self, t):
        self.error = True
        t.lexer.skip(1)

    #########
    # parser
    #########

    def p_prog(self, p):
        '''prog : DEF RUN M_LBRACE stmt M_RBRACE'''
        stmt = p[4]

        @self.callout
        def fn(karel_world):
            stmt(karel_world)
        p[0] = stmt

    def p_stmt(self, p):
        '''stmt : while
                | repeat
                | stmt_stmt
                | action
                | if
                | ifelse
        '''
        function = p[1]

        @self.callout
        def fn(karel_world):
            function(karel_world)
        p[0] = fn

    def p_stmt_stmt(self, p):
        '''stmt_stmt : stmt stmt
        '''
        stmt1, stmt2 = p[1], p[2]

        @self.callout
        def fn(karel_world):
            stmt1(karel_world)
            stmt2(karel_world)
        p[0] = fn

    def p_if(self, p):
        '''if : IF C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE
        '''
        cond, stmt = p[3], p[6]

        @self.callout
        def fn(karel_world):
            condition = cond(karel_world)
            karel_world.add_branch_call("if")
            if condition != 'timeout' and condition:
                stmt(karel_world)

        p[0] = fn

    def p_ifelse(self, p):
        '''ifelse : IFELSE C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE ELSE E_LBRACE stmt E_RBRACE
        '''
        cond, stmt1, stmt2 = p[3], p[6], p[10]

        @self.callout
        def fn(karel_world):
            condition = cond(karel_world)
            karel_world.add_branch_call("if")
            if condition != 'timeout' and condition:
                karel_world.add_branch_call("ifelse")
                stmt1(karel_world)
            elif condition != 'timeout':
                karel_world.add_branch_call("else")
                stmt2(karel_world)
            else:
                return

        p[0] = fn

    def p_while(self, p):
        '''while : WHILE C_LBRACE cond C_RBRACE W_LBRACE stmt W_RBRACE
        '''
        cond, stmt = p[3], p[6]

        @self.callout
        def fn(karel_world):
            condition = cond(karel_world)
            karel_world.add_branch_call("while")
            while(condition != 'timeout' and condition):
                stmt(karel_world)
                condition = cond(karel_world)

        p[0] = fn

    def p_repeat(self, p):
        '''repeat : REPEAT cste R_LBRACE stmt R_RBRACE
        '''
        cste, stmt = p[2], p[4]

        @self.callout
        def fn(karel_world):
            karel_world.add_branch_call("repeat")
            for _ in range(cste()):
                stmt(karel_world)

        p[0] = fn

    def p_cond(self, p):
        '''cond : cond_without_not
                | NOT C_LBRACE cond_without_not C_RBRACE
        '''
        if callable(p[1]):
            cond_without_not = p[1]

            def fn(karel_world):
                return cond_without_not(karel_world)
            p[0] = fn
        else:  # NOT
            cond_without_not = p[3]
            def fn(karel_world):
                karel_world.add_function_call("not")
                return not cond_without_not(karel_world)
            p[0] = fn

    def p_cond_without_not(self, p):
        '''cond_without_not : Dynamic docstring for PRL
        '''
        cond_without_not = p[1]

        def fn(karel_world):
            if cond_without_not in self.conditional_functions_dict:
                karel_world.add_function_call(cond_without_not)
                return self.conditional_functions_dict[cond_without_not](karel_world)()
            else:
                raise ValueError("No such condition")

        p[0] = fn

    def p_action(self, p):
        '''action : Dynamic docstring for PRL
        '''
        action = p[1]

        def fn(karel_world):
            action_v = np.array(self.action_functions) == action
            karel_world.state_transition(action_v)
            karel_world.add_callee()
        p[0] = fn

    def p_cste(self, p):
        '''cste : INT
        '''
        value = p[1]
        p[0] = lambda: int(value)