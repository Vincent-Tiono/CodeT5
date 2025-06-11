# Microsoft RobustFill DSL

import os
from functools import wraps
import ipdb

import ply.lex as lex
from string_trans.consts import MAX_INDEX, MAX_INT, MAX_POSITION, SCHAR_LIST, INT_PREFIX, MIN_INT
from string_trans.third_party import yacc

def callout(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        r = f(*args, **kwargs)
        return r
    return wrapped

class Parser:
    '''
    Base class for a lexer/parser that has the rules defined as methods
    '''
    tokens = ()
    precedence = ()

    def __init__(self, **kw):
        self.debug = kw.get('debug', 0)
        self.tabmodule = "dsl_parsetab"
        self.max_func_call = 220

        self.callout = callout

        self.names = {}
        try:
            modname = os.path.split(os.path.splitext(__file__)[0])[
                1] + '_' + self.__class__.__name__
        except:
            modname = 'parser' + '_' + self.__class__.__name__
        self.debugfile = modname + '.dbg'
        # print self.debugfile

        # Build the lexer and parser
        self.lexer = lex.lex(module=self, debug=self.debug)
        self.yacc, self.grammar = yacc.yacc(optimize=True,
                                            module=self,
                                            debug=self.debug,
                                            debugfile=self.debugfile,
                                            tabmodule=self.tabmodule,
                                            with_grammar=True)

    def parse(self, code, **kwargs):
        self.error = False
        program = yacc.parse(code, **kwargs)
        return program

    def run(self, world, code, **kwargs):
        program = self.parse(code, **kwargs)

        # run program
        program(world)
        return world.s_h


class StringTransformationDSL(Parser):
    tokens = (
        'CONCAT', 'CONSTSTR', 'SUBSTR',
        'REGEX', 'CONSTPOS',
        'WORD', 'NUM', 'ALPHANUM', 'ALLCAPS',
        'PROPCASE', 'LOWER', 'DIGIT', 'CHAR',
        'PROPER',
        'GETTOKEN', 'TOCASE', 'REPLACE', 'TRIM',
        'GETUPTO', 'GETFROM', 'GETFIRST', 'GETALL',
        'C_LBRACE', 'C_RBRACE', 'K_LBRACE', 'K_RBRACE',
        'S_LBRACE', 'S_RBRACE', 'P_LBRACE', 'P_RBRACE',
        'R_LBRACE', 'R_RBRACE', 'N_LBRACE', 'N_RBRACE',
        'V_LBRACE', 'V_RBRACE',
        'INT', 'SCHAR', 'START', 'END'
    )

    # funcion
    t_CONCAT = 'Concat'
    t_CONSTSTR = 'ConstStr'
    t_SUBSTR = 'SubStr'
    t_REGEX = 'Regex'
    t_CONSTPOS = 'ConstPos'

    # regex | case
    t_WORD = 'Word'
    t_NUM = 'Num'
    t_ALPHANUM = 'Alphanum'
    t_ALLCAPS = 'Allcaps'
    t_PROPCASE = 'Propcase'
    t_LOWER = 'Lower'
    t_DIGIT = 'Digit'
    t_CHAR = 'Char'
    t_PROPER = 'Proper'

    # nesting
    t_GETTOKEN = 'GetToken'
    t_TOCASE = 'ToCase'
    t_REPLACE = 'Replace'
    t_TRIM = 'Trim'
    t_GETUPTO = 'GetUpTo'
    t_GETFROM = 'GetFrom'
    t_GETFIRST = 'GetFirst'
    t_GETALL = 'GetAll'

    # boundry
    t_START = 'Start'
    t_END = 'End'

    # ingore
    t_ignore = ' \t\n'

    # braces
    t_C_LBRACE = 'c\('
    t_C_RBRACE = 'c\)'
    t_K_LBRACE = 'k\('
    t_K_RBRACE = 'k\)'
    t_S_LBRACE = 's\('
    t_S_RBRACE = 's\)'
    t_P_LBRACE = 'p\('
    t_P_RBRACE = 'p\)'
    t_R_LBRACE = 'r\('
    t_R_RBRACE = 'r\)'
    t_N_LBRACE = 'n\('
    t_N_RBRACE = 'n\)'
    t_V_LBRACE = 'v\('
    t_V_RBRACE = 'v\)'

    def construct_vocab(self):
        program_tokens = []
        for term in self.tokens:
            token = getattr(self, "t_{}".format(term))
            if callable(token):
                if token == self.t_INT:
                    for i in range(MIN_INT, MAX_INT + 1):
                        program_tokens.append("{}{}".format(INT_PREFIX, i))
                if token == self.t_SCHAR:
                    for c in SCHAR_LIST:
                        program_tokens.append("\"{}\"".format(c))
            else:
                program_tokens.append(str(token).replace("\\", ""))
        return program_tokens

    def t_INT(self, t):
        r'-?\d+'

        value = int(t.value.replace(INT_PREFIX, ''))
        if not (- MAX_INT <= value <= MAX_INT):
            raise Exception(" [!] Out of range ({} ~ {}): `{}`".
                            format(-MAX_INT, MAX_INT, value))

        t.value = value
        return t

    def t_SCHAR(self, t):
        r'".(PACE)?"'

        if t.value[1:-1] not in SCHAR_LIST:
            raise Exception(" [!] Not in {}".
                            format("".join(SCHAR_LIST)))

        return t

    def t_error(self, t):
        self.error = True
        t.lexer.skip(1)

    def p_error(self, p):
        raise RuntimeError('Syntax Error')

    def p_prog(self, p):
        '''prog : CONCAT C_LBRACE expr C_RBRACE'''
        expr = p[3]
        p[0] = expr

    def p_expr(self, p):
        '''expr : conststr
                | substr
                | expr_expr
                | nesting
                | nesting_plus
        '''

        function = p[1]

        @self.callout
        def fn(st_world):
            function(st_world)

        p[0] = fn

    def p_nesting(self, p):
        '''nesting : GETTOKEN N_LBRACE type index N_RBRACE
                   | TOCASE N_LBRACE case N_RBRACE
                   | REPLACE N_LBRACE SCHAR SCHAR N_RBRACE
                   | TRIM N_LBRACE N_RBRACE
                   | GETUPTO N_LBRACE reg N_RBRACE
                   | GETFROM N_LBRACE reg N_RBRACE
                   | GETFIRST N_LBRACE type index N_RBRACE
                   | GETALL N_LBRACE type N_RBRACE
        '''
        function = p[1]

        if function == 'GetToken':
            type = p[3]
            index = p[4]()

            @self.callout
            def fn(st_world):
                st_world.nesting(function, type=type, index=index)
            p[0] = fn
        elif function == 'ToCase':
            case = p[3]

            @self.callout
            def fn(st_world):
                st_world.nesting(function, case=case)
            p[0] = fn
        elif function == 'Replace':
            sch1 = p[3]
            sch2 = p[4]

            @self.callout
            def fn(st_world):
                st_world.nesting(function, sch1=sch1, sch2=sch2)
            p[0] = fn
        elif function == 'Trim':
            @self.callout
            def fn(st_world):
                st_world.nesting(function)
            p[0] = fn
        elif function == 'GetUpTo':
            reg = p[3]

            @self.callout
            def fn(st_world):
                st_world.nesting(function, reg=reg)
            p[0] = fn
        elif function == 'GetFrom':
            reg = p[3]

            @self.callout
            def fn(st_world):
                st_world.nesting(function, reg=reg)
            p[0] = fn
        elif function == 'GetFirst':
            type = p[3]
            index = p[4]()

            @self.callout
            def fn(st_world):
                st_world.nesting(function, type=type, index=index)
            p[0] = fn
        elif function == 'GetAll':
            type = p[3]

            @self.callout
            def fn(st_world):
                st_world.nesting(function, type=type)
            p[0] = fn

    def p_nesting_plus(self, p):
        '''nesting_plus : nesting V_LBRACE substr V_RBRACE
                        | nesting V_LBRACE nesting V_RBRACE
        '''
        function1 = p[1]
        function2 = p[3]

        @self.callout
        def fn(st_world):
            function2(st_world)
            st_world.set_nesting()
            function1(st_world)
            st_world.unset_nesting()

        p[0] = fn

    def p_case(self, p):
        '''case : PROPER
                | ALLCAPS
                | LOWER
        '''
        p[0] = p[1]

    def p_expr_expr(self, p):
        '''expr_expr : expr expr
        '''
        expr1, expr2 = p[1], p[2]

        @self.callout
        def fn(st_world):
            expr1(st_world)
            st_world.current_length += 1
            expr2(st_world)

        p[0] = fn

    def p_conststr(self, p):
        '''conststr : CONSTSTR K_LBRACE SCHAR K_RBRACE
        '''

        schar = p[3]

        @self.callout
        def fn(st_world):
            st_world.const_str(schar)

        p[0] = fn

    def p_substr(self, p):
        '''substr : SUBSTR S_LBRACE pos pos S_RBRACE
        '''

        pos1 = p[3]
        pos2 = p[4]

        @self.callout
        def fn(st_world):
            st_world.sub_str(pos1(st_world), pos2(st_world))

        p[0] = fn

    def p_pos(self, p):
        '''pos : REGEX R_LBRACE reg index START R_RBRACE
               | REGEX R_LBRACE reg index END R_RBRACE
               | CONSTPOS P_LBRACE position P_RBRACE
        '''

        if len(p) == 5:
            pos = p[3]()

            def fn(st_world):
                return pos

            p[0] = fn
        else:
            reg = p[3]
            index = p[4]()
            boundary = p[5]

            def fn(st_world):
                return st_world.regex(reg, index, boundary)

            p[0] = fn

    def p_reg(self, p):
        '''reg : SCHAR
               | type
        '''
        p[0] = p[1]

    def p_type(self, p):
        '''type : WORD
                | NUM
                | ALPHANUM
                | ALLCAPS
                | PROPCASE
                | LOWER
                | DIGIT
                | CHAR
        '''
        p[0] = p[1]

    def p_index(self, p):
        '''index : INT
        '''
        value = p[1]
        if abs(int(value)) <= MAX_INDEX:
            p[0] = lambda: int(value)
        else:
            raise ValueError("Index out of range")

    def p_position(self, p):
        '''position : INT
        '''
        value = p[1]
        if abs(int(value)) <= MAX_POSITION:
            p[0] = lambda: int(value)
        else:
            raise ValueError("Position out of range")
