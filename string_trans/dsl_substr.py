import os
import ipdb

import ply.lex as lex
from string_trans.consts import MAX_INDEX, MAX_INT, MAX_POSITION, SCHAR_LIST, INT_PREFIX
from string_trans.third_party import yacc
from string_trans.dsl import callout

class Parser:
    '''
    Base class for a lexer/parser that has the rules defined as methods
    '''
    tokens = ()
    precedence = ()

    def __init__(self, **kw):
        self.debug = kw.get('debug', 0)
        self.tabmodule = "substr_parsetab"
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

class SubStrDSL(Parser):
    tokens = (
        'CONCAT', 'CONSTSTR', 'SUBSTR',
        'REGEX', 'CONSTPOS',
        'WORD', 'NUM', 'ALPHANUM', 'ALLCAPS',
        'PROPCASE', 'LOWER', 'DIGIT', 'CHAR',
        'PROPER',
        'C_LBRACE', 'C_RBRACE', 'K_LBRACE', 'K_RBRACE',
        'S_LBRACE', 'S_RBRACE', 'P_LBRACE', 'P_RBRACE',
        'R_LBRACE', 'R_RBRACE',
        'INT', 'SCHAR', 'START', 'END'
    )

    # funcion
    t_CONCAT = 'Concat'
    t_CONSTSTR = 'ConstStr'
    t_SUBSTR = 'SubStr'
    t_REGEX = 'Regex'
    t_CONSTPOS = 'ConstPos'

    # regex
    t_WORD = 'Word'
    t_NUM = 'Num'
    t_ALPHANUM = 'Alphanum'
    t_ALLCAPS = 'Allcaps'
    t_PROPCASE = 'Propcase'
    t_LOWER = 'Lower'
    t_DIGIT = 'Digit'
    t_CHAR = 'Char'

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
        '''

        function = p[1]

        @self.callout
        def fn(st_world):
            function(st_world)

        p[0] = fn

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

