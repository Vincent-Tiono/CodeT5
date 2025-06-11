from string_trans.dsl import StringTransformationDSL
from string_trans.dsl_substr import SubStrDSL

from copy import deepcopy
import ipdb


class ProgramFunctionality(StringTransformationDSL):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.nesting = [self.t_GETTOKEN, self.t_TOCASE, self.t_REPLACE, self.t_TRIM,
                        self.t_GETUPTO, self.t_GETFROM, self.t_GETFIRST, self.t_GETALL]

        self.nesting_plus = deepcopy(self.nesting)
        self.nesting_plus.append(self.t_SUBSTR)

        self.all_functions = {self.t_CONSTSTR: 0, self.t_SUBSTR: 0}
        for n in self.nesting:
            self.all_functions[n] = 0
        for n in self.nesting:
            for n_p in self.nesting_plus:
                self.all_functions[f"{n}_{n_p}"] = 0
        
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
        p[0] = p[1]

    def p_conststr(self, p):
        '''conststr : CONSTSTR K_LBRACE SCHAR K_RBRACE
        '''
        p[0] = self.t_CONSTSTR

    def p_substr(self, p):
        '''substr : SUBSTR S_LBRACE pos pos S_RBRACE
        '''
        p[0] = self.t_SUBSTR

    def p_expr_expr(self, p):
        '''expr_expr : expr expr
        '''
        p[0] = f"{p[1]}-{p[2]}"

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
        p[0] = p[1]

    def p_nesting_plus(self, p):
        '''nesting_plus : nesting V_LBRACE substr V_RBRACE
                        | nesting V_LBRACE nesting V_RBRACE
        '''
        p[0] = f"{p[1]}_{p[3]}"


class SubstrFunctionality(SubStrDSL):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.all_functions = {self.t_CONSTSTR: 0, self.t_SUBSTR: 0}

    def p_prog(self, p):
        '''prog : CONCAT C_LBRACE expr C_RBRACE'''
        expr = p[3]
        p[0] = expr

    def p_expr(self, p):
        '''expr : conststr
                | substr
                | expr_expr
        '''
        p[0] = p[1]

    def p_conststr(self, p):
        '''conststr : CONSTSTR K_LBRACE SCHAR K_RBRACE
        '''
        p[0] = self.t_CONSTSTR

    def p_substr(self, p):
        '''substr : SUBSTR S_LBRACE pos pos S_RBRACE
        '''
        p[0] = self.t_SUBSTR

    def p_expr_expr(self, p):
        '''expr_expr : expr expr
        '''
        p[0] = f"{p[1]}-{p[2]}"


if __name__ == "__main__":
    p = "Concat c( GetToken n( Allcaps -2 n) v( SubStr s( Regex r( Allcaps 0 Start r) Regex r( Allcaps 4 Start r) s) v) c)"
    pf = ProgramFunctionality()
    res = pf.yacc.parse(p)
    print(res)

    p = "Concat c( SubStr s( Regex r( \"(\" 3 End r) ConstPos p( 28 p) s) SubStr s( Regex r( \"/\" 3 End r) Regex r( \"SPACE\" 0 Start r) s) c)"
    sf = SubstrFunctionality()
    res = sf.yacc.parse(p)
    print(res)

