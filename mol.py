import string
from error_handling import *

# Variables

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS



# position

class Position:
    def __init__(self, index, line, column, fn, ftxt ):
        self.index = index
        self.line = line
        self.column = column
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.index += 1
        self.column += 1

        if current_char == '\n':
            self.line += 1
            self.column = 0
        
        return self
        
    def copy(self):
        return Position(self.index, self.line, self.column, self.fn, self.ftxt)

# Tokens

TOKEN_INT = 'INT'
TOKEN_FLOAT = 'FLOAT'
TOKEN_PLUS = 'PLUS'
TOKEN_IDENTIFIER = 'IDENTIFIER'
TOKEN_KEYWORK = 'KEYWORD'
TOKEN_MINUS = 'MINUS'
TOKEN_MUL = 'MUL'
TOKEN_DIV = 'DIV'
TOKEN_POW = 'POW'
TOKEN_EQUALS = 'EQ'
TOKEN_LPAREN = 'LPAREN'
TOKEN_RPAREN = 'RPAREN'
TOKEN_EE = 'EE'
TOKEN_NE = 'NE'
TOKEN_LT = 'LT'
TOKEN_GT = 'GT'
TOKEN_LTE = 'LTE'
TOKEN_GTE = 'GTE'
TOKEN_COMMA = 'COMMA'
TOKEN_ARROW = 'ARROW'
TOKEN_EOF = 'EOF'

KEYWORKS = [
    'VAR',
    'AND',
    'OR',
    'NOT',
    'IF',
    'THEN',
    'ELIF',
    'ELSE',
    'FOR',
    'TO',
    'STEP',
    'WHILE'
]

class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'


class Lexer:
    def __init__(self, text, file_name):
        self.text = text
        self.fn = file_name
        self.pos = Position(-1, 0, -1, file_name, text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.index] if self.pos.index < len(self.text) else None
        # print(f'{len(self.text)} - {self.pos.index} - {self.current_char}')

    def make_tokens(self):
        tokens = []
        while self.current_char is not None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_numbers())
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char == '+':
                tokens.append(Token(TOKEN_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TOKEN_MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '*':
                tokens.append(Token(TOKEN_MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TOKEN_DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == '^':
                tokens.append(Token(TOKEN_POW, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TOKEN_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TOKEN_RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == '!':
                tok, error = self.make_not_equals()
                if error: return [], error
                tokens.append(tok)
            elif self.current_char == '=':
                tokens.append(self.make_equals())
            elif self.current_char == '<':
                tokens.append(self.make_less_than())
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError("'" + char + "'", pos_start, self.pos)
        
        tokens.append(Token(TOKEN_EOF, pos_start=self.pos))
        return tokens, None

    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
            id_str += self.current_char 
            self.advance()

        tok_type = TOKEN_KEYWORK if id_str in KEYWORKS else TOKEN_IDENTIFIER
        return Token(tok_type, id_str, pos_start, self.pos)

    def make_numbers(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TOKEN_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TOKEN_FLOAT, float(num_str), pos_start, self.pos)
    
    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            return Token(TOKEN_NE, pos_start=pos_start, pos_end=self.pos), None

        self.advance()
        return None, ExpectedCharError("Expected '=' after '!'", pos_start, self.pos)

    def make_equals(self):
        tok_type = TOKEN_EQUALS
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TOKEN_EE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_less_than(self):
        tok_type = TOKEN_LT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TOKEN_LTE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_greater_than(self):
        tok_type = TOKEN_GT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TOKEN_GTE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

class NumberNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    
    def __repr__(self):
        return f'{self.tok}'
    
class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok
        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end

class VarAssignNode:
    def __init__(self, var_name_tok, value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end

class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'

class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node
    
        self.pos_start = self.op_tok.pos_start
        self.pos_end = node.pos_end

    def __repr__(self):
        return f'({self.op_tok}, {self.node})'

class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case

        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (self.else_case or self.cases[len(self.cases) - 1][0]).pos_end

class ForNode:
    def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node):
        self.var_name_tok = var_name_tok
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.step_value_node = step_value_node
        self.body_node = body_node

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.body_node.pos_end

class WhileNode:
    def __init__(self, condition_node, body_node):
        self.condition_node = condition_node
        self.body_node = body_node


        self.pos_start = self.condition_node.pos_start
        self.pos_end = self.body_node.pos_end
class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0

    def register_advancement(self):
        self.advance_count += 1
        pass

    def register(self, res):
        self.advance_count += res.advance_count
        if res.error: self.error = res.error
        return res.node
        
    def success(self, node):
        self.node = node
        return self
    
    def failure(self, error):
        if not self.error or self.advance_count == 0:
            self.error = error
        return self
    
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    def parse(self):
        res = self.expr()

        if not res.error and self.current_tok.type != TOKEN_EOF:
            return res.failure(InvalidSyntaxError(
                "Expected '+', '-', '*' or '/'",
                self.current_tok.pos_start, 
                self.current_tok.pos_end,
                ))
        
        return res

    def atom(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TOKEN_INT, TOKEN_FLOAT):
            res.register_advancement()
            self.advance()
            return res.success(NumberNode(tok))

        elif tok.type == TOKEN_IDENTIFIER:
            res.register_advancement()
            self.advance()
            return res.success(VarAccessNode(tok))

        elif tok.type == TOKEN_LPAREN:
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.type == TOKEN_RPAREN:
                res.register_advancement()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    "Expected ')'",
                    self.current_tok.pos_start,
                    self.current_tok.pos_end
                ))
        
        if self.current_tok.matches(TOKEN_KEYWORK, 'IF'):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)
        
        elif self.current_tok.matches(TOKEN_KEYWORK, 'FOR'):
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)

        elif self.current_tok.matches(TOKEN_KEYWORK, 'WHILE'):
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)

        return res.failure(InvalidSyntaxError(
            "Expected int, float, identifier, '+', '-', or '('",
            tok.pos_start, 
            tok.pos_end,
        ))

    def for_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(TOKEN_KEYWORK, 'FOR'):
            return res.failure(InvalidSyntaxError(
                f"Expected 'FOR'",
                self.current_tok.pos_start, 
                self.current_tok.pos_end
            ))

        res.register_advancement()
        self.advance()

        if self.current_tok.type != TOKEN_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                f"Expected identifier",
                self.current_tok.pos_start, 
                self.current_tok.pos_end
            ))

        var_name = self.current_tok
        res.register_advancement()
        self.advance()

        if self.current_tok.type != TOKEN_EQUALS:
            return res.failure(InvalidSyntaxError(
                f"Expected '='",
                self.current_tok.pos_start, 
                self.current_tok.pos_end
            ))
        
        res.register_advancement()
        self.advance()

        start_value = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TOKEN_KEYWORK, 'TO'):
            return res.failure(InvalidSyntaxError(
                f"Expected 'TO'",
                self.current_tok.pos_start,
                self.current_tok.pos_end
            ))
        
        res.register_advancement()
        self.advance()

        end_value = res.register(self.expr())
        if res.error: return res

        if self.current_tok.matches(TOKEN_KEYWORK, 'STEP'):
            res.register_advancement()
            self.advance()

            step_value = res.register(self.expr())
            if res.error: return res
        else:
            step_value = None

        if not self.current_tok.matches(TOKEN_KEYWORK, 'THEN'):
            return res.failure(InvalidSyntaxError(
                f"Expected 'THEN'",
                self.current_tok.pos_start, 
                self.current_tok.pos_end
            ))

        res.register_advancement()
        self.advance()

        body = res.register(self.expr())
        if res.error: return res

        return res.success(ForNode(var_name, start_value, end_value, step_value, body))

    def while_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(TOKEN_KEYWORK, 'WHILE'):
            return res.failure(InvalidSyntaxError(
                f"Expected 'WHILE'",
                self.current_tok.pos_start,
                self.current_tok.pos_end
            ))

        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TOKEN_KEYWORK, 'THEN'):
            return res.failure(InvalidSyntaxError(
                f"Expected 'THEN'",
                self.current_tok.pos_start, 
                self.current_tok.pos_end
            ))

        res.register_advancement()
        self.advance()

        body = res.register(self.expr())
        if res.error: return res

        return res.success(WhileNode(condition, body))

    def if_expr(self):
        res = ParseResult()
        cases = []
        else_case = None

        if not self.current_tok.matches(TOKEN_KEYWORK, 'IF'):
            return res.failure(InvalidSyntaxError(
                f"Expected 'IF'",
                self.current_tok.pos_start, 
                self.current_tok.pos_end
            ))

        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(TOKEN_KEYWORK, 'THEN'):
            return res.failure(InvalidSyntaxError(
                f"Expected 'THEN'",
                self.current_tok.pos_start,
                self.current_tok.pos_end
            ))

        res.register_advancement()
        self.advance()

        expr = res.register(self.expr())
        if res.error: return res
        cases.append((condition, expr))

        while self.current_tok.matches(TOKEN_KEYWORK, 'ELIF'):
            res.register_advancement()
            self.advance()

            condition = res.register(self.expr())
            if res.error: return res

            if not self.current_tok.matches(TOKEN_KEYWORK, 'THEN'):

                return res.failure(InvalidSyntaxError(
                    f"Expected 'THEN'",
                    self.current_tok.pos_start, 
                    self.current_tok.pos_end
                ))

            res.register_advancement()
            self.advance()

            expr = res.register(self.expr())
            if res.error: return res
            cases.append((condition, expr))

        if self.current_tok.matches(TOKEN_KEYWORK, 'ELSE'):
            res.register_advancement()
            self.advance()

            else_case = res.register(self.expr())
            if res.error: return res

        return res.success(IfNode(cases, else_case))

    def power(self):
        return self.bin_op(self.atom, (TOKEN_POW, ), self.factor)

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TOKEN_PLUS, TOKEN_MINUS):
            res.register_advancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok,factor))


        return self.power()

    def term(self):
        return self.bin_op(self.factor, (TOKEN_MUL, TOKEN_DIV))
    
    def arith_expr(self):
        return self.bin_op(self.term, (TOKEN_PLUS, TOKEN_MINUS))


    def comp_expr(self):
        res = ParseResult()

        if self.current_tok.matches(TOKEN_KEYWORK, 'NOT'):
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()

            node = res.register(self.comp_expr())
            if res.error: return res

            return res.success(UnaryOpNode(op_tok, node))
        
        node = res.register(self.bin_op(self.arith_expr, (TOKEN_EE, TOKEN_NE, TOKEN_LT, TOKEN_GT, TOKEN_LTE, TOKEN_GTE)))
        
        if res.error:
            return res.failure(InvalidSyntaxError(
                    "Expected int, float, identifier, '+', '-', 'NOT' or '('",
                    self.current_tok.pos_start,
                    self.current_tok.pos_end
                ))

        return res.success(node)

    def expr(self):
        res = ParseResult()

        if self.current_tok.matches(TOKEN_KEYWORK, 'VAR'):
            res.register_advancement()
            self.advance()
            if self.current_tok.type != TOKEN_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    "Expected Identifier",
                    self.current_tok.pos_start,
                    self.current_tok.pos_end
                ))
            
            var_name = self.current_tok
            res.register_advancement()
            self.advance()

            if self.current_tok.type != TOKEN_EQUALS:
                return res.failure(InvalidSyntaxError(
                    "Expected '=",
                    self.current_tok.pos_start,
                    self.current_tok.pos_end
                ))
            
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())

            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))

        node = res.register(self.bin_op(self.comp_expr, ((TOKEN_KEYWORK, 'AND'), (TOKEN_KEYWORK, 'OR'))))

        if res.error:
            return res.failure(InvalidSyntaxError(
                    "Expected 'VAR', int, float, identifier, '+', '-', 'NOT', or '('",
                    self.current_tok.pos_start,
                    self.current_tok.pos_end
                ))
        
        return res.success(node)

    def bin_op(self, func_a, ops, func_b=None):
        if func_b == None:
            func_b = func_a

        
        res = ParseResult()
        left = res.register(func_a())

        if res.error: return res

        while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
            op_tok = self.current_tok
            res.register_advancement()
            self.advance()
            right = res.register(func_b())
            if res.error: return res

            left = BinOpNode(left, op_tok, right)

        return res.success(left)

class RTResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error: self.error = res.error
        return res.value

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self

class Number:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    
    def set_context(self, context=None):
        self.context = context
        return self
    
    def added_by(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
   
    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
    
    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
    
    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    "Division by zero",
                    other.pos_start,
                    other.pos_end,
                    self.context
                )
            
            return Number(self.value / other.value).set_context(self.context), None
    
    def powed_by(self , other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None
        
    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None

    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None  

    def notted(self):
        return Number(1 if self.value == 0 else 0).set_context(self.context), None
        
    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy
    
    def is_true(self):
        return self.value != 0

    def __repr__(self):
        return str(self.value)

class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None

    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value
    
    def set(self, name, value):
        self.symbols[name] = value

    def remove(self, name):
        del self.symbols[name]


class Interpreter:
    def visit(self,node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)    
        )
    
    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)

        if not value:
            return res.failure(RTError(
                f"'{var_name}' is not defined",
                node.pos_start,
                node.pos_end,
                context
            ))
        
        value = value.copy().set_pos(node.pos_start, node.pos_end)
        return res.success(value)
    
    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, context))
        if res.error: return res

        context.symbol_table.set(var_name,value)
        return res.success(value)


    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error : return res
        right = res.register(self.visit(node.right_node, context))
        if res.error : return res

        print('-----------------------------------')
        print(f'Operator {node.op_tok.type}')
        print(f'Left node {left.value}')
        print(f'Right node {right.value}')

        if node.op_tok.type == TOKEN_PLUS:
            result, error = left.added_by(right)
        elif node.op_tok.type == TOKEN_MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == TOKEN_MUL:
            result, error = left.multed_by(right)
        elif node.op_tok.type == TOKEN_DIV:
            result, error = left.dived_by(right)
        elif node.op_tok.type == TOKEN_POW:
            result, error = left.powed_by(right)
        elif node.op_tok.type == TOKEN_EE:
            result, error = left.get_comparison_eq(right)
        elif node.op_tok.type == TOKEN_NE:
            result, error = left.get_comparison_ne(right)
        elif node.op_tok.type == TOKEN_LT:
            result, error = left.get_comparison_lt(right)
        elif node.op_tok.type == TOKEN_GT:
            result, error = left.get_comparison_gt(right)
        elif node.op_tok.type == TOKEN_LTE:
            result, error = left.get_comparison_lte(right)
        elif node.op_tok.type == TOKEN_GTE:
            result, error = left.get_comparison_gte(right)
        elif node.op_tok.matches(TOKEN_KEYWORK, 'AND'):
            result, error = left.anded_by(right)
        elif node.op_tok.matches(TOKEN_KEYWORK, 'OR'):
            result, error = left.ored_by(right)

        print(f'Result {result}')
        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))
    
    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error : return res

        error = None

        if node.op_tok.type == TOKEN_MINUS:
            number,error = number.multed_by(Number(-1))
        elif node.op_tok.type.matches(TOKEN_KEYWORK, 'NOT'):
            number, error = number.notted()

        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))

    def visit_IfNode(self, node, context):
        res = RTResult()

        for condition, expr in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.error: return res

            if condition_value.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.error: return res
                return res.success(expr_value)

        if node.else_case:
            else_value = res.register(self.visit(node.else_case, context))
            if res.error: return res
            return res.success(else_value)

        return res.success(None)

    def visit_ForNode(self, node, context):
        res = RTResult()

        start_value = res.register(self.visit(node.start_value_node, context))
        if res.error: return res

        end_value = res.register(self.visit(node.end_value_node, context))
        if res.error: return res

        if node.step_value_node:
            step_value = res.register(self.visit(node.step_value_node, context))
            if res.error: return res
        else:
            step_value = Number(1)

        i = start_value.value

        if step_value.value >= 0:
            condition = lambda: i < end_value.value
        else:
            condition = lambda: i > end_value.value
        
        while condition():
            context.symbol_table.set(node.var_name_tok.value, Number(i))
            i += step_value.value

            res.register(self.visit(node.body_node, context))
            if res.error: return res

        return res.success(None)

    def visit_WhileNode(self, node, context):
        res = RTResult()

        while True:
            condition = res.register(self.visit(node.condition_node, context))
            if res.error: return res

            if not condition.is_true(): break

            res.register(self.visit(node.body_node, context))
            if res.error: return res

        return res.success(None)


global_symbol_table = SymbolTable()
global_symbol_table.set("NULL", Number(0))
global_symbol_table.set("TRUE", Number(1))
global_symbol_table.set("FALSE", Number(0))

def run(text, fn):
    lexer = Lexer(text, fn)
    tokens, error = lexer.make_tokens()
    print(f'Tokens: {tokens}')
    if error: return None, error

    parser = Parser(tokens)
    ast = parser.parse()
    # print(f'Node: {ast.node}')
    if ast.error: return None, ast.error

    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)

    return result.value, result.error