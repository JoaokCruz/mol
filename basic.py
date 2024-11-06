from strings_with_arrows import *

# DIGITS

DIGITS = '0123456789'

# ERRORS

class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f'{self.error_name}: {self.details}\n'
        result += f'File {self.pos_start.fn}, line {self.pos_start.line + 1}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

class IllegalCharError(Error):
    def __init__(self, details, pos_start, pos_end):
        super().__init__(pos_start, pos_end, "Illegal character error", details)

class InvalidSyntaxError(Error):
    def __init__(self, details, pos_start, pos_end):
        super().__init__(pos_start, pos_end, "Invalid Syntax", details)

class RTError(Error):
    def __init__(self, details, pos_start, pos_end, context):
        super().__init__(pos_start, pos_end, "Runtime Error", details)
        self.context = context

    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.error_name}: {self.details}\n'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context

        while ctx:
            result = f'  File {pos.fn}, line {str(pos.line + 1)}, in {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent

        return 'Traceback (most recent call last):\n' + result
    

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
TOKEN_MINUS = 'MINUS'
TOKEN_MUL = 'MUL'
TOKEN_DIV = 'DIV'
TOKEN_LPAREN = 'LPAREN'
TOKEN_RPAREN = 'RPAREN'
TOKEN_EOF = 'EOF'


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
            elif self.current_char == '(':
                tokens.append(Token(TOKEN_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TOKEN_RPAREN, pos_start=self.pos))
                self.advance()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError("'" + char + "'", pos_start, self.pos)
        
        tokens.append(Token(TOKEN_EOF, pos_start=self.pos))
        return tokens, None

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
            

class NumberNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    
    def __repr__(self):
        return f'{self.tok}'


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

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error: self.error = res.error
            return res.node

        return res
        
    def success(self, node):
        self.node = node
        return self
    
    def failure(self, error):
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

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TOKEN_PLUS, TOKEN_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok,factor))

        elif tok.type in (TOKEN_INT, TOKEN_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))

        elif tok.type == TOKEN_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.type == TOKEN_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    "Expected ')'",
                    self.current_tok.pos_start,
                    self.current_tok.pos_end
                ))
        
        return res.failure(InvalidSyntaxError(
            "Expected int or float",
            tok.pos_start, 
            tok.pos_end,
        ))

    def term(self):
        return self.bin_op(self.factor, (TOKEN_MUL, TOKEN_DIV))

    def expr(self):
        return self.bin_op(self.term, (TOKEN_PLUS, TOKEN_MINUS))

    def bin_op(self, func, ops):
        res = ParseResult()
        left = res.register(func())

        if res.error: return res

        while self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func())
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
    
    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
   
    def subbed_to(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
    
    def multed_to(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
    
    def dived_to(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    "Division by zero",
                    other.pos_start,
                    other.pos_end,
                    self.context
                )
            
            return Number(self.value / other.value).set_context(self.context), None
        
    def __repr__(self):
        return str(self.value)

class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos


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
    
    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error : return res
        right = res.register(self.visit(node.right_node, context))
        if res.error : return res

        if node.op_tok.type == TOKEN_PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.type == TOKEN_MINUS:
            result, error = left.subbed_to(right)
        elif node.op_tok.type == TOKEN_MUL:
            result, error = left.multed_to(right)
        elif node.op_tok.type == TOKEN_DIV:
            result, error = left.dived_to(right)

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
            number,error = number.multed_to(Number(-1))

        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))


def run(text, fn):
    lexer = Lexer(text, fn)
    tokens, error = lexer.make_tokens()
    if error: return None, error

    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    interpreter = Interpreter()
    context = Context('<program>')
    result = interpreter.visit(ast.node, context)

    return result.value, result.error