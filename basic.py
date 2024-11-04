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
        result += f'File {self.pos_start.fn}, line {self.pos_end.line + 1}'
        return result

class IllegalCharError(Error):
    def __init__(self, details, pos_start, pos_end):
        super().__init__(pos_start, pos_end, "Illegal character error", details)

# position

class Position:
    def __init__(self, index, line, column, fn, ftxt ):
        self.index = index
        self.line = line
        self.column = column
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char):
        self.index += 1
        self.column += 1

        if current_char == '\n':
            self.line += 1
            self.column = 0
        
        return self
        
    def copy(self):
        return Position(self.index, self.line, self.column, self.fn, self.ftxt)

# Tokens

TOKEN_INT = 'TOKEN_INT'
TOKEN_FLOAT = 'TOKEN_FLOAT'
TOKEN_PLUS = 'TOKEN_PLUS'
TOKEN_MINUS = 'TOKEN_MINUS'
TOKEN_MUL = 'TOKEN_MUL'
TOKEN_DIV = 'TOKEN_DIV'
TOKEN_LPAREN = 'TOKEN_LPAREN'
TOKEN_RPAREN = 'TOKEN_RPAREN'

class Token:
    def __init__(self, type, value=None):
        self.type = type
        self.value = value

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

    def make_tokens(self):
        tokens = []
        while self.current_char != None:
            if self.current_char in ' \t':
                self.advance()
            if self.current_char.isdigit():
                tokens.append(self.make_numbers())
            elif self.current_char == '+':
                tokens.append(Token(TOKEN_PLUS))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TOKEN_MINUS))
                self.advance()
            elif self.current_char == '*':
                tokens.append(Token(TOKEN_MUL))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TOKEN_DIV))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TOKEN_LPAREN))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TOKEN_RPAREN))
                self.advance()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError("'" + char + "'", pos_start, self.pos)
            
        return tokens, None

    def make_numbers(self):
        num_str = ''
        dot_count = 0

        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TOKEN_INT, int(num_str))
        else:
            return Token(TOKEN_FLOAT, float(num_str))
            

def run(text, fn):
    lexer = Lexer(text, fn)
    tokens, error = lexer.make_tokens()
    return tokens, error