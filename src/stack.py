
class Stack:
    """A simple first in last out stack.

    Args:
        CFG_obj: an name of grammar
        start_symbol: an instance of nltk.Nonterminal that is the
            start symbol the grammar
    """
    def __init__(self, CFG_obj, start_symbol):
        self.CFG_obj = CFG_obj
        self._stack = [start_symbol]

    def pop(self):
        return self._stack.pop()

    def push(self, symbol):
        self._stack.append(symbol)

    def __str__(self):
        return str(self._stack)

    @property
    def nonempty(self):
        return bool(self._stack)


if __name__ == '__main__':
    import grammar
    from grammar import S

    stack = Stack(grammar=grammar.grammar_g3, start_symbol=S)
    print(stack.nonempty)
    

