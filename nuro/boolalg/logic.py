# Copyright 2023 The AI Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=too-many-function-args, invalid-name, missing-module-docstring
# pylint: disable=missing-class-docstring

from typing import Tuple, Dict


class _Sentence():
  """An abstract model used for implementing symbols and logical connectives
  used in Propositional Logic.

  Note:

    See `Symbol` for more details.
  """
  def evaluate(self, model: dict[str, bool]) -> bool:
    """Evaluates the value of a symbol in a propositional logic (PL)."""
    raise NotImplementedError("nothing to evaluate")

  def formula(self) -> str:
    """Returns string formula representing logical sentence."""
    return ""

  def symbols(self) -> str:
    """Returns a set of all symbols in the logical sentence."""
    return set()

  @classmethod
  def validate(cls, sentence: "_Sentence"):
    """Validates if the given sentence is actually a `Sentence` instance.

    Raises:
      TypeError - In case of a non-logical sentence.
    """
    if not isinstance(sentence, _Sentence):
      raise TypeError("must be a logical sentence")

  @classmethod
  def parenthesize(cls, s: str) -> str:
    """Parenthesizes an expression if not already parenthesized.

    Args:
      s: Expression to parenthesize.

    Returns:
      Parenthesized expression.
    """
    def balanced(s: str) -> bool:
      """Checks if a string has balanced parentheses.

      Args:
        s: Expression to check for balanced parentheses.

      Returns:
        `True` if expression contains balanced parentheses, `False` otherwise.
      """
      count = 0
      for c in s:
        if c == "(":
          count += 1
        elif c == ")":
          if count <= 0:
            return False
          count -= 1
      return count == 0

    # Empty or alpha strings are always considered to be balanced.
    if not s or s.isalpha(
    ) or (s[0] == "(" and s[-1] == ")" and balanced(s[1:-1])):
      return s
    else:
      return f"({s})"


class Symbol(_Sentence):
  """`Symbol` class creates a symbol for a propositional logic (PL).

  A proposition is a declarative statement which is either true or false. The
  symbols used in propositional logic determines whether the logic holds `true`
  or turns out to be `false`.

  Example:

      >>> from logic import (Symbol, And)
      >>> P = Symbol('P')
      >>> Q = Symbol('Q')
      >>> knowledge = And(P, Q)
      >>> knowledge.formula()
      'P ∧ Q'

  """
  def __init__(self, name: str):
    """Constructs a `Symbol` instance.

    A symbol in propositional logic (PL) either holds a truth value or a false
    value, based on the combination of logical operations these values are then
    transformed to a result.

    Note:

      You can use letters, words, digits, or even special characters to contruct
      symbols as long as they hold some meaning for the viewer.

    Args:
      name: `Symbol` name; can be anything including digits, words, letters, or
        special characters.

    Example:

        >>> P = Symbol('P')
        >>> Q = Symbol('Q')
        >>> R = Symbol('R')

    """
    self._name = name

  def __eq__(self, other: "Symbol") -> bool:
    """Compares `self` with the other."""
    return isinstance(other, Symbol) and self._name == other._name

  def __hash__(self) -> int:
    """Returns the hash of the current `Symbol`."""
    return hash(("symbol", self._name))

  def __repr__(self) -> str:
    """Returns the name of the symbol."""
    return self._name

  def evaluate(self, model: Dict[str, bool]) -> bool:
    """Evaluates the value of a symbol in a propositional logic (PL).

    Evaluating a model means evaluating the value of each symbol in the
    propositional logic (PL) which either holds `True` or `False`.

    For example, for a symbol $P \\implies \\mathrm{True}$ in a
    propositional logic (PL) —

    $$¬P \\implies \\mathrm{False}$$

    Args:
      model: A propositional logic model mapping from the symbol name to its
        truth or false value.

    Returns:
      The evaluated model of the symbol.

    Example:

        >>> from ai import Symbol, Not
        >>> P = Symbol('P')
        >>> knowledge = Not(P)
        >>> model_true[str(P)] = True
        >>> knowledge.evaluate(model_true)
        False

    """
    try:
      return bool(model[self._name])
    except KeyError as exc:
      raise ValueError(f"variable {self._name} not in model") from exc

  def formula(self) -> str:
    """Returns the name of the symbol to be used in the formula.

    This function is later used to combine the operators with the operands i.e.,
    the symbols which are a `Symbol` instance. This function returns the string
    representation of the symbols which is then later joined with the operators
    to complete the propositional logic (PL).

    Returns:
      String representation of the current symbol.

    Example:

        >>> P = Symbol('P')
        >>> P.formula()
        'P'

    """
    return self._name

  def symbols(self) -> str:
    """Returns a set containing the name of the symbol."""
    return {self._name}


class Not(_Sentence):
  """`Not` class implements the properties and behaviour of the `(¬)` symbol.

  In a propositional logic (PL) the `NOT` symbol inverts the value of the
  symbol it is used with. For example, if the value of the symbol is `true` then
  it will be evaluated to `false` and vice-versa.

  Example:

      >>> from logic import (Not, Symbol)
      >>> rain = Symbol('Rain')
      >>> Not(rain).formula()
      '¬Rain'

  """
  def __init__(self, operand: "_Sentence"):
    """Constructs a `Not` instance.

    A `Symbol` instance is used while constructing a `Not` instance. This symbol
    will then be evaluated with a `(¬)` operator every time a propositional
    logic model is evaluated. If the value of the symbol holds `true` then the
    evaluated value of the entire expression will be `false` and vice-versa.

    Args:
      operand: An instance of a `Symbol` which is a operand in a propositional
        logic.

    Example:

        >>> from logic import (Not, Symbol)
        >>> P = Symbol('P')
        >>> knowledge = Not(P)
        >>> knowledge.formula()
        '¬P'

    """
    _Sentence.validate(operand)
    self._operand = operand

  def __eq__(self, other: "Not") -> bool:
    """Compares `self` with the `other`."""
    return isinstance(other, Not) and self._operand == other._operand

  def __hash__(self) -> int:
    """Returns the hash of the current `Not` expression."""
    return hash(("not", hash(self._operand)))

  def __repr__(self) -> str:
    """Returns a string representation of `self`."""
    return f"Not({self._operand})"

  def evaluate(self, model: Dict[str, bool]) -> bool:
    """Evaluates the value of the current expression i.e., `self` in the
    propositional logic (PL).

    Evaluating a model for `self` means evaluating the value of the current
    expression. For example, for a symbol $¬P \\implies \\mathrm{True}$ in
    a propositional logic (PL) —

    $$¬P \\implies \\mathrm{False}$$

    Args:
      model: A propositional logic model mapping from the symbol name to its
        truth or false value.

    Returns:
      The evaluated model of the current expression.

    Example:

        >>> from ai import Symbol, Not
        >>> P = Symbol('P')
        >>> knowledge = Not(P)
        >>> model_true = dict()
        >>> model_true[str(P)] = True
        >>> knowledge.evaluate(model_true)
        False

    """
    return not self._operand.evaluate(model)

  def formula(self) -> str:
    """Returns the expression for `self` that is to be used in the formula.

    This function returns a string representation of the `Symbol` with the `(¬)`
    operator that can later be joined with other operators and operands to
    complete the propositional logic (PL).

    Returns:
      String representation of the current symbol.

    Example:

        >>> P = Symbol('P')
        >>> knowledge(Not(P))
        >>> knowledge.formula()
        '¬P'

    """
    return "¬" + _Sentence.parenthesize(self._operand.formula())

  def symbols(self) -> set:
    """Returns a set containing the name of the symbols in the expression."""
    return self._operand.symbols()


class And(_Sentence):
  """`And` class implements the properties and behaviour of the `(∧)` symbol.

  In a propositional logic (PL) the `AND` symbol works similar to the (*)
  multiplication operator in algebra that multiplies multiple experssions into
  one single value. For example, if we have two symbols with value `true` and
  `false` (where `true` is equals to `1` and `false` is equals to `0`) then the
  `AND` operator between these two symbols will result in a new value `0`.

  Example:

      >>> from logic import (And, Symbol)
      >>> rain = Symbol('Rain')
      >>> run = Symbol('Run')
      >>> knowledge = And(rain, run)
      >>> knowledge.formula()
      'Rain ∧ Run'

  """
  def __init__(self, *conjuncts: Tuple["_Sentence", ...]):
    """Constructs a `And` instance.

    A `Symbol` instance is used while constructing a `And` instance as
    conjunctions. These symbol will then be evaluated with a `(∧)` operator
    every time a propositional logic model is evaluated. If the value of all the
    symbols holds `true` then the evaluated value of the entire expression will
    be `true` and if the value of any one symbol holds `false` in the
    conjunctions then the entire propositional logic expression evaluates to
    `false`.

    Args:
      conjunct: A tuple of conjunctions constructing a `And` expression in a
        propositional logic.

    Example:

        >>> from logic import (And, Symbol)
        >>> rain = Symbol('Rain')
        >>> run = Symbol('Run')
        >>> knowledge = And(rain, run)
        >>> knowledge.formula()
        'Rain ∧ Run'

    """
    for conjunct in conjuncts:
      _Sentence.validate(conjunct)
    self.conjuncts = list(conjuncts)

  def __eq__(self, other: "And") -> bool:
    """Compares `self` with the `other`."""
    return isinstance(other, And) and self.conjuncts == other.conjuncts

  def __hash__(self) -> int:
    """Returns the hash of the current `Not` expression."""
    return hash(("and", tuple(hash(conjunct) for conjunct in self.conjuncts)))

  def __repr__(self) -> str:
    """Returns a string representation of `self`."""
    conjunctions = ", ".join([str(conjunct) for conjunct in self.conjuncts])
    return f"And({conjunctions})"

  def add(self, conjunct: "_Sentence") -> None:
    """Appends a conjuction to the current experssion for `And`.

    Args:
      conjunct: A conjunction that constructs a `And` expression in a
        propositional logic.

    Example:

        >>> from ai.boolalg import (Symbol, And)
        >>> rain = Symbol('rain')
        >>> run = Symbol('run')
        >>> knowledge = And(rain, run)
        >>> knowledge.formula()
        'rain ∧ run'
        >>> umbrella = Symbol('umbrella')
        >>> knowledge.add(umbrella)
        >>> knowledge.formula()
        'rain ∧ run ∧ umbrella'

    """
    _Sentence.validate(conjunct)
    self.conjuncts.append(conjunct)

  def evaluate(self, model: Dict[str, bool]) -> bool:
    """Evaluates the value of the current expression i.e., `self` in the
    propositional logic (PL).

    Evaluating a model for `self` means evaluating the values of the
    conjunctions the current expression holds. For example if,
    $P \\implies \\mathrm{True}$ and $Q \\implies \\mathrm{false}$
    in a propositional logic (PL) —

    $$P ∧ Q \\implies \\mathrm{false}$$

    Args:
      mode: A propositional logic model mapping from the symbol name to its
        truth or false value.

    Returns:
      The evaluated model of the current expression.

    Example:

        >>> from ai.boolalg import (Symbol, And)
        >>> rain = Symbol('rain')
        >>> run = Symbol('run')
        >>> umbrella = Symbol('umbrella')
        >>> knowledge = And(rain, run, umbrella)
        >>> knowledge.formula()
        'rain ∧ run ∧ umbrella'
        >>> model = dict()
        >>> model[str(rain)] = True
        >>> model[str(umbrella)] = True
        >>> model[str(run)] = False
        >>> knowledge.evaluate(model)
        False

    """
    return all(conjunct.evaluate(model) for conjunct in self.conjuncts)

  def formula(self) -> str:
    """Returns the expression for `self` that is to be used in the formula.

    This function returns a string representation of the conjunctions with the
    `(∧)` operator that can later be joined with other operators and operands
    to complete the propositional logic (PL).

    Returns:
      String representation of the current symbol.

    Example:

        >>> from ai.boolalg import Symbol, And
        >>> rain = Symbol('rain')
        >>> run = Symbol('run')
        >>> knowledge = And(rain, run)
        >>> knowledge.formula()
        'rain ∧ run'

    """
    if len(self.conjuncts) == 1:
      return self.conjuncts[0].formula()
    return " ∧ ".join(
      [
        _Sentence.parenthesize(conjunct.formula())
        for conjunct in self.conjuncts
      ]
    )

  def symbols(self) -> set:
    """Returns a set containing the name of the symbols in the expression."""
    return set.union(*[conjunct.symbols() for conjunct in self.conjuncts])


class Or(_Sentence):
  """`Or` class implements the properties and behaviour of the `(∨)` symbol.

  In a propositional logic (PL) the `OR` symbol works similar to the (+)
  addition operator in algebra that adds multiple experssions into one single
  value. For example, if we have two symbols with value `true` and `false`
  (where `true` is equals to `1` and `false` is equals to `0`) then the `OR`
  operator between these two symbols will result in a new value `1`.

  Note:

    Unlike algebra, boolean algebra adds multiple `true` values into one single
    `true` value which is equals to `1`;
    $\\mathrm{true} + \\mathrm{true} \\implies 1$

  Example:

      >>> from logic import (Or, Symbol)
      >>> rain = Symbol('Rain')
      >>> run = Symbol('Run')
      >>> knowledge = Or(rain, run)
      >>> knowledge.formula()
      'Rain ∨ Run'

  """
  def __init__(self, *disjuncts: "_Sentence"):
    """Constructs a `Or` instance.

    A `Symbol` instance is used while constructing a `Or` instance as
    disjunctions. These symbol will then be evaluated with a `(∨)` operator
    every time a propositional logic model is evaluated. If the value of all the
    symbols holds `true` then the evaluated value of the entire expression will
    be `true` and if the value of any all the symbols holds `false` in the
    disjunctions then the entire propositional logic expression evaluates to
    `false`, unless a `true` symbol sneaks its way into the `Or` expression.

    Args:
      disjuncts: A tuple of disjunctions constructing a `Or` expression in a
        propositional logic.

    Example:

        >>> from logic import (And, Symbol)
        >>> rain = Symbol('Rain')
        >>> run = Symbol('Run')
        >>> knowledge = And(rain, run)
        >>> knowledge.formula()
        'Rain ∨ Run'

    """
    for disjunct in disjuncts:
      _Sentence.validate(disjunct)
    self.disjuncts = list(disjuncts)

  def __eq__(self, other):
    """Compares `self` with the `other`."""
    return isinstance(other, Or) and self.disjuncts == other.disjuncts

  def __hash__(self):
    """Returns the hash of the current `Not` expression."""
    return hash(("or", tuple(hash(disjunct) for disjunct in self.disjuncts)))

  def __repr__(self):
    """Returns a string representation of `self`."""
    disjuncts = ", ".join([str(disjunct) for disjunct in self.disjuncts])
    return f"Or({disjuncts})"

  def evaluate(self, model):
    """Evaluates the value of the current expression i.e., `self` in the
    propositional logic (PL).

    Evaluating a model for `self` means evaluating the values of the
    disjunctions the current expression holds. For example if,
    $P \\implies \\mathrm{True}$ and $Q \\implies \\mathrm{false}$
    in a propositional logic (PL) —

    $$P ∨ Q \\implies \\mathrm{true}$$

    Args:
      mode: A propositional logic model mapping from the symbol name to its
        truth or false value.

    Returns:
      The evaluated model of the current expression.

    Example:

        >>> from ai.boolalg import (Symbol, Or)
        >>> rain = Symbol('rain')
        >>> run = Symbol('run')
        >>> umbrella = Symbol('umbrella')
        >>> knowledge = Or(rain, run, umbrella)
        >>> knowledge.formula()
        'rain ∨ run ∨ umbrella'
        >>> model = dict()
        >>> model[str(rain)] = True
        >>> model[str(umbrella)] = True
        >>> model[str(run)] = False
        >>> knowledge.evaluate(model)
        True

    """
    return any(disjunct.evaluate(model) for disjunct in self.disjuncts)

  def formula(self):
    """Returns the expression for `self` that is to be used in the formula.

    This function returns a string representation of the disjunctions with the
    `(∨)` operator that can later be joined with other operators and operands
    to complete the propositional logic (PL).

    Returns:
      String representation of the current symbol.

    Example:

        >>> from ai.boolalg import Symbol, Or
        >>> rain = Symbol('rain')
        >>> run = Symbol('run')
        >>> knowledge = Or(rain, run)
        >>> knowledge.formula()
        'rain ∨ run'

    """
    if len(self.disjuncts) == 1:
      return self.disjuncts[0].formula()
    return " ∨ ".join(
      [
        _Sentence.parenthesize(disjunct.formula())
        for disjunct in self.disjuncts
      ]
    )

  def symbols(self):
    """Returns a set containing the name of the symbols in the expression."""
    return set.union(*[disjunct.symbols() for disjunct in self.disjuncts])


class Implication(_Sentence):
  def __init__(self, antecedent, consequent):
    _Sentence.validate(antecedent)
    _Sentence.validate(consequent)
    self.antecedent = antecedent
    self.consequent = consequent

  def __eq__(self, other):
    """Compares `self` with the `other`."""
    return (
      isinstance(other, Implication) and
      self.antecedent == other.antecedent and
      self.consequent == other.consequent
    )

  def __hash__(self):
    """Returns the hash of the current `Not` expression."""
    return hash(("implies", hash(self.antecedent), hash(self.consequent)))

  def __repr__(self):
    """Returns a string representation of `self`."""
    return f"Implication({self.antecedent}, {self.consequent})"

  def evaluate(self, model):
    return (
      (not self.antecedent.evaluate(model)) or self.consequent.evaluate(model)
    )

  def formula(self):
    antecedent = _Sentence.parenthesize(self.antecedent.formula())
    consequent = _Sentence.parenthesize(self.consequent.formula())
    return f"{antecedent} => {consequent}"

  def symbols(self):
    """Returns a set containing the name of the symbols in the expression."""
    return set.union(self.antecedent.symbols(), self.consequent.symbols())


class Biconditional(_Sentence):
  def __init__(self, left, right):
    _Sentence.validate(left)
    _Sentence.validate(right)
    self.left = left
    self.right = right

  def __eq__(self, other):
    """Compares `self` with the `other`."""
    return (
      isinstance(other, Biconditional) and self.left == other.left and
      self.right == other.right
    )

  def __hash__(self):
    """Returns the hash of the current `Not` expression."""
    return hash(("biconditional", hash(self.left), hash(self.right)))

  def __repr__(self):
    """Returns a string representation of `self`."""
    return f"Biconditional({self.left}, {self.right})"

  def evaluate(self, model):
    return (
      (self.left.evaluate(model) and self.right.evaluate(model)) or
      (not self.left.evaluate(model) and not self.right.evaluate(model))
    )

  def formula(self):
    left = _Sentence.parenthesize(str(self.left))
    right = _Sentence.parenthesize(str(self.right))
    return f"{left} <=> {right}"

  def symbols(self):
    """Returns a set containing the name of the symbols in the expression."""
    return set.union(self.left.symbols(), self.right.symbols())


def model_check(knowledge, query):
  """Checks if knowledge base entails query."""
  def check_all(knowledge, query, symbols, model):
    """Checks if knowledge base entails query, given a particular model."""

    # If model has an assignment for each symbol
    if not symbols:

      # If knowledge base is true in model, then query must also be true
      if knowledge.evaluate(model):
        return query.evaluate(model)
      return True
    else:

      # Choose one of the remaining unused symbols
      remaining = symbols.copy()
      p = remaining.pop()

      # Create a model where the symbol is true
      model_true = model.copy()
      model_true[p] = True

      # Create a model where the symbol is false
      model_false = model.copy()
      model_false[p] = False

      # Ensure entailment holds in both models
      return (
        check_all(knowledge, query, remaining, model_true) and
        check_all(knowledge, query, remaining, model_false)
      )

  # Get all symbols in both knowledge and query
  symbols = set.union(knowledge.symbols(), query.symbols())

  # Check that knowledge entails query
  return check_all(knowledge, query, symbols, {})
