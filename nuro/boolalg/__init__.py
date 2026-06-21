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

"""Symbolic Propositional Logic for Knowledge representation.

Propositional logic (PL) is the simplest form of logic where all the statements
are made by propositions. A proposition is a declarative statement which is
either true or false. It is a technique of knowledge representation in logical
and mathematical form.

Example:

    ```python
    from ai import (Symbol, And, Or, Not, Implication)

    rain = Symbol("rain")
    hagrid = Symbol("hagrid")
    dumbledore = Symbol("dumbledore")

    knowledge = And(
    Implication(Not(rain), hagrid),
    Or(hagrid, dumbledore),
    Not(And(hagrid, dumbledore)),
    dumbledore
    )

    print(model_check(knowledge, rain))
    ```
"""

from .logic import Symbol
from .logic import Not
from .logic import And
from .logic import Or
from .logic import Implication
from .logic import Biconditional
from .logic import model_check
