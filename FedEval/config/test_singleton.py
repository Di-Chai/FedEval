from typing import Any, Iterable
from unittest import TestCase

from .singleton import Singleton


class ChildOfSingleton(Singleton):
    pass


class GrandSonOfSingleton(ChildOfSingleton):
    pass


class SingletonTestCase(TestCase):
    def setUp(self) -> None:
        self.singleton = Singleton()

    def test_singleton_on_inheritance_tree(self):
        # make sure that all cls on the inheritance tree own its unique instance
        child = ChildOfSingleton()
        grandson = GrandSonOfSingleton()
        instances = [child, grandson, self.singleton]
        self.assertTrue(SingletonTestCase.no_equal_elements(instances))

    @staticmethod
    def no_equal_elements(elements: Iterable[Any]) -> bool:
        for e in elements:
            equal_elements = [x for x in elements if x == e]
            if len(equal_elements) > 1:
                return False
        return True

    def test_singleton_uniqueness(self):
        singleton = Singleton()
        self.assertIs(singleton, self.singleton)
        self.assertEqual(singleton, self.singleton)
