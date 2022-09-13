import unittest


class ArrayTest(unittest.TestCase):

    def test_pop(self):
        counter = []
        for i in range(10):
            counter.append(i)
            print(counter)
        self.assertEqual(len(counter), 10)

        for i in range(10):
            counter.append(counter.pop() + 1)
            print(counter)

        for i in range(1, 10):
            print(counter.pop(), counter)
