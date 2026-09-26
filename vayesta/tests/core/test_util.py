import pytest
import unittest

import numpy as np
from vayesta.core.util import einsum, getattr_recursive, setattr_recursive

from vayesta.tests.common import TestCase


@pytest.mark.fast
class TestEinsum(TestCase):
    allclose_atol = 1e-12
    allclose_rtol = 1e-10

    def test_mmm(self):
        a = np.random.rand(3, 4)
        b = np.random.rand(4, 5)
        c = np.random.rand(5, 6)
        ops = (a, b, c)

        expected = np.einsum("ab,bc,cd->ad", *ops)

        res = einsum("ab,bc,cd->ad", *ops)
        self.assertAllclose(res, expected)

        res = einsum("ab^,b^c,cd2->ad2", *ops)
        self.assertAllclose(res, expected)

        res = einsum("ab,bc,cd", *ops)
        self.assertAllclose(res, expected)

        res = einsum("(ab,bc,cd->ad)", *ops)
        self.assertAllclose(res, expected)

        res = einsum("(ab,bc),cd->ad", *ops)
        self.assertAllclose(res, expected)

        res = einsum("(ab# ,b#c),cd->ad", *ops)
        self.assertAllclose(res, expected)

        res = einsum("(ab,bc),cd", *ops)
        self.assertAllclose(res, expected)

        res = einsum("[(ab,bc),cd]", *ops)
        self.assertAllclose(res, expected)

        res = einsum("(ab,bc->ac),cd", *ops)
        self.assertAllclose(res, expected)

        res = einsum("(ab,bc->ac),cd->ad", *ops)
        self.assertAllclose(res, expected)

        res = einsum("ab,(bc,cd)->ad", *ops)
        self.assertAllclose(res, expected)

        res = einsum("ab,(bc,cd->bd)->ad", *ops)
        self.assertAllclose(res, expected)

        res = einsum("ab,(bc,cd)->ad", *ops)
        self.assertAllclose(res, expected)

    def test_mmmm(self):
        a = np.random.rand(3, 4)
        b = np.random.rand(4, 5)
        c = np.random.rand(5, 6)
        d = np.random.rand(6, 7)
        ops = (a, b, c, d)

        expected = np.einsum("ab,bc,cd,de->ae", *ops)

        res = einsum("ab,bc,cd,de->ae", *ops)
        self.assertAllclose(res, expected)

        res = einsum("ab,bc,cd,de", *ops)
        self.assertAllclose(res, expected)

        res = einsum("(ab,bc,cd,de->ae)", *ops)
        self.assertAllclose(res, expected)

        res = einsum("(ab,bc),(cd,de)->ae", *ops)
        self.assertAllclose(res, expected)

        res = einsum("[ab,bc],{cd,de}->ae", *ops)
        self.assertAllclose(res, expected)

        res = einsum("(ab,bc->ac),(cd,de)->ae", *ops)
        self.assertAllclose(res, expected)

        res = einsum("(ab,bc->ac),(cd,de)", *ops)
        self.assertAllclose(res, expected)

        res = einsum("(...b,bc->...c),(cd,de->ce)->...e", *ops)
        self.assertAllclose(res, expected)

        res = einsum("(ab,bc->ac),(c...,...e->ce)->ae", *ops)
        self.assertAllclose(res, expected)

        res = einsum("[(ab,bc),cd],de->ae", *ops)
        self.assertAllclose(res, expected)

    def test_mtm(self):
        a = np.random.rand(3, 4)
        b = np.random.rand(4, 5, 6)
        c = np.random.rand(6, 7)
        ops = (a, b, c)

        expected = np.einsum("ab,bcd,de->ace", *ops)

        res = einsum("ab,bcd,de->ace", *ops)
        self.assertAllclose(res, expected)

        res = einsum("(ab,bcd),de->ace", *ops)
        self.assertAllclose(res, expected)

        res = einsum("ab,(bcd,de)->ace", *ops)
        self.assertAllclose(res, expected)

    def test_mtm_2(self):
        a = np.random.rand(3, 4)
        b = np.random.rand(4, 5, 6)
        c = np.random.rand(6, 7)
        ops = (a, b, c)

        expected = np.einsum("ab,bcd,de->e", *ops)

        res = einsum("ab,bcd,de->e", *ops)
        self.assertAllclose(res, expected)

        res = einsum("(ab,bcd),de->e", *ops)
        self.assertAllclose(res, expected)

        res = einsum("ab,(bcd,de)->e", *ops)
        self.assertAllclose(res, expected)


@pytest.mark.fast
class TestAttrRecursive(TestCase):
    class Obj:
        pass

    def make_obj(self):
        obj = self.Obj()
        obj.a = self.Obj()
        obj.a.b = self.Obj()
        obj.a.b.c = 1
        return obj

    def test_getattr_recursive(self):
        obj = self.make_obj()
        self.assertEqual(getattr_recursive(obj, "a.b.c"), 1)
        self.assertIsNone(getattr_recursive(obj, "a.b.d", None))

    def test_setattr_recursive(self):
        obj = self.make_obj()
        setattr_recursive(obj, "a.b.c", 2)
        self.assertEqual(obj.a.b.c, 2)
        setattr_recursive(obj, "d", 3)
        self.assertEqual(obj.d, 3)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
