from enum import Enum, IntEnum


class HyperParam(Enum):

    @property
    def what(self):
        return self.value[1]

    @property
    def val(self):
        return self.value[0]

    def __call__(self, *args, **kwargs):
        return self.val

    def __eq__(self, other) -> bool:
        if isinstance(other, HyperParam) and self.val == other.val and self.what == other.what:
            return True
        return False

    def __add__(self, other) -> float:
        if isinstance(other, HyperParam):
            other = other.val
        return float(self.val + other.val)

    def __mul__(self, other) -> float:
        if isinstance(other, HyperParam):
            other = other.val
        return float(self.val * other.val)

    def __str__(self) -> str:
        return "{}".format(self.val)

    def __float__(self) -> float:
        return self.val

    def __int__(self) -> int:
        return int(self.val)

    def __hash__(self):
        return str.__hash__("{}:{}".format(self.val, self.what))

    def __abs__(self):
        return abs(self.val)

    def __cmp__(self, other):
        if isinstance(other, HyperParam):
            other = other.val
        return self.val.__cmp__(other)


class CartPoleHyperParam(HyperParam):
    gamma_1 = [1.0, "Descr"]
    gamma_2 = [2.0, "Descr"]
    gamma_3 = [1.0, "Descr"]


def func(x: float):
    print(x)
    print(type(x))
    d = dict()
    d[x] = "Hello"
    print(d[x])
    return


if __name__ == "__main__":
    x = CartPoleHyperParam.gamma_1
    print(x)
    func(x)
