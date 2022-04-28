import sys
import math
import gmpy2
import logging
import fractions


class GaloisFieldParams:

    def __init__(self, p):
        self._p = p
        self._max_int = p // 3 - 1

    @property
    def p(self):
        return self._p

    @property
    def max_int(self):
        return self._max_int


class GaloisFieldNumber:

    BASE = 16
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig
    FULL_PRECISION = math.floor(-FLOAT_MANTISSA_BITS / LOG2_BASE)

    def __init__(self, encoding: int, exponent: int, gfp: GaloisFieldParams):
        self.encoding = encoding
        self.exponent = exponent
        self.gfp = gfp

    @classmethod
    def encode(cls, scalar, gfp: GaloisFieldParams, exponent=FULL_PRECISION):
        int_rep = round(scalar * cls.BASE ** -exponent)
        return GaloisFieldNumber(gmpy2.mod(int_rep, gfp.p), exponent=exponent, gfp=gfp)

    def decode(self):
        """Decode plaintext and return the result.
        Returns:
          an int or float: the decoded number. N.B. if the number
            returned is an integer, it will not be of type float.
        Raises:
          OverflowError: if overflow is detected in the decrypted number.
        """
        if self.encoding >= self.gfp.p:
            # Should be mod n
            raise ValueError('Attempted to decode corrupted number')
        elif self.encoding <= self.gfp.max_int:
            # Positive
            mantissa = int(self.encoding)
        elif self.encoding >= self.gfp.p - self.gfp.max_int:
            # Negative
            mantissa = int(self.encoding - self.gfp.p)
        else:
            raise OverflowError('Overflow detected in decrypted number')

        if self.exponent >= 0:
            # Integer multiplication. This is exact.
            return mantissa * self.BASE ** self.exponent
        else:
            # BASE ** -e is an integer, so below is a division of ints.
            # Not coercing mantissa to float prevents some overflows.
            try:
                return mantissa / self.BASE ** -self.exponent
            except OverflowError as e:
                raise OverflowError('decoded result too large for a float') from e

    def decrease_exponent_to(self, new_exp):
        if new_exp > self.exponent:
            raise ValueError('New exponent %i should be more negative than'
                             'old exponent %i' % (new_exp, self.exponent))
        if self.encoding > self.gfp.max_int:
            return GaloisFieldNumber.encode(self.decode(), exponent=new_exp, gfp=self.gfp)
        else:
            factor = pow(self.BASE, self.exponent - new_exp)
            new_enc = int(self.encoding * factor) % self.gfp.p
            return GaloisFieldNumber(encoding=new_enc, exponent=new_exp, gfp=self.gfp)

    def __add__(self, other):
        if type(other) in [int, float]:
            return self + GaloisFieldNumber.encode(other, gfp=self.gfp)
        elif type(other) is GaloisFieldNumber:
            if self.exponent == other.exponent:
                return GaloisFieldNumber(
                    gmpy2.mod(gmpy2.add(self.encoding, other.encoding), self.gfp.p),
                    exponent=self.exponent, gfp=self.gfp
                )
            elif self.exponent > other.exponent:
                return self.decrease_exponent_to(other.exponent) + other
            else:
                return other.decrease_exponent_to(self.exponent) + self
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if type(other) in [int, float]:
            return self + GaloisFieldNumber.encode(-other, gfp=self.gfp)
        elif type(other) is GaloisFieldNumber:
            return self + (-1) * other
        else:
            raise NotImplementedError

    def __rsub__(self, other):
        if type(other) in [int, float]:
            return GaloisFieldNumber.encode(other, gfp=self.gfp) + (-1) * self
        elif type(other) is GaloisFieldNumber:
            return other + (-1) * self
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if type(other) in [int, float]:
            return self * self.encode(other, gfp=self.gfp)
        elif type(other) is GaloisFieldNumber:
            if self.encoding < self.gfp.max_int and other.encoding < self.gfp.max_int:
                return GaloisFieldNumber(
                    int(self.encoding * other.encoding) % self.gfp.p, self.exponent + other.exponent,
                    gfp=self.gfp
                )
            elif self.encoding < self.gfp.max_int < other.encoding:
                encoding = int(-self.encoding * (self.gfp.p - other.encoding)) % self.gfp.p
                return GaloisFieldNumber(
                    encoding=encoding, exponent=self.exponent + other.exponent, gfp=self.gfp
                )
            elif self.encoding > self.gfp.max_int > other.encoding:
                encoding = int(-(self.gfp.p - self.encoding) * other.encoding) % self.gfp.p
                return GaloisFieldNumber(
                    encoding=encoding,
                    exponent=self.exponent + other.exponent, gfp=self.gfp
                )
            else:
                encoding = int((self.gfp.p - self.encoding) * (self.gfp.p - other.encoding)) % self.gfp.p
                return GaloisFieldNumber(
                    encoding=encoding,
                    exponent=self.exponent + other.exponent, gfp=self.gfp
                )
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if type(other) in [int, float]:
            return self / GaloisFieldNumber.encode(other, gfp=self.gfp)
        elif type(other) is GaloisFieldNumber:
            if (self.exponent - other.exponent) > self.FULL_PRECISION:
                return self.decrease_exponent_to(self.FULL_PRECISION + other.exponent) / other
            else:
                if self.encoding < self.gfp.max_int and other.encoding < self.gfp.max_int:
                    return GaloisFieldNumber(
                        int(self.encoding / other.encoding) % self.gfp.p, self.exponent - other.exponent,
                        gfp=self.gfp
                    )
                elif self.encoding < self.gfp.max_int < other.encoding:
                    encoding = int(-self.encoding / (self.gfp.p - other.encoding)) % self.gfp.p
                    return GaloisFieldNumber(
                        encoding=encoding,
                        exponent=self.exponent - other.exponent, gfp=self.gfp
                    )
                elif self.encoding > self.gfp.max_int > other.encoding:
                    encoding = int(-(self.gfp.p - self.encoding) / other.encoding) % self.gfp.p
                    return GaloisFieldNumber(
                        encoding=encoding,
                        exponent=self.exponent - other.exponent, gfp=self.gfp
                    )
                else:
                    encoding = int((self.gfp.p - self.encoding) / (self.gfp.p - other.encoding)) % self.gfp.p
                    return GaloisFieldNumber(
                        encoding=encoding,
                        exponent=self.exponent - other.exponent, gfp=self.gfp
                    )
        else:
            raise NotImplementedError

    def __rtruediv__(self, other):
        if type(other) in [int, float]:
            return GaloisFieldNumber.encode(other, gfp=self.gfp) / self
        elif type(other) is GaloisFieldNumber:
            return other / self
        else:
            raise NotImplementedError
