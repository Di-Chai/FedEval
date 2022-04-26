import fractions
import logging
import math
import sys


class GaloisFieldNumber:

    BASE = 16
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig
    FULL_PRECISION = math.floor(-FLOAT_MANTISSA_BITS / LOG2_BASE)

    def __init__(self, encoding, exponent, p=2**521-1, max_int=None):
        self.encoding = encoding
        self.exponent = exponent
        self._p = p
        self._max_int = max_int or self._get_max_int(self._p)

    @staticmethod
    def _get_max_int(p):
        return p // 3 - 1

    @classmethod
    def encode(cls, scalar, p=2 ** 521 - 1, exponent=FULL_PRECISION):
        # Use rationals instead of floats to avoid overflow.
        int_rep = round(fractions.Fraction(scalar) * fractions.Fraction(cls.BASE) ** -exponent)
        _max_int = cls._get_max_int(p)
        # if abs(int_rep) > _max_int:
        #     logging.warning('Integer needs to be within +/- %d but got %d' % (_max_int, int_rep))
        # Wrap negative numbers by adding n
        return GaloisFieldNumber(int_rep % p, exponent, max_int=_max_int)

    def decode(self):
        """Decode plaintext and return the result.
        Returns:
          an int or float: the decoded number. N.B. if the number
            returned is an integer, it will not be of type float.
        Raises:
          OverflowError: if overflow is detected in the decrypted number.
        """
        if self.encoding >= self._p:
            # Should be mod n
            raise ValueError('Attempted to decode corrupted number')
        elif self.encoding <= self._max_int:
            # Positive
            mantissa = self.encoding
        elif self.encoding >= self._p - self._max_int:
            # Negative
            mantissa = self.encoding - self._p
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
        if self.encoding > self._max_int:
            return GaloisFieldNumber.encode(self.decode(), exponent=new_exp)
        else:
            factor = pow(self.BASE, self.exponent - new_exp)
            new_enc = int(self.encoding * factor) % self._p
            return GaloisFieldNumber(encoding=new_enc, exponent=new_exp, max_int=self._max_int)

    def __add__(self, other):
        if type(other) in [int, float]:
            return self + GaloisFieldNumber.encode(other)
        elif type(other) is GaloisFieldNumber:
            if self.exponent == other.exponent:
                return GaloisFieldNumber(int(self.encoding + other.encoding) % self._p, self.exponent)
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
            return self + GaloisFieldNumber.encode(-other)
        elif type(other) is GaloisFieldNumber:
            return self + (-1) * other
        else:
            raise NotImplementedError

    def __rsub__(self, other):
        if type(other) in [int, float]:
            return GaloisFieldNumber.encode(other) + (-1) * self
        elif type(other) is GaloisFieldNumber:
            return other + (-1) * self
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if type(other) in [int, float]:
            return self * self.encode(other)
        elif type(other) is GaloisFieldNumber:
            if self.encoding < self._max_int and other.encoding < self._max_int:
                return GaloisFieldNumber(int(self.encoding * other.encoding) % self._p, self.exponent + other.exponent)
            elif self.encoding < self._max_int < other.encoding:
                encoding = int(-self.encoding * (self._p - other.encoding)) % self._p
                return GaloisFieldNumber(
                    encoding=encoding,
                    exponent=self.exponent + other.exponent
                )
            elif self.encoding > self._max_int > other.encoding:
                encoding = int(-(self._p - self.encoding) * other.encoding) % self._p
                return GaloisFieldNumber(
                    encoding=encoding,
                    exponent=self.exponent + other.exponent
                )
            else:
                encoding = int((self._p - self.encoding) * (self._p - other.encoding)) % self._p
                return GaloisFieldNumber(
                    encoding=encoding,
                    exponent=self.exponent + other.exponent
                )
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if type(other) in [int, float]:
            return self / GaloisFieldNumber.encode(other)
        elif type(other) is GaloisFieldNumber:
            if (self.exponent - other.exponent) > self.FULL_PRECISION:
                return self.decrease_exponent_to(self.FULL_PRECISION + other.exponent) / other
            else:
                if self.encoding < self._max_int and other.encoding < self._max_int:
                    return GaloisFieldNumber(int(self.encoding / other.encoding) % self._p, self.exponent - other.exponent)
                elif self.encoding < self._max_int < other.encoding:
                    encoding = int(-self.encoding / (self._p - other.encoding)) % self._p
                    return GaloisFieldNumber(
                        encoding=encoding,
                        exponent=self.exponent - other.exponent
                    )
                elif self.encoding > self._max_int > other.encoding:
                    encoding = int(-(self._p - self.encoding) / other.encoding) % self._p
                    return GaloisFieldNumber(
                        encoding=encoding,
                        exponent=self.exponent - other.exponent
                    )
                else:
                    encoding = int((self._p - self.encoding) / (self._p - other.encoding)) % self._p
                    return GaloisFieldNumber(
                        encoding=encoding,
                        exponent=self.exponent - other.exponent
                    )
        else:
            raise NotImplementedError

    def __rtruediv__(self, other):
        if type(other) in [int, float]:
            return GaloisFieldNumber.encode(other) / self
        elif type(other) is GaloisFieldNumber:
            return other / self
        else:
            raise NotImplementedError
