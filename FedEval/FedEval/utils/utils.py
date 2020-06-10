import pickle
import codecs
import numpy as np


def obj_to_pickle_string(x, file_path=None):
    if file_path is not None:
        print("save model to file")
        output = open(file_path, 'wb')
        pickle.dump(x, output)
        return file_path
    else:
        print("turn model to byte")
        x = codecs.encode(pickle.dumps(x), "base64").decode()
        print(len(x))
        return x
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO


def pickle_string_to_obj(s):
    if ".pkl" in s:
        df = open(s, "rb")
        print("load model from file")
        return pickle.load(df)
    else:
        print("load model from byte")
        return pickle.loads(codecs.decode(s.encode(), "base64"))


def list_tree(value_lists):
    result = []
    counter = np.zeros(len(value_lists), dtype=np.int32)
    for i in range(int(np.prod([len(e) for e in value_lists]))):
        tmp = []
        for j in range(len(value_lists)):
            v = value_lists[j][counter[j]]
            if type(v) is list:
                tmp += v
            else:
                tmp.append(v)
        result.append(tmp)
        counter[-1] += 1
        for j in range(len(value_lists)-1, -1, -1):
            if counter[j] > (len(value_lists[j]) - 1):
                counter[j] = 0
                if j > 0:
                    counter[j-1] += 1
    return result