import struct


def dataPack(matrix, data, value):
    r'''transform the (matrix, value) tuple into a bytes data
    usage example:
    with open(path, 'wb') as f:
        f.write(dataPack(matrix, value))
    '''
    # 将数组处理为二进制文件
    if len(matrix) != 25 or len(matrix[0]) != 25:
        raise ValueError('matrix must be 5x5')
    if len(data) != 5:
        raise ValueError('len(data) must be 5')
    matrix = [float(e) for each in matrix for e in each]
    return struct.pack('f' * 631, *matrix, *data, value)


def dataUnpack(data):
    r'''transform the data from bytes file into a tuple (matrix, value)'''
    # 使用struct将二进制文件处理为元组
    temp = struct.unpack('f' * 631, data)
    matrix, odata, value = temp[:625], temp[625:630], temp[630]
    # dBZ矩阵为索引0-624的元素，气象要素为625-630，降水量为631
    matrix = [[matrix[i * 25 + j] for j in range(25)] for i in range(25)]
    return matrix, odata, value


def filePack(f, dataset):
    r'''dataset is a list like: [bytes, bytes, bytes, ...]
    you can get bytes data from dataPack(matrix, value)'''
    for e in dataset:
        f.write(dataPack(*e))


def fileUnpack(f):
    data = []
    while True:
        bdata = f.read(631 * 4)  # 从文件中读取631*4个字节数据
        if bdata:
            data.append(dataUnpack(bdata))
        else:
            break
    return data


if __name__ == '__main__':
    pass
