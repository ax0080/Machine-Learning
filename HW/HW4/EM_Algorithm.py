import numpy as np
import os

def readmnist(mnist_dir, mode='training'):
    if mode == 'training':
        image_dir = os.path.join(mnist_dir, 'train-images.idx3-ubyte')
        label_dir = os.path.join(mnist_dir, 'train-labels.idx1-ubyte')
    elif mode == 'testing':
        image_dir = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
        label_dir = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte')

    with open(image_dir, 'rb') as fimage:
        magic, num, row, col = np.fromfile(fimage, dtype=np.dtype('>i'), count=4)
        images = np.fromfile(fimage, dtype=np.dtype('>B'), count=-1)

    with open(label_dir, 'rb') as flabel:
        magic, num = np.fromfile(flabel, dtype=np.dtype('>i'), count=2)
        labels = np.fromfile(flabel, dtype=np.dtype('>B'), count=-1)

    pixels = row * col
    images = images.reshape(num, pixels)

    return num, images, labels, pixels

def ExpectationStep(p, pi, eta, X):
    for i in range(60000):
        for k in range(10):
            eta[i, k] = pi[k, 0]
            eta[i, k] *= np.prod(p[k, X[i, :] == 1])
            eta[i, k] *= np.prod(1 - p[k, X[i, :] == 0])

        normalize = np.sum(eta[i, :])
        if normalize != 0:
            eta[i, :] /= normalize
    return eta

def MaximizationStep(p, pi, eta, X):
    a1 = 1e-8
    a2 = 1e-8
    Nk = np.sum(eta, axis=0)

    for k in range(10):
        p[k, :] = (np.sum(eta[:, k, None] * X, axis=0) + a1) / (Nk[k] + a2 * X.shape[1])

    pi[:, 0] = (Nk + a1) / (np.sum(Nk) + a2 * 10)

    return p, pi

def show_imageination(p, count, diff):
    im = (p >= 0.5) * 1
    for c in range(10):
        print("clustering class {}:".format(c))
        for row in range(28):
            for col in range(28):
                print(im[c][row * 28 + col], end=' ')
            print(" ")
        print("")

    print("No. of Iteration: {}, Difference: {}".format(count, diff))
    print("-----------------------------------------\n")

def final_imagination(p, predict_gt_relation):
    im = (p >= 0.5) * 1
    for c in range(10):
        for k in range(10):
            if predict_gt_relation[k] == c:
                choose = k
        print("labeled class {}:".format(c))
        for row in range(28):
            for col in range(28):
                print(im[choose][row * 28 + col], end=' ')
            print(" ")
        print("")
    print('-----------------------------------------\n')

def make_prediction(X, p):
    predict_gt = np.zeros((10, 10))
    pdistribution = np.zeros(10)
    for i in range(60000):
        for k in range(10):
            pdistribution[k] = pi[k, 0]
            for d in range(784):
                if X[i][d] == 1:
                    pdistribution[k] *= p[k][d]
                else:
                    pdistribution[k] *= (1 - p[k][d])
        predict = np.argmax(pdistribution)
        predict_gt[predict, train_labels[i]] += 1
    return predict_gt

def remake_prediction(X, p, predict_gt_relation):
    predict_gt = np.zeros((10, 10))
    pdistribution = np.zeros(10)
    for i in range(60000):
        for k in range(10):
            pdistribution[k] = pi[k, 0]
            for d in range(784):
                if X[i][d] == 1:
                    pdistribution[k] *= p[k][d]
                else:
                    pdistribution[k] *= (1 - p[k][d])
        pred = np.argmax(pdistribution)
        predict_gt[predict_gt_relation[pred], train_labels[i]] += 1
    return predict_gt

def shift_cluster(predict_gt):
    predict_gt_relation = np.full((10), -1, dtype=int)
    for k1 in range(10):
        ind = np.unravel_index(np.argmax(predict_gt, axis=None), (10, 10))
        predict_gt_relation[ind[0]] = ind[1]
        for k2 in range(10):
            predict_gt[ind[0]][k2] = -1
            predict_gt[k2][ind[1]] = -1
    return predict_gt_relation

def calculate_confusion(k, predict_gt):
    tp = fn = fp = tn = 0
    for pred in range(10):
        for tar in range(10):
            if pred == k and tar == k:
                tp += predict_gt[pred, tar]
            elif pred == k:
                fp += predict_gt[pred, tar]
            elif tar == k:
                fn += predict_gt[pred, tar]
            else:
                tn += predict_gt[pred, tar]
    return int(tp), int(fn), int(fp), int(tn)

def confusion(predict_gt, count):
    hit = 60000
    for k in range(10):
        tp, fn, fp, tn = calculate_confusion(k, predict_gt)
        hit -= tp
        print('Confusion Matrix {}:'.format(k))
        print('{:^20}{:^25}{:^25}'.format(' ', 'Predict number %d' % k, 'Predict not number %d' % k))
        print('{:^20}{:^25}{:^25}'.format('Is number %d' % k, tp, fn))
        print('{:^20}{:^25}{:^25}\n'.format('Isn\'t number %d' % k, fp, tn))
        print('Sensitivity (Successfully predict number {}):     {}'.format(k, tp / (tp + fn)))
        print('Specificity (Successfully predict not number {}): {}'.format(k, tn / (fp + tn)))
    print('Total iteration to converge:', count)
    print('Total error rate:', hit / 60000)

if __name__ == '__main__':
    mnist_dir = ''
    train_num, train_images, train_labels, num_pixels = readmnist(mnist_dir, 'training')
    N = train_num
    D = num_pixels
    K = 10
    X = np.zeros(train_images.shape, dtype=int)
    for r in range(60000):
        for c in range(784):
            if train_images[r, c] >= 128:
                X[r, c] = 1

    eta = np.zeros((train_num, K))
    p = np.random.uniform(0.0, 1.0, (10, 784))
    for k in range(K):
        tmp = np.sum(p[k, :])
        p[k, :] /= tmp

    pi = np.full((10, 1), 0.1)

    count = 0
    while True:
        p_old = p
        count += 1
        eta = ExpectationStep(p, pi, eta, X)
        p, pi = MaximizationStep(p, pi, eta, X)
        show_imageination(p, count, np.linalg.norm(p - p_old))
        if count == 20 and np.linalg.norm(p - p_old) < 1e-10:
            break

    predict_gt = make_prediction(X, p)
    predict_gt_relation = shift_cluster(predict_gt)
    predict_gt = remake_prediction(X, p, predict_gt_relation)
    final_imagination(p, predict_gt_relation)
    confusion(predict_gt, count)
