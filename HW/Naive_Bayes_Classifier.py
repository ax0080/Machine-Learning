import numpy as np

def load(x_file_path, y_file_path):

    x_file = open(x_file_path,'rb')
    y_file = open(y_file_path,'rb')
    x_file.read(4)
    num_imgs = int.from_bytes(x_file.read(4), byteorder='big')
    num_rows = int.from_bytes(x_file.read(4), byteorder='big')
    num_cols = int.from_bytes(x_file.read(4), byteorder='big')
    
    y_file.read( 8 )
    x = np.zeros((num_imgs, num_rows*num_cols), dtype='uint8')
    y = np.zeros(num_imgs, dtype='uint8')
    for i in range(num_imgs):
        for j in range(num_rows*num_cols):
            x[i, j] = int.from_bytes(x_file.read(1), byteorder='big')
        y[i] = int.from_bytes(y_file.read(1), byteorder='big')

    return x, y

class Count:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.num_cls = 10
    
    # prior count
    def prior(self, cls):
        return self.train_y[self.train_y == cls].shape[0]  / self.train_y.shape[0]
    
    def test(self, likelihood):
        error = 0
        _, num_pix, bins = likelihood.shape
        
        for image_idx, label in enumerate(self.test_y):
            posterior = np.zeros(self.num_cls)
            for cls in range(self.num_cls):
                for pixel_idx in range(num_pix):
                    bin_loc = int(self.test_x[image_idx, pixel_idx])//(256//bins)
                    posterior[cls] += np.log(max(1e-10, likelihood[cls, pixel_idx, bin_loc]))
                posterior[cls] += np.log(self.prior(cls))
            posterior /= np.sum(posterior)

            print('Posterior (in log scale):')
            for cls in range(self.num_cls):
                print(f'{cls}: {posterior[cls]}')
            
            prediction = np.argmin(posterior)
            print(f'Prediction: {prediction}, Ans: {label}\n')
            
            if prediction != label:
                error += 1
        print('Error rate: {:.4f}'.format(error / len(self.test_y)))
        print()

    def print_number(self, likelihood, threshold):
        print('Imagination of numbers in Bayesian classifier:')
        for cls in range(self.num_cls):
            print(f'{cls}:')
            for idx, pix in enumerate(likelihood[cls]):
                endsign = ' ' if (idx+1)%28 else '\n'
                print('1' if np.argmax(pix)>=threshold else '0', end=endsign)
            print()
        print()

# count likelihood[class, pixel, bin]
class Discrete(Count):
    def get_probability(self):
        _, num_pix = self.train_x.shape
        probs = np.zeros((self.num_cls, num_pix, 256//8))
        for image_idx, cls in enumerate(self.train_y):
            for pixel_idx in range(num_pix):
                probs[cls, pixel_idx, int(self.train_x[image_idx, pixel_idx])//8] += 1

        for cls in range(self.num_cls):
            for pixel_idx in range(num_pix):
                probs[cls, pixel_idx, : ] /= np.sum(probs[cls, pixel_idx])
        return probs


class Continuous(Count):
    def __init__(self, train_x, train_y, test_x, test_y, var_def=10):
        super().__init__(train_x, train_y, test_x, test_y)
        self.var_def = var_def
        
    def get_probability(self):
        num_pix = self.train_x.shape[1]
        probs = np.zeros((self.num_cls, num_pix, 256))

        for cls in range(self.num_cls):
            img_cls = self.train_x[self.train_y == cls]
            for pixel_idx in range(num_pix):
                m = np.mean(img_cls[:, pixel_idx])
                var = np.var(img_cls[:, pixel_idx]) + self.var_def
                x = np.arange(256)
                probs[cls, pixel_idx] = (1/np.sqrt(2*np.pi*var))*np.exp((-(x-m)**2)/(2*var))
        return probs
    


train_x, train_y = load('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
test_x, test_y = load('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')


toggle_option = input('Toggle Option (0: discrete mode/1: continuous mode): ')

if toggle_option == '0':
    threshold = 16
    mode = Discrete(train_x, train_y, test_x, test_y)
            
elif toggle_option == '1':
    threshold = 128
    mode = Continuous(train_x, train_y, test_x, test_y)

L = mode.get_probability()
mode.test(L)
mode.print_number(L, threshold)