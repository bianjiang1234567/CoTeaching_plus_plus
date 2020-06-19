import matplotlib.pyplot as plt
import sys

def main():
    filename = sys.argv[1]
    #filename = 'mnist_coteaching_symmetric_0.2.txt'
    Epoch, test_acc, pure_ratio = [], [], []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            Epoch.append(int(line[0][:-1]))
            test_acc.append((float(line[9])+float(line[12]))/2)
            pure_ratio.append((float(line[15])+float(line[18]))/2)

    figure1 = plt.figure()
    plt.plot(Epoch, test_acc)
    plt.xlabel(u'epoch')
    plt.ylabel(u'test_acc')
    plt.title('test_acc of '+ filename)
    plt.savefig(fname='test_acc of ' + filename + '.jpg')

    figure2 = plt.figure()
    plt.plot(Epoch, pure_ratio)
    plt.xlabel(u'epoch')
    plt.ylabel(u'pure_ratio')
    plt.title('pure_ratio of '+ filename)
    plt.savefig(fname='pure_ratio of '+ filename + '.jpg')
    plt.show()

if __name__ == "__main__":
   main()