import random
from sklearn import svm
from src.circle import load_circle
from src.experiment import run
from src.mnist import load_mnist
from matplotlib.pyplot import boxplot
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def jaspertest():
    NUMBER_OF_MNIST_SAMPLES = 400
    inputs, targets = load_mnist()
    inputs = inputs[:NUMBER_OF_MNIST_SAMPLES]
    targets = targets[:NUMBER_OF_MNIST_SAMPLES]

    results = run(svm.SVC(),
                 "grid_search",
                 {"kernel": ["linear", "poly", "rbf"], "C": [0.1, 1, 10]},
                 inputs,
                 targets)

    raw = [result[3] for result in results]
    df = pd.DataFrame(raw)
    df = df.transpose()
    df.columns = [f"{result[0]['kernel']}-{result[0]['C']}" for result in results]
    
    df = df[[
            "linear-0.1", "poly-0.1", "rbf-0.1",
            "linear-1", "poly-1", "rbf-1",
            "linear-10", "poly-10", "rbf-10"
    ]]
    df2 = pd.DataFrame(
        {
            "linear": df["linear-0.1"].append(df["linear-1"]).append(df["linear-10"]),
            "poly": df["poly-0.1"].append(df["poly-1"]).append(df["poly-10"]),
            "rbf": df["rbf-0.1"].append(df["rbf-1"]).append(df["rbf-10"])
        }
    )
    print(df2)
    print(df2.shape)
    print(stats.mannwhitneyu(df2["linear"], df2["poly"]))
    print(stats.mannwhitneyu(df2["poly"], df2["rbf"]))
    print(stats.mannwhitneyu(df2["linear"], df2["rbf"]))
    
    fig1, ax1 = plt.subplots()
    ax1 = df2.boxplot()
    ax1.set_title("Accuracy as a function of kernel value")
    plt.ylabel("Accuracy")
    plt.xlabel("Kernel value. n=60 for each kernel value")
    plt.savefig("plotz/yo6.png")
    """
    df2 = pd.DataFrame(
        {
            "0.1": df["linear-0.1"].append(df["poly-0.1"]).append(df["rbf-0.1"]),
            "1": df["linear-1"].append(df["poly-1"]).append(df["rbf-1"]),
            "10": df["linear-10"].append(df["poly-10"]).append(df["rbf-10"])
        }
    )
    print(df2)
    print(df2.shape)
    print(stats.mannwhitneyu(df2["0.1"], df2["1"]))
    print(stats.mannwhitneyu(df2["1"], df2["10"]))
    
    fig1, ax1 = plt.subplots()
    ax1 = df2.boxplot()
    ax1.set_title("Accuracy as a function of slack C")
    plt.ylabel("Accuracy")
    #bx.savefig("yo.png")
    plt.savefig("plotz/yo6.png")
    """
    """
    for fuckthis in results:
        my_params = fuckthis[0]
        df = fuckthis[3]
        fig1, ax1 = plt.subplots()
        ax1.set_title(f"Accuracy as a function of Slack C, {my_params['kernel']}-{my_params['C']}")
        ax1.boxplot(df)
        plt.savefig(f"plotz/{my_params['kernel']}-{my_params['C']}.png")
    """

def jgt2():
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    import matplotlib.pyplot as plt
    image_index = 7777 # You may select anything up to 60,000
    print(y_train[image_index]) # The label is 8
    plt.imshow(x_train[image_index], cmap='Greys')

if __name__=="__main__":
    jgt2()