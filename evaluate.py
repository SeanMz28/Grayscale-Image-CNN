import os

# This code prints 0 if any issue occurs, otherwise prints accuracy of 'predlabels.txt' compared to ground truth
# of 'testlabels_42.text'.


def main():
    # eval_labels_name = "testlabels.txt"
    eval_labels_name = "predlabels.txt"

    try:
        os.system("python3 classifyall.py")
        os.system("python classifyall.py")
    except Exception as e:
        raise RuntimeError(f"An error in classifyall.py occurred. Exception={e}")

    if not os.path.isfile(eval_labels_name):
        raise FileNotFoundError(f"{eval_labels_name} not found")

    try:
        with open(eval_labels_name, "r") as f:
            pred_labels = f.readlines()

        with open("targetlabels.txt", "r") as f:
            test_labels = f.readlines()

        if len(pred_labels) != len(test_labels):
            print(0.0)
            return

        n_correct = 0
        n_labels = len(test_labels)
        for curr_pred_label, curr_test_label in zip(pred_labels, test_labels):
            if int(curr_pred_label) == int(curr_test_label):
                n_correct += 1
        print("Accuracy:", (n_correct/n_labels)*100)

    except FileNotFoundError as e:
        raise FileNotFoundError(e)
    except Exception as e:
        raise RuntimeError(e)
        # # Dodgy?
        # print(0.0)
        # raise RuntimeError(e)
        # # raise Va

    if os.path.exists("predlabels.txt"):
        os.remove("predlabels.txt")


if __name__ == "__main__":
    main()