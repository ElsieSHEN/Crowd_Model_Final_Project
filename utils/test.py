import torch

def test(model, testloader):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Total Accuracy: %2d %%' % (100 * sum(class_correct) / sum(class_total)))

    for i in range(10): 
        print('Accuracy of %5s : %2f %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))