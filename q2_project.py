from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import preprocessing
import itertools, random
import matplotlib.pyplot as plt

random.seed('1880336')

class RandomForest:
    def __init__(self, num_models):
        self.samples = []
        self.attributes = []
        self.models = []
        self.num_models = num_models
        self.enc = None
    def encode(self, data, has_class=True):
        if self.enc is None:
            self.enc = [{} for _ in range(len(data[0]))]
        ans = []
        for row in data:
            enc_row = []
            for idx, attribute in enumerate(row[:-1]):
                if attribute not in self.enc[idx]:
                    self.enc[idx][attribute] = len(self.enc[idx])
                enc_row.append(self.enc[idx][attribute])
            if has_class:
                enc_row.append(row[-1]) # Keep class
            ans.append(enc_row)
        return ans

    def get_model_params(self, index):
        params = {}
        if index % 2 == 0: # Pre-pruning
            params["max_depth"] = None
        else: # Post-pruning
            params["ccp_alpha"] = 0.001
        return params
    
    def train(self, data_header, data):
        num_samples = len(data) # Works well for smaller datasets
        total_attributes = len(data_header)-1 # One is class
        num_attributes = int(total_attributes**0.5)+1
        self.samples = [[] for _ in range(self.num_models)]
        self.models = [None]*self.num_models
        data = self.encode(data)
        for i in range(self.num_models):
            for j in range(num_samples):
                self.samples[i].append(random.choice(data))
            attributes = random.choice([*itertools.permutations(range(total_attributes), num_attributes)])
            self.attributes.append(attributes + (total_attributes,))
            for j in range(num_samples):
                self.samples[i][j] = [x for idx,x in enumerate(self.samples[i][j]) if idx in self.attributes[-1]]
        for i in range(self.num_models):
            attributes = self.attributes[i]
            subset_header = [col for idx,col in enumerate(data_header) if idx in attributes]
            model_params = self.get_model_params(i)
            self.models[i] = DecisionTreeClassifier(**model_params)
            attributes = [row[:-1] for row in self.samples[i]]
            classes = [row[-1] for row in self.samples[i]]
            self.models[i].fit(attributes, classes)
            
            if i == 0:
                path = self.models[i].cost_complexity_pruning_path(attributes, classes)
                print(path.ccp_alphas)
    def classify(self, row):
        c = {}
        for i in range(self.num_models):
            attributes = self.encode([row], has_class=False)[0]
            data_subset = [x for idx,x in enumerate(attributes) if idx in self.attributes[i]]
            pred = self.models[i].predict([data_subset])[0]
            c[pred] = c.get(pred, 0)+1
        return max(c.items(), key=lambda x: x[1])[0]
    def print(self):
        out = ""
        for model in self.models:
            out += export_text(model)
            out += '\n\n\n'
        with open('model_out.txt', 'w') as f:
            f.write(out)

def run(num_models):
    with open('pima-indians-binned-training.csv') as f:
        lines = f.read().split('\n')
    data_header = lines[0].split(',')
    data = [line.split(',') for line in lines[1:] if line]
    for row in data:
        row[-1] = str(int(float(row[-1])))
    model = RandomForest(num_models)
    model.train(data_header, data)
    with open('pima-indians-binned-testing.csv') as f:
        lines = f.read().split('\n')
    data_header = lines[0].split(',')
    testing_data = [line.split(',') for line in lines[1:] if line]
    for row in testing_data:
        row[-1] = str(int(float(row[-1])))
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for idx,row in enumerate(testing_data):
        output = idx < 0
        if output: print()
        if output: print(row)
        prediction = model.classify(row)
        if output: print(f'Prediction: {prediction}')
        if prediction == row[-1]:
            if row[-1] == '1': true_positive += 1
            elif row[-1] == '0': true_negative += 1
            else: raise ValueError
            if output: print("Correct!")
        else:
            if row[-1] == '1': false_negative += 1
            elif row[-1] == '0': false_positive += 1
            else: raise ValueError
            if output: print("Wrong")
    print()
    print('Confusion matrix (rows predicted, columns actual):')
    print(f'{true_positive}\t | {false_positive}')
    print(f'{false_negative}\t | {true_negative}')
    accuracy = (true_positive+true_negative)/(true_positive+false_positive+false_negative+true_negative)
    print(f'Accuracy: {accuracy:.2%}')
    if true_positive+false_positive == 0: precision = 0
    else: precision = true_positive/(true_positive+false_positive)
    if true_positive+false_negative == 0: recall = 0
    else: recall = true_positive/(true_positive+false_negative)
    model.print()
    return accuracy, precision, recall

def main():
    accuracies = []
    precisions = []
    recalls = []
    x = list(range(1, 115, 2))
    for num_models in x:
        print(f'{num_models=}')
        accuracy, precision, recall = run(num_models)
        accuracies.append(accuracy * 100)
        precisions.append(precision * 100)
        recalls.append(recall * 100)
        
    plt.figure(1)
    plt.plot(x, accuracies)
    plt.xlabel('Number of models')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)

    plt.figure(2)
    plt.plot(x, precisions)
    plt.xlabel('Number of models')
    plt.ylabel('Precision (%)')
    plt.ylim(0, 100)

    plt.figure(3)
    plt.plot(x, recalls)
    plt.xlabel('Number of models')
    plt.ylabel('Recall (%)')
    plt.ylim(0, 100)

    plt.show()
    
if __name__ == "__main__":
    main()
