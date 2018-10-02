
import numpy as np
from sklearn import linear_model
from sklearn import svm

def fizz_buzz_game(dataset):
    game_result = []
    for i in range(dataset.shape[0]):
        if dataset[i] % 3 ==0 and dataset[i]!=0:
            if dataset[i]% 5==0 and dataset[i] !=0:
                game_result.append('fizzbuzz')
            else:
                game_result.append('fizz')
        elif dataset[i] % 5==0 and dataset[i]!=0:
             if dataset[i]%3 ==0 and dataset[i]!=0:
                 game_result.append('fizzbuzz')
             else:
                 game_result.append('buzz')
        else:
            game_result.append(dataset[i])
    return game_result


def traditional_main():
    input_dataset = np.arange(1,101,1)
    print(input_dataset)
    output_dataset = fizz_buzz_game(input_dataset)
    print(output_dataset)

def feature_rnginer(dataset):
    train_dataset = np.zeros((dataset.shape[0],3),dtype=int)
    for i in range(dataset.shape[0]):
         train_dataset[i,:] =[dataset[i]%3 ,dataset[i]%5 , dataset[i]%15]
    print('train dataset is ',train_dataset)
    return train_dataset

def fizzbuzz_label(dataset):
    label = []
    for i in range(dataset.shape[0]):
        if dataset[i] % 15 == 0:
            label.append(30)
        elif dataset[i] % 3 ==0:
            label.append(10)
        elif dataset[i]% 5 == 0:
            label.append(20)
        else:
            label.append(0)
    label= np.asarray(label)
    print('the train label is',label)
    return label


def fizzbuzz_predict_to_label(valid_Y,valid_dataset):
        pred_label = []
        if valid_Y.shape == valid_dataset.shape:
            for i in range(valid_Y.shape[0]):
                if valid_Y[i] == 10:
                    pred_label.append('fizz')
                elif valid_Y[i] ==20:
                    pred_label.append('buzz')
                elif valid_Y[i] == 30:
                    pred_label.append('fizzbuzz')
                else:
                    pred_label.append(valid_dataset[i])
            pred_label = np.asarray(pred_label)
            return pred_label
        else:
            print('data shape error')


def machinelearning_main():
    train_dataset_X = np.arange(101,201,1)
    train_X=feature_rnginer(train_dataset_X)
    train_Y = fizzbuzz_label(train_dataset_X)

    test_dataset = np.arange(201,301,1)
    test_X = feature_rnginer(test_dataset)
    test_Y = fizzbuzz_label(test_dataset)
    valid_dataset = np.arange(1,101,1)
    valid_X = feature_rnginer(valid_dataset)
    logistic = linear_model.LogisticRegression()
    svc = svm.SVC()
    logistic.fit(train_X,train_Y)
    svc.fit(train_X,train_Y)
    # 代表模型精准程度
    print('LogisticRegression score: %f',logistic.score(test_X, test_Y))
    print('svm score: %f', svc.score(test_X, test_Y))
    valid_pred = logistic.predict(valid_X)
    valid_pred_svm = svc.predict(valid_X)
    valid_pred_label = fizzbuzz_predict_to_label(valid_pred,valid_dataset)
    valid_pred_label_svm = fizzbuzz_predict_to_label(valid_pred_svm,valid_dataset)
    print('the predivt value is :%f ', valid_pred_label)
    print('the svm predivt value is :%f ', valid_pred_label_svm)


#traditional_main()
machinelearning_main()