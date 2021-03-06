from multiprocessing import Process
import requests, json
import pandas as pd
import time
import os
import argparse

# import definition of datasets
from lib import datasets

directory = os.getcwd()

parser = argparse.ArgumentParser()

parser.add_argument('API_KEY', type=str,
                    help='API_KEY', default='',nargs='?',const='')
parser.add_argument('API_URL', type=str,
                    help='API_URL',  nargs='?', const='',default='')
parser.add_argument('AUTO_CONF_SUPP', type=bool,
                    help='AUTO_CONF_SUPP',  nargs='?', const=False,default=False)
parser.add_argument('MAX_RULES_COUNT', type=int,
                    help='MAX_RULES_COUNT',  nargs='?', const=80000, default=80000)
parser.add_argument('PARALLEL_THREADS', type=int,
                    help='PARALLEL_THREADS',  nargs='?', const=5, default = 5)
parser.add_argument('USE_CBA', type=bool,
                    help='USE_CBA',  nargs='?', const=True, default=True)
parser.add_argument('IM_CONF', type=float,
                    help='IM_CONF',  nargs='?', const=0.5, default=0.5)
parser.add_argument('IM_SUPP', type=float,
                    help='IM_SUPP',  nargs='?', const=0.01, default=0.01)
parser.add_argument('IM_AUTO_CONF_SUPP_MAX_RULE_LENGTH', type=float,
                    help='IM_AUTO_CONF_SUPP_MAX_RULE_LENGTH',  nargs='?', const=5,default=5)

args = parser.parse_args()


if not os.path.exists("tempresult"):
    os.makedirs("tempresult")


#region import config from easyminercenter_api_config.py
if os.path.isfile(os.curdir+os.sep+'easyminercenter_api_config.py'):
    # noinspection PyUnresolvedReferences
    from easyminercenter_api_config import *
    args.API_KEY=API_KEY
    args.API_URL=API_URL
elif args.API_KEY == '' or args.API_URL == '':
    print ("You need to specify args.API_KEY and args.API_URL")
    quit()
#endregion import config from easyminercenter_api_config.py

# region config check
if (args.API_URL.endswith('/')):
    args.API_URL.rstrip('/')

check_url = args.API_URL + '/auth?apiKey=' + args.API_KEY
r = requests.get(check_url)
if (r.status_code != 200):
    print("You have to input valid args.API_KEY and URL of the EasyMinerCenter API endpoint!")
    print(check_url)
    print(r.status_code)
    quit()
# endregion config check

def write_lock(lock_file):
        with open(lock_file, 'w') as f:
            f.write("locked")

def delete_lock(lock_file):
    os.remove(lock_file)


def api_call(train,test,dataset,fold,prediction_output_file):
    print("\nprocessing " + train)
    files = {("file", open(train, 'rb'))}

    df = pd.read_csv(train)
    if "id" in dataset.keys():
        df = df.drop(dataset["id"], 1)
    # region step 1: create datasource
    headers = {"Accept": "application/json"}
    print(args.API_URL + '/datasources?separator=%2C&encoding=utf8&type=limited&apiKey=' + args.API_KEY)
    r = requests.post(args.API_URL + '/datasources?separator=%2C&encoding=utf8&type=limited&apiKey=' + args.API_KEY,
                      files=files, headers=headers)
    print ("response code:" + str(r.status_code))
    datasource_id = r.json()["id"]
    print("datasource_id:" + str(datasource_id))
    # endregion step 1: create datasource

    # region step 2: create miner
    headers = {'Content-Type': 'application/json', "Accept": "application/json"}
    json_data = json.dumps(
        {"name": "test miner " + dataset["filename"], "type": "cloud", "datasourceId": datasource_id})
    r = requests.post(args.API_URL + "/miners?apiKey=" + args.API_KEY, headers=headers, data=json_data.encode())
    miner_id = r.json()["id"]
    print("miner_id:" + str(miner_id))
    # endregion step 2: create miner

    # region step 3: preprocess fields
    attributesMap = {}
    for col in df.columns:
        json_data = json.dumps(
            {"miner": miner_id, "name": col, "columnName": col, "specialPreprocessing": "eachOne"})
        print(json_data.encode())
        r = requests.post(args.API_URL + "/attributes?apiKey=" + args.API_KEY, headers=headers, data=json_data.encode())
        print("attribute creation response status code:" + str(r.status_code))
        if r.status_code != 201:
            break
        attributesMap[col] = r.json()['name']
    # endregion step 3: preprocess fields

    # region step 4: create task
    headers = {'Content-Type': 'application/json', "Accept": "application/json"}
    consequent = attributesMap[dataset["targetvariablename"]]

    antecedent = []
    for col in df.columns:
        attribute_name = attributesMap[col]
        if attribute_name != consequent:
            antecedent.append({"attribute": attribute_name})

    task_config = {"miner": miner_id,
                   "name": "Test task", "limitHits": args.MAX_RULES_COUNT,
                   "IMs": [],
                   "specialIMs": [],
                   "antecedent": antecedent,
                   "consequent": [
                       {
                           "attribute": consequent
                       }
                   ]
                   }

    if args.AUTO_CONF_SUPP:
        task_config['IMs'].append({"name": "AUTO_CONF_SUPP"})
        task_config['IMs'].append({"name": "RULE_LENGTH", "value": args.IM_AUTO_CONF_SUPP_MAX_RULE_LENGTH})
    else:
        task_config['IMs'].append({"name": "CONF", "value": args.IM_CONF})
        task_config['IMs'].append({"name": "SUPP", "value": args.IM_SUPP})

    if args.USE_CBA:
        task_config["specialIMs"].append({"name": "CBA"})

    r = requests.post(args.API_URL + "/tasks/simple?apiKey=" + args.API_KEY, headers=headers,
                      data=json.dumps(task_config).encode())
    print("create task response code:" + str(r.status_code))
    task_id = r.json()["id"]
    print("task_id:" + str(task_id))
    # endregion step 4: create task

    # region step 5: task run
    # start task
    r = requests.get(args.API_URL + "/tasks/" + str(task_id) + "/start?apiKey=" + args.API_KEY, headers=headers)
    if r.status_code > 400:
        print("Task creation failed. Please try to modify the task config or try it later.")
        raise

    # check status task

    r = requests.get(args.API_URL + "/tasks/" + str(task_id) + "/start?apiKey=" + args.API_KEY, headers=headers)
    # task_id=r.json()["id"]
    while True:
        time.sleep(1)
        # check state
        r = requests.get(args.API_URL + "/tasks/" + str(task_id) + "/state?apiKey=" + args.API_KEY, headers=headers)
        task_state = r.json()
        print("task_state:" + task_state["state"] + ", import_state:" + task_state["importState"])
        if task_state["state"] == "solved" and task_state["importState"] == "done":
            break
        if task_state["state"] == "failed" or task_state["state"] == "interrupted":
            print(dataset["filename"] + ": task failed executing")
            raise

    # endregion step 5: task run

    print("---proceed to evaluation---")

    # region step 6: create datasource from test
    r = requests.post(args.API_URL + '/datasources?separator=%2C&encoding=utf8&type=limited&apiKey=' + args.API_KEY, files={("file", open(test, 'rb'))}, headers={"Accept": "application/json"})
    test_datasource_id = r.json()["id"]
    print("test datasource_id:" + str(test_datasource_id))
    time.sleep(1)
    # endregion step 6: create datasource from test

    # region step 7: evaluation
    uri = args.API_URL + "/evaluation/classification?scorer=easyMinerScorer&task=" + str(
        task_id) + "&datasource=" + str(test_datasource_id) + "&apiKey=" + args.API_KEY
    print("evaluation uri:" + uri)
    r = requests.get(uri, headers={"Accept": "application/json"})
    # endregion step 7: evaluation



    print("response status:" + str(r.status_code))

    # save classification result to prediction_file
    pred_output = open(prediction_output_file, "w")
    pred_output.write(r.text)
    pred_output.close()
#endregion

def train_and_test():
    #region run test tasks and evaluate partial results
    # noinspection PyUnresolvedReferences
    for dataset in datasets.datasets:
        for fold in range(0, 10):
            prediction_output_file = directory + os.sep + "tempresult" + os.sep + dataset["filename"] + str(fold) + ".evalResult.json"
            lock_file = directory + os.sep  + dataset["filename"] + str(fold) + ".lock"
            if os.path.isfile(prediction_output_file):
                print "results for " + dataset["filename"] + " fold " + str(fold) + " already available, skipping"
                continue
            if os.path.isfile(lock_file):
                print "result for " + dataset["filename"] + " fold " + str(fold) + " being computed (locked), skipping"
                continue
            else:
                write_lock(lock_file)

            train = directory + os.sep + "data" + os.sep + "folds" + os.sep + "train" + os.sep + dataset["filename"] + str(fold) + ".csv"
            test = directory + os.sep + "data" + os.sep + "folds" + os.sep + "test" + os.sep + dataset["filename"] + str(fold) + ".csv"

            if not (os.path.isfile(train) and os.path.isfile(test)):
                # train or test CSV file is not available
                print("FILE ERROR: " + train)
                continue
            try:
                api_call(train,test,dataset,fold,prediction_output_file)
            except  Exception as inst:
                print inst
                print "Giving the API time to cool down and trying again (just once)"
                try:
                    time.sleep(2)
                    api_call(train,test,dataset,fold,prediction_output_file)
                except  Exception as inst:
                    print inst
                    delete_lock(lock_file)
                    print "Second try failed, EXITING (lock file deleted)"
                    raise
            delete_lock(lock_file)
    #region process results CSV


def process_results(resultsFile):
    output = open(resultsFile, "w")
    output.write("dataset,accuracy,rules\n");
    for dataset in datasets.datasets:
        ruleCount = 0
        rowCount = 0
        correct = 0
        incorrect = 0
        unclassified = 0
        accuracyAvg = 0

        datasetResultsFile = directory + os.sep + "tempresult" + os.sep + dataset["filename"] + ".summary.txt"
        allExist=True
        for i in range(0, 10):
            resultFile = directory + os.sep + "tempresult" + os.sep + dataset["filename"] + str(i) + ".evalResult.json"
            if not os.path.isfile(resultFile):
                print("SKIPPED ROW: "+dataset["filename"])
                allExist=False
                break
        if not allExist:
            continue           
        for i in range(0, 10):
            resultFile = directory + os.sep + "tempresult" + os.sep + dataset["filename"] + str(i) + ".evalResult.json"

            jsonDataFile = open(resultFile, "r")
            data = json.load(jsonDataFile)
            jsonDataFile.close()
            dataCorrect = int(data["correct"])
            dataRowCount = int(data["rowCount"])
            ruleCount += int(data["task"]["rulesCount"])
            rowCount += dataRowCount
            correct += dataCorrect
            incorrect += int(data["incorrect"])
            unclassified += int(data["unclassified"])
            accuracyAvg+=(float(dataCorrect)/dataRowCount)
        acc_micro=float(correct) / rowCount
        acc_macro=float(accuracyAvg) / 10
        output.write(dataset["filename"] + ","                     
                     + str(acc_macro) + ","
                     + str(ruleCount / 10)
                     + "\n")

        datasetOutput = open(datasetResultsFile, "w")
        datasetOutput.write("Number of rules:" + str(ruleCount) + "\n")
        datasetOutput.write("Number of test instances:" + str(rowCount) + "\n")
        datasetOutput.write("True positives:" + str(correct) + "\n")
        datasetOutput.write("False positives:" + str(incorrect) + "\n")
        datasetOutput.write("Uncovered:" + str(unclassified) + "\n\n")
        datasetOutput.write("Accuracy (micro):" + str(acc_micro) + "\n")
        datasetOutput.write("Accuracy (macro):" + str(acc_macro) + "\n")
        datasetOutput.close()
    output.close()




#print("Deleting previously computed result files")
#for result_file in os.listdir(directory + os.sep + "tempresult"):
#    os.remove(directory + os.sep + "tempresult" + os.sep + result_file)

if __name__ == '__main__':
    processes = []
    for i in range(0,args.PARALLEL_THREADS):
        p = Process(target=train_and_test)
        p.start()
        print("started process " + str(i))
        processes.append(p)
    for p in processes:
        p.join()

#warn if any result computation hanged
files = os.listdir(directory)
for lockfile in files:
    if lockfile.endswith(".lock"):
        print ("There are the following .lock files: ")
        print (lockfile)
resultsFile = directory + os.sep + "result/rCBA-accuracy.csv"
os.remove(resultsFile)
process_results(resultsFile)
print "results written to:" + resultsFile
#endregion process results CSV
