from data_read import readObj

def prepare_gold_labels():
    nk_malicious_ids = []
    gold_labels = []

    with open("EventData/malicious_data.txt") as nk_malicious:
        while True:
            line = nk_malicious.readline()
            if not line:
                break
            if line[:4] == "data":
                record = readObj(nk_malicious)
                nk_malicious_ids.append(record.Id)

    with open("EventData/north_korea_apt_attack_data_debug.out") as nk:
        while True:
            line = nk.readline()
            if not line:
                break
            if line[:4] == "data":
                record = readObj(nk)
                if (record.Id in nk_malicious_ids):
                    gold_labels.append("malicious")
                else:
                    gold_labels.append("benign")
    
    return gold_labels
