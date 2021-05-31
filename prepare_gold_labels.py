from utils import readObj

malicious_id=[7681408, 7704960, 7961984, 8574336, 9117824, 9206400, 9581696, 9605760, 9721728, 9758080, 9802880, 9904256, 10360960, 10378368, 10488192, 11339136, 51257216]

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
