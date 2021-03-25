import record as rec
from globals import GlobalVariable as gv
import event.event_parser as ep


def pre_process(record: rec.Record):
    src_id=record.srcId
    des_id=record.desId
    rec_id=record.Id
    subtype=record.subtype
    gv.set_event_by_id(rec_id, record)

    if subtype == 4 or subtype == 5 or subtype == 6 or subtype == 7:
        # read
        src_node = gv.get_processNode(src_id)
        des_node = gv.get_fileNode(des_id)
        src_node.add_event(rec_id)
        des_node.add_event(rec_id)
    elif subtype == 8 or subtype == 9 or subtype == 10 or subtype == 11:
        # write
        src_node = gv.get_processNode(src_id)
        des_node = gv.get_fileNode(des_id)
        src_node.add_event(rec_id)
        des_node.add_event(rec_id)
    else:
        pass


def post_train():
    for node_id in gv.processNodeSet:
        node = gv.get_processNode(node_id)
        if node:
            # event_list = node.get_event_list()
            event_list_id = node.get_event_id_list()
            # print(event_list)
            for event_id in event_list_id:

                event = gv.get_event_by_id(event_id)
                # print(event)
                if isinstance(event, rec.Record):
                    ep.EventParser.parse(event)
                
        state_sequence = node.generate_sequence(5)
        # do rnn process
        # calculate loss
        # do back propagate

    pass

