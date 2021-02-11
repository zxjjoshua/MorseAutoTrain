from record import Record
import numpy as np

class EventParser:
    def parse(self, record: Record):
        if record.subtype == 1:
            pass
        elif record.subtype == 2:
            pass
        elif record.subtype == 3:
            pass
        elif record.subtype == 4:
            pass
        elif record.subtype == 5:
            pass
        elif record.subtype == 6:
            pass
        elif record.subtype == 7:
            pass
        elif record.subtype == 8:
            pass
        elif record.subtype == 9:
            pass
        elif record.subtype == 10:
            pass
        elif record.subtype == 11:
            pass
        elif record.subtype == 12:
            pass
        elif record.subtype == 13:
            pass
        elif record.subtype == 14:
            pass
        elif record.subtype == 15:
            pass
        elif record.subtype == 16:
            pass
        elif record.subtype == 17:
            pass
        elif record.subtype == 18:
            pass
        elif record.subtype == 19:
            pass
        elif record.subtype == 20:
            pass
        elif record.subtype == 21:
            pass
        elif record.subtype == 22:
            pass
        elif record.subtype == 23:
            pass
        elif record.subtype == 24:
            pass
        elif record.subtype == 25:
            pass
        elif record.subtype == 26:
            pass
        elif record.subtype == 27:
            pass
        elif record.subtype == 28:
            pass
        elif record.subtype == 29:
            pass
        elif record.subtype == 30:
            pass
        elif record.subtype == 31:
            pass
        elif record.subtype == 32:
            pass
        elif record.subtype == 33:
            pass
        elif record.subtype == 34:
            pass
        elif record.subtype == 35:
            pass
        elif record.subtype == 36:
            pass
        elif record.subtype == 37:
            pass

    def parse_sys_create(self, record: Record):
        if record.subtype!=27:
            return None
        if record.srcid==-1:
            return None
        srcid=record.srcId
        desid=record.desId



