from typing import Dict
from processnode import ProcessNode
from filenode import FileNode



class GlobalVariable:
    quote: str = "hello"
    fileNodeSet: Dict[int, ProcessNode] = {}
    processNodeSet: Dict[int, FileNode] = {}


    @classmethod
    def set_qupte(cls, quote):
        cls.quote=quote

    @classmethod
    def get_quote(cls):
        return cls.quote

    @classmethod
    def get_fileNode(cls, index: int)-> FileNode:
        return cls.fileNodeSet[index] if index in cls.fileNodeSet else None

    @classmethod
    def set_fileNode(cls, index: int, node: FileNode):
        cls.fileNodeSet[index]=node

    @classmethod
    def exist_fileNode(cls, index: int)-> bool:
        return True if index in cls.fileNodeSet else False


    @classmethod
    def get_processNode(cls, index: int) -> ProcessNode:
        return cls.processNodeSet[index] if index in cls.processNodeSet else None

    @classmethod
    def set_processNode(cls, index: int, node: ProcessNode):
        cls.processNodeSet[index] = node

    @classmethod
    def exist_processNode(cls, index: int) -> bool:
        return True if index in cls.processNodeSet else False

