# -*- coding: utf-8 -*-
 
def _init():#初始化
    global _global_dict
    _global_dict = {'CL_ratio':0.0}
 
def set_value(key,value):
    """ 定义一个全局变量 """
    _global_dict[key] = value

def get_value(key,defValue=None):
    """ 获得一个全局变量,不存在则返回默认值 """
    return _global_dict[key]
